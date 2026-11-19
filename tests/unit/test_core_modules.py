"""Unit tests: errors, offline guard, integrity verification, data loading, text, model loader (fake library).

No real model, no private data, no network.
"""

import importlib.abc
import json
import logging
import os
import socket
import subprocess
import sys
import types

import numpy as np
import pytest

from grc_mapper import errors, integrity, offline
from grc_mapper.data import Statement, load_control_catalog, load_statements, natural_key
from grc_mapper.errors import (
    ConfigError, GrcError, InputValidationError, IntegrityError, ModelLoadError, OfflineViolation,
)
from grc_mapper.integrity import (
    load_full_manifest, load_runtime_manifest, require_full_verification, verify_model_dir,
)
from grc_mapper.text import TokenProfile, text_fingerprint, validate_statement_text

pytestmark = pytest.mark.unit
SECRET = "ZQX-SENTINEL policy wording that must never leak"


# ------------------------------------------------------------------ errors

def test_exit_codes():
    assert (ConfigError.exit_code, IntegrityError.exit_code, InputValidationError.exit_code,
            ModelLoadError.exit_code, OfflineViolation.exit_code, errors.OutputWriteError.exit_code) == (2, 3, 4, 5, 5, 6)
    assert issubclass(OfflineViolation, ModelLoadError) and issubclass(ModelLoadError, GrcError)


# ------------------------------------------------------------------ offline

def test_enforce_offline_env_sets_all_vars():
    env = {}
    applied = offline.enforce_offline_env(env)
    assert env == applied == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                              "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}


@pytest.mark.parametrize("call", [
    lambda: socket.create_connection(("127.0.0.1", 9)),
    lambda: socket.socket().connect(("127.0.0.1", 9)),
    lambda: socket.socket().connect_ex(("127.0.0.1", 9)),
])
def test_guard_blocks_each_entry_point(call):
    with offline.network_blocked():
        with pytest.raises(OfflineViolation):
            call()


def test_guard_restores_after_exception_and_nesting():
    orig = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)
    with pytest.raises(RuntimeError):
        with offline.network_blocked():
            with offline.network_blocked():
                pass
            assert socket.create_connection is not orig[2]
            raise RuntimeError("boom")
    assert (socket.socket.connect, socket.socket.connect_ex, socket.create_connection) == orig


# ------------------------------------------------------------------ integrity

def _load(rt, full):
    return load_runtime_manifest(rt), load_full_manifest(full)


def test_ok_full(fake_model):
    model, rt, full = fake_model
    rep = verify_model_dir(model, *_load(rt, full))
    assert rep.status == "OK_FULL" and rep.runtime_files_verified == 8
    assert dict(rep.non_runtime) == {"README.md": "verified",
                                     "eval/similarity_evaluation_train_dev_results.csv": "verified"}
    require_full_verification(rep)


def test_ok_partial_is_diagnostic_only(fake_model):
    model, rt, _ = fake_model
    rep = verify_model_dir(model, load_runtime_manifest(rt), None)
    assert rep.status == "OK_PARTIAL"
    assert set(dict(rep.non_runtime).values()) == {"present_unverified"}
    with pytest.raises(IntegrityError, match="full model manifest required"):
        require_full_verification(rep)


def test_non_runtime_absent_is_allowed(fake_model):
    model, rt, full = fake_model
    (model / "README.md").unlink()
    rep = verify_model_dir(model, *_load(rt, full))
    assert rep.status == "OK_FULL" and dict(rep.non_runtime)["README.md"] == "absent"


@pytest.mark.parametrize("mutate,msg", [
    (lambda m: (m / "config.json").unlink(), "runtime file missing: config.json"),
    (lambda m: (m / "config.json").write_bytes(b'{"model_type": "bert"} '), "size mismatch: config.json"),
    (lambda m: (m / "config.json").write_bytes(b'{"model_type": "xert"}'), "sha256 mismatch: config.json"),
    (lambda m: (m / "eval" / "other.csv").write_bytes(b"x"), "unexpected file: eval/other.csv"),
    (lambda m: (m / "pytorch_model.bin").write_bytes(b"x"), "unexpected file: pytorch_model.bin"),
    (lambda m: (m / "2_Normalize" / "x.json").write_bytes(b"{}"), "directory must be empty: 2_Normalize/"),
    (lambda m: (m / "extra_dir").mkdir(), "unexpected directory: extra_dir/"),
    (lambda m: (m / "README.md").write_bytes(b"# changed, longer readme\n"), "non-runtime file size mismatch: README.md"),
])
def test_verification_failures(fake_model, mutate, msg):
    model, rt, full = fake_model
    mutate(model)
    with pytest.raises(IntegrityError, match=msg.replace("/", "/").replace(".", r"\.")):
        verify_model_dir(model, *_load(rt, full))


# Windows error 1314 = ERROR_PRIVILEGE_NOT_HELD: creating symlinks needs Developer Mode or admin rights.
_SYMLINK_UNAVAILABLE = (
    "symbolic-link creation is unavailable on this machine "
    "(Windows needs Developer Mode or the SeCreateSymbolicLinkPrivilege; WinError 1314)"
)


def _make_symlink(link, target, *, is_dir=False):
    """Create a symlink or skip the calling test with an explicit reason if the OS refuses.

    Only a privilege/permission refusal skips. Any other OSError fails the test.
    """
    if not hasattr(os, "symlink"):
        pytest.skip("os.symlink is not available on this platform")
    try:
        os.symlink(target, link, target_is_directory=is_dir)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314 or isinstance(exc, (PermissionError, NotImplementedError)):
            pytest.skip(_SYMLINK_UNAVAILABLE)
        raise
    assert os.path.islink(link), "symlink creation reported success but no link exists"


def test_symlink_helper_skips_only_on_privilege_error(monkeypatch, tmp_path):
    def refuse(*a, **k):
        e = OSError(22, "A required privilege is not held by the client")
        e.winerror = 1314
        raise e
    monkeypatch.setattr(os, "symlink", refuse)
    with pytest.raises(pytest.skip.Exception, match="WinError 1314"):
        _make_symlink(tmp_path / "l", tmp_path / "t")


def test_symlink_helper_fails_on_other_errors(monkeypatch, tmp_path):
    def broken(*a, **k):
        raise OSError(5, "some other failure")
    monkeypatch.setattr(os, "symlink", broken)
    with pytest.raises(OSError, match="some other failure"):
        _make_symlink(tmp_path / "l", tmp_path / "t")


def test_symlink_file_inside_model_dir_rejected(fake_model, tmp_path):
    model, rt, full = fake_model
    target = tmp_path / "outside.txt"
    target.write_bytes(b"x")
    _make_symlink(model / "link.txt", target)
    with pytest.raises(IntegrityError, match="symlink, junction or reparse point not allowed: link.txt"):
        verify_model_dir(model, *_load(rt, full))


def test_symlink_model_root_rejected(fake_model, tmp_path):
    model, rt, full = fake_model
    link_root = tmp_path / "model_link"
    _make_symlink(link_root, model, is_dir=True)
    with pytest.raises(IntegrityError, match="must not be a symlink"):
        verify_model_dir(link_root, *_load(rt, full))


def _make_junction(link, target):
    """Create an NTFS directory junction (no special privilege needed) or skip off Windows."""
    if sys.platform != "win32":
        pytest.skip("directory junctions are Windows-only")
    try:
        import _winapi
    except ImportError:
        pytest.skip("_winapi is not available")
    if not hasattr(_winapi, "CreateJunction"):
        pytest.skip("_winapi.CreateJunction is not available in this Python build")
    _winapi.CreateJunction(str(target), str(link))  # any failure here fails the test
    st = os.lstat(link)
    assert getattr(st, "st_file_attributes", 0) & 0x400, "junction created without the reparse-point attribute"


def test_junction_inside_model_dir_rejected(fake_model, tmp_path):
    model, rt, full = fake_model
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "x.json").write_bytes(b"{}")
    _make_junction(model / "eval_junction", outside)
    with pytest.raises(IntegrityError, match="symlink, junction or reparse point not allowed: eval_junction"):
        verify_model_dir(model, *_load(rt, full))


def test_junction_model_root_rejected(fake_model, tmp_path):
    model, rt, full = fake_model
    link_root = tmp_path / "model_junction"
    _make_junction(link_root, model)
    with pytest.raises(IntegrityError, match="must not be a symlink, junction or reparse point"):
        verify_model_dir(link_root, *_load(rt, full))


def test_full_manifest_must_match_runtime(fake_model, tmp_path):
    model, rt, full = fake_model
    obj = json.loads(full.read_text())
    obj["runtime_files"][0]["size"] += 1
    bad = tmp_path / "bad_full.json"
    bad.write_text(json.dumps(obj))
    with pytest.raises(IntegrityError, match="runtime_files differ"):
        verify_model_dir(model, load_runtime_manifest(rt), load_full_manifest(bad))


@pytest.mark.parametrize("edit,msg", [
    (lambda o: o["files"].append(dict(o["files"][0])), "duplicate path"),
    (lambda o: o["files"][0].update(path="../x"), r"'\.\.' segment"),
    (lambda o: o["files"][0].update(path="C:/x"), "must be relative"),
    (lambda o: o["files"][0].update(path="a\\b"), "must be relative"),
    (lambda o: o["files"][0].update(sha256="ABC"), "64 lower-case hex"),
    (lambda o: o["files"][0].update(size=-1), "non-negative"),
    (lambda o: o["files"].append({"path": "README.md", "size": 1, "sha256": "a" * 64}), "non-runtime paths"),
    (lambda o: o.update(schema_version=2), "schema_version"),
    (lambda o: o["files"][0].update(extra=1), "exactly the keys"),
])
def test_runtime_manifest_rejects(fake_model, tmp_path, edit, msg):
    _, rt, _ = fake_model
    obj = json.loads(rt.read_text())
    edit(obj)
    p = tmp_path / "rt2.json"
    p.write_text(json.dumps(obj))
    with pytest.raises(IntegrityError, match=msg):
        load_runtime_manifest(p)


def test_full_manifest_closed_non_runtime_list(fake_model, tmp_path):
    _, _, full = fake_model
    obj = json.loads(full.read_text())
    obj["non_runtime_files"].append({"path": "eval/other.csv", "size": 1, "sha256": "b" * 64})
    p = tmp_path / "f2.json"
    p.write_text(json.dumps(obj))
    with pytest.raises(IntegrityError, match="outside the closed list"):
        load_full_manifest(p)


def test_reverify_after_load_passes_when_unchanged(fake_model):
    from grc_mapper.integrity import reverify_after_load
    model, rt, full = fake_model
    r, f = _load(rt, full)
    rep = verify_model_dir(model, r, f)
    assert reverify_after_load(model, r, f, rep).status == "OK_FULL"


@pytest.mark.parametrize("mutate,msg", [
    # same size, different bytes: only a fresh SHA-256 can detect these
    (lambda m: (m / "README.md").write_bytes(b"# SYNTHETIC readme\n"), "non-runtime file sha256 mismatch: README.md"),
    (lambda m: (m / "model.safetensors").write_bytes(b"\x00\x01FAKE-weights\x02"), "runtime file sha256 mismatch: model.safetensors"),
    (lambda m: (m / "eval" / "late.csv").write_bytes(b"x"), "unexpected file: eval/late.csv"),
])
def test_reverify_after_load_detects_change(fake_model, mutate, msg):
    from grc_mapper.integrity import reverify_after_load
    model, rt, full = fake_model
    r, f = _load(rt, full)
    rep = verify_model_dir(model, r, f)
    mutate(model)
    with pytest.raises(IntegrityError, match="changed during load") as ei:
        reverify_after_load(model, r, f, rep)
    assert msg in str(ei.value)


def test_reverify_after_load_detects_mtime_only_change(fake_model):
    from grc_mapper.integrity import reverify_after_load
    model, rt, full = fake_model
    r, f = _load(rt, full)
    rep = verify_model_dir(model, r, f)
    st = os.stat(model / "config.json")
    os.utime(model / "config.json", ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    with pytest.raises(IntegrityError, match=r"size or modification time\): \['config.json'\]"):
        reverify_after_load(model, r, f, rep)


def test_reverify_requires_full_manifest(fake_model):
    from grc_mapper.integrity import reverify_after_load
    model, rt, full = fake_model
    r, f = _load(rt, full)
    rep = verify_model_dir(model, r, f)
    with pytest.raises(IntegrityError, match="full model manifest required"):
        reverify_after_load(model, r, None, rep)


def test_integrity_and_data_never_import_sentence_transformers():
    code = ("import sys; import grc_mapper.integrity, grc_mapper.data, grc_mapper.text, "
            "grc_mapper.offline, grc_mapper.model; "
            "bad=[m for m in ('sentence_transformers','transformers','torch') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    from _support import child_env
    r = subprocess.run([sys.executable, "-c", code], env=child_env(), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ------------------------------------------------------------------ text

def test_validate_returns_same_object():
    s = "  Access must be reviewed quarterly.  "
    assert validate_statement_text(s, item_id="P1") is s


@pytest.mark.parametrize("bad", [None, 3, b"bytes", "", "   \n\t", "12345 - 6.7", "___"])
def test_validate_rejects(bad):
    with pytest.raises(InputValidationError):
        validate_statement_text(bad, item_id="P1")


def test_validate_error_has_no_text():
    with pytest.raises(InputValidationError) as ei:
        validate_statement_text(12, item_id="P9")
    assert "P9" in str(ei.value)


def test_token_profile_invariants():
    assert TokenProfile(257, 256, True).truncated
    for args in [(256, 257, False), (257, 256, False), (10, 10, True), (-1, 0, False), (True, 1, False)]:
        with pytest.raises(ValueError):
            TokenProfile(*args)


def test_fingerprint():
    h, n = text_fingerprint("abc")
    assert n == 3 and h == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# ------------------------------------------------------------------ data

def _csv(tmp_path, name, text, encoding="utf-8"):
    p = tmp_path / name
    p.write_bytes(text.encode(encoding))
    return p


CTRL = "Control_ID,Control_Name,Control_Text\nA.5.10,x,Ten text\nA.5.2,y,Two text\nA.8.1,z,Eight text\n"


def test_controls_natural_order_and_verbatim(tmp_path):
    p = _csv(tmp_path, "c.csv", CTRL.replace("Two text", '" Two text, padded "'))
    cat = load_control_catalog(p, expected_count=3)
    assert cat.control_ids == ("A.5.2", "A.5.10", "A.8.1")
    assert cat.texts[0] == " Two text, padded "
    assert len(cat.source_sha256) == 64 and cat.source_kind == "research"


def test_controls_bom_and_crlf(tmp_path):
    p = _csv(tmp_path, "c.csv", "\ufeff" + CTRL.replace("\n", "\r\n"))
    assert len(load_control_catalog(p)) == 3


@pytest.mark.parametrize("content,msg", [
    ("Control_ID,Name\nA.1,x\n", "missing required columns"),
    ("Control_ID,Control_Text\n,Text\n", "blank control IDs"),
    ("Control_ID,Control_Text\n A.1,Text\n", "leading/trailing whitespace"),
    ("Control_ID,Control_Text\nA.1,Text\nA.1,Other\n", "duplicate control IDs"),
    (f"Control_ID,Control_Text\nA.1,   \n", "empty or whitespace-only"),
    ("Control_ID,Control_Text\nA.1,123\n", "contains no letters"),
    ("Control_ID,Control_Text\nA.1,Text,extra\n", "wrong number of fields"),
    ("Control_ID,Control_Text,Control_ID\nA.1,T,A\n", "duplicate column names"),
    ("Control_ID,,Control_Text\nA.1,,T\n", "blank column names"),
    ("", "empty"),
    ("Control_ID,Control_Text\n", "no control rows"),
])
def test_controls_rejects(tmp_path, content, msg):
    with pytest.raises(InputValidationError, match=msg):
        load_control_catalog(_csv(tmp_path, "c.csv", content))


def test_controls_count_and_kind(tmp_path):
    p = _csv(tmp_path, "c.csv", CTRL)
    with pytest.raises(InputValidationError, match="expected 93 controls, found 3"):
        load_control_catalog(p, expected_count=93)
    with pytest.raises(ConfigError):
        load_control_catalog(p, source_kind="other")


def test_invalid_utf8(tmp_path):
    p = tmp_path / "c.csv"
    p.write_bytes(b"Control_ID,Control_Text\nA.1,caf\xe9\n")
    with pytest.raises(InputValidationError, match="not valid UTF-8"):
        load_control_catalog(p)


def test_missing_file(tmp_path):
    with pytest.raises(InputValidationError, match="file not found: nope.csv"):
        load_control_catalog(tmp_path / "nope.csv")


def test_statements_and_labels(tmp_path):
    cat = load_control_catalog(_csv(tmp_path, "c.csv", CTRL))
    s = _csv(tmp_path, "s.csv", "Policy_ID,Policy_Text,True_Control_ID\nP2,Second,A.5.2\nP1,First,A.8.1\n")
    stmts, labels = load_statements(s, id_column="Policy_ID", text_column="Policy_Text",
                                    label_column="True_Control_ID", known_controls=cat, expected_count=2)
    assert [x.statement_id for x in stmts] == ["P2", "P1"]  # file order kept
    assert labels == {"P2": "A.5.2", "P1": "A.8.1"}
    stmts2, none = load_statements(s, id_column="Policy_ID", text_column="Policy_Text")
    assert none is None and len(stmts2) == 2


def test_statement_label_errors(tmp_path):
    cat = load_control_catalog(_csv(tmp_path, "c.csv", CTRL))
    s = _csv(tmp_path, "s.csv", "Policy_ID,Policy_Text,True_Control_ID\nP1,First,A.9.9\n")
    with pytest.raises(InputValidationError, match="not in the control catalogue"):
        load_statements(s, id_column="Policy_ID", text_column="Policy_Text",
                        label_column="True_Control_ID", known_controls=cat)
    s2 = _csv(tmp_path, "s2.csv", "Policy_ID,Policy_Text,True_Control_ID\nP1,First,\n")
    with pytest.raises(InputValidationError, match="blank or padded labels"):
        load_statements(s2, id_column="Policy_ID", text_column="Policy_Text",
                        label_column="True_Control_ID", known_controls=cat)
    with pytest.raises(ConfigError):
        load_statements(s, id_column="Policy_ID", text_column="Policy_Text", label_column="True_Control_ID")


def test_errors_and_repr_never_contain_text(tmp_path):
    p = _csv(tmp_path, "s.csv", f'Policy_ID,Policy_Text\nP1,"{SECRET}"\nP1,"{SECRET}"\n')
    with pytest.raises(InputValidationError) as ei:
        load_statements(p, id_column="Policy_ID", text_column="Policy_Text")
    assert SECRET not in str(ei.value)
    assert SECRET not in repr(Statement("P1", SECRET))


def test_natural_key():
    ids = ["A.8.10", "A.5.2", "A.5.10", "A.5.1", "A.8.9"]
    assert sorted(ids, key=natural_key) == ["A.5.1", "A.5.2", "A.5.10", "A.8.9", "A.8.10"]


# ------------------------------------------------------------------ model (fake library)

class _ImportSpy(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.seen = []

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("sentence_transformers", "transformers", "torch"):
            self.seen.append(name)
        return None


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens=True, truncation=False, verbose=True):
        ids = [5] * len(text.split())
        return {"input_ids": ([101] + ids + [102]) if add_special_tokens else ids}


class _FakeST:
    instances = []
    attempt_network = False
    dim = 384
    max_seq = 256
    scale = 1.0
    mutate_during_load = None  # callable(model_path) run inside the fake load

    def __init__(self, path, **kwargs):
        if _FakeST.mutate_during_load is not None:
            _FakeST.mutate_during_load(path)
        self.kwargs = kwargs
        self.env = {k: os.environ.get(k) for k in offline.OFFLINE_ENV}
        try:
            socket.create_connection(("127.0.0.1", 9))
            self.guarded = False
        except OfflineViolation:
            self.guarded = True
            if _FakeST.attempt_network:
                raise
        self.tokenizer = _FakeTokenizer()
        self.max_seq_length = _FakeST.max_seq
        _FakeST.instances.append(self)

    def get_sentence_embedding_dimension(self):
        return _FakeST.dim

    def encode(self, items, **kw):
        out = np.zeros((len(items), 384), dtype=np.float32)
        out[:, 0] = _FakeST.scale
        return out

    def tokenize(self, texts):
        n = len(self.tokenizer(texts[0])["input_ids"])
        return {"input_ids": np.zeros((1, min(n, self.max_seq_length)), dtype=np.int64)}


@pytest.fixture
def fake_st(monkeypatch):
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    for k in offline.OFFLINE_ENV:
        monkeypatch.delenv(k, raising=False)
    _FakeST.instances.clear()
    _FakeST.attempt_network = False
    _FakeST.mutate_during_load = None
    _FakeST.dim, _FakeST.max_seq, _FakeST.scale = 384, 256, 1.0
    return _FakeST


def _load_model(fake_model, **kw):
    from grc_mapper.model import load_verified_model
    model, rt, full = fake_model
    return load_verified_model(model, runtime_manifest_path=rt, full_manifest_path=full, **kw)


def test_load_success_offline_kwargs(fake_model, fake_st):
    enc = _load_model(fake_model)
    inst = fake_st.instances[0]
    assert inst.kwargs == {"device": "cpu", "local_files_only": True, "trust_remote_code": False}
    assert inst.guarded is True
    assert set(inst.env.values()) == {"1"}
    assert enc.kind == "SBERT_PRODUCTION" and enc.report.status == "OK_FULL"
    assert enc.embedding_dim == 384 and enc.max_seq_length == 256
    assert socket.create_connection.__name__ != "_blocked"  # guard removed after load


def test_full_manifest_required_and_no_import(fake_model, monkeypatch):
    from grc_mapper.model import load_verified_model
    spy = _ImportSpy()
    monkeypatch.setattr(sys, "meta_path", [spy] + sys.meta_path)
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    model, rt, full = fake_model
    with pytest.raises(ConfigError, match="full manifest path is required"):
        load_verified_model(model, runtime_manifest_path=rt, full_manifest_path=None)
    (model / "config.json").write_bytes(b"tampered!!!!!!!!!!!!!!")
    with pytest.raises(IntegrityError):
        load_verified_model(model, runtime_manifest_path=rt, full_manifest_path=full)
    assert spy.seen == []


def test_ok_partial_cannot_load(fake_model, fake_st, monkeypatch):
    import grc_mapper.model as m
    model, rt, _ = fake_model
    monkeypatch.setattr(m, "load_full_manifest", lambda p: None)  # simulate a missing full manifest
    with pytest.raises(IntegrityError, match="full model manifest required"):
        m.load_verified_model(model, runtime_manifest_path=rt, full_manifest_path=rt)
    assert fake_st.instances == []


@pytest.mark.parametrize("attr,value,msg", [
    ("dim", 768, "embedding dimension 768"),
    ("max_seq", 512, "max_seq_length 512"),
    ("scale", 2.0, "not unit-normalised"),
])
def test_post_load_checks(fake_model, fake_st, attr, value, msg):
    setattr(fake_st, attr, value)
    with pytest.raises(ModelLoadError, match=msg):
        _load_model(fake_model)


def test_same_size_change_during_load_detected_by_rehash(fake_model, fake_st):
    from pathlib import Path
    fake_st.mutate_during_load = lambda p: (Path(p) / "README.md").write_bytes(b"# SYNTHETIC readme\n")
    with pytest.raises(IntegrityError, match="changed during load"):
        _load_model(fake_model)


def test_network_attempt_during_load(fake_model, fake_st):
    fake_st.attempt_network = True
    with pytest.raises(OfflineViolation):
        _load_model(fake_model)


def test_invalid_device(fake_model, fake_st):
    with pytest.raises(ConfigError):
        _load_model(fake_model, device="tpu")


def test_token_profile_and_logging(fake_model, fake_st, caplog):
    from grc_mapper.model import profile_texts
    enc = _load_model(fake_model)
    t254 = " ".join(["word"] * 254)
    t255 = " ".join(["word"] * 255)
    assert enc.token_profile(t254) == TokenProfile(256, 256, False)
    assert enc.token_profile(t255) == TokenProfile(257, 256, True)
    with caplog.at_level(logging.INFO, logger="grc_mapper.model"):
        profs = profile_texts(enc, ["C1", "P1"], [t254, SECRET + " " + t255], kind="statement")
    assert [p.truncated for p in profs] == [False, True]
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "P1 truncated" in joined and SECRET not in joined and "word" not in joined


def test_encode_contract(fake_model, fake_st):
    enc = _load_model(fake_model)
    assert enc.encode([], batch_size=4).shape == (0, 384)
    out = enc.encode(["a b", "c"], batch_size=1)
    assert out.dtype == np.float32 and out.shape == (2, 384)
    with pytest.raises(ConfigError):
        enc.encode(["a"], batch_size=0)
