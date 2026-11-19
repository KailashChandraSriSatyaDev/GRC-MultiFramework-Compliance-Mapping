"""Unit tests for tools/check_publishable.py (U-SN-1 ... U-SN-10). Temporary trees only.

Personal-path and identity strings are assembled from pieces so this file itself stays clean.
"""

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from _support import REPO_ROOT, child_env

SPEC = importlib.util.spec_from_file_location("check_publishable", REPO_ROOT / "tools" / "check_publishable.py")
cp = importlib.util.module_from_spec(SPEC)
sys.modules["check_publishable"] = cp
SPEC.loader.exec_module(cp)

@pytest.fixture(autouse=True)
def _isolate_from_real_configuration(monkeypatch):
    """The scanner's --config defaults to GRC_CONFIG; tests must never inherit the real workspace config."""
    monkeypatch.delenv("GRC_CONFIG", raising=False)


LOCK = (REPO_ROOT / "requirements" / "grc-mvp.lock.txt").read_bytes()
WIN_USER = "C:" + "\\" + "Users" + "\\" + "someone" + "\\" + "x"
TOKEN = "Zq" + "xIdentity"
CONTROL_TEXT = "the quick brown fox jumps over the lazy dog near the riverbank today"


def _tree(tmp_path):
    root = tmp_path / "pub"
    (root / "requirements").mkdir(parents=True)
    (root / "requirements" / "grc-mvp.lock.txt").write_bytes(LOCK)
    (root / "README.md").write_text("# demo\nclean text\n", encoding="utf-8")
    (root / "src" / "grc_mapper").mkdir(parents=True)
    (root / "src" / "grc_mapper" / "a.py").write_text("x = 1\n", encoding="utf-8")
    return root


def _local(tmp_path, *, tokens=True, texts=True, registry=True, blocked=(), approved=()):
    loc = tmp_path / "local"
    loc.mkdir(exist_ok=True)
    ctrl = loc / "c.csv"
    ctrl.write_text(f"Control_ID,Control_Text\nA.1,{CONTROL_TEXT}\n", encoding="utf-8")
    pol = loc / "p.csv"
    pol.write_text("Policy_ID,Policy_Text\nP1,completely different sentence about other things here\n", encoding="utf-8")
    reg = loc / "r.json"
    reg.write_text(json.dumps({"blocked": [{"sha256": h} for h in blocked], "approved_runtime_hashes": list(approved)}))
    lines = ["[paths]"]
    if texts:
        lines += [f"controls_csv = '{ctrl}'", f"policies_csv = '{pol}'"]
    if registry:
        lines += [f"restricted_hashes = '{reg}'"]
    lines += ["[columns]", "control_text = 'Control_Text'", "statement_text = 'Policy_Text'", "[scanner]"]
    if tokens:
        lines += [f"identity_tokens = ['{TOKEN}']"]
    cfg = loc / "local.toml"
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cp.load_local_inputs(cfg), cfg


def test_sn1_clean_tree_passes(tmp_path):
    root = _tree(tmp_path)
    local, _ = _local(tmp_path)
    rep = cp.scan(root, local)
    assert rep["overall"] == "PASS", rep["checks"]


def test_sn2_incomplete_without_local_inputs(tmp_path):
    root = _tree(tmp_path)
    rep = cp.scan(root, cp.load_local_inputs(None))
    assert rep["overall"] == "INCOMPLETE"
    assert {c for c, r in rep["checks"].items() if r["status"] == "NOT_RUN"} == {"S4", "S5", "S6"}


def test_sn3_final_mode_requires_config_and_pass(tmp_path):
    root = _tree(tmp_path)
    assert cp.main(["--root", str(root), "--final"]) == 2
    _, cfg = _local(tmp_path)
    assert cp.main(["--root", str(root), "--final", "--config", str(cfg), "--report-dir", str(tmp_path / "rep")]) == 0
    reports = list((tmp_path / "rep").rglob("scan_report.json"))
    assert len(reports) == 1 and json.loads(reports[0].read_text())["overall"] == "PASS"
    assert cp.main(["--root", str(root), "--final", "--config", str(cfg)]) == 2  # report not saved -> not PASS
    assert cp.main(["--root", str(root), "--config", str(cfg), "--report-dir", str(root / "inside")]) == 2


def test_sn3b_final_mode_configuration_resolution(tmp_path, monkeypatch, capsys):
    root = _tree(tmp_path)
    _, cfg = _local(tmp_path)
    rep = str(tmp_path / "rep")
    monkeypatch.setenv("GRC_CONFIG", "   ")                       # empty value counts as missing
    assert cp.main(["--root", str(root), "--final", "--report-dir", rep]) == 2
    monkeypatch.setenv("GRC_CONFIG", str(tmp_path / "missing.toml"))  # named file absent
    assert cp.main(["--root", str(root), "--final", "--report-dir", rep]) == 2
    assert "configuration file not found" in capsys.readouterr().err
    bad = tmp_path / "bad.toml"
    bad.write_text("not = [valid", encoding="utf-8")                # unreadable TOML
    assert cp.main(["--root", str(root), "--final", "--config", str(bad), "--report-dir", rep]) == 2
    cfg_text = cfg.read_text(encoding="utf-8")
    broken = tmp_path / "broken_registry.toml"                       # registry named but unreadable
    broken.write_text(cfg_text.replace("r.json", "absent.json"), encoding="utf-8")
    assert cp.main(["--root", str(root), "--final", "--config", str(broken), "--report-dir", rep]) == 2
    assert not (tmp_path / "rep").exists()                           # nothing scanned or saved
    monkeypatch.setenv("GRC_CONFIG", str(cfg))                       # valid config through the environment
    assert cp.main(["--root", str(root), "--final", "--report-dir", rep]) == 0
    monkeypatch.delenv("GRC_CONFIG")
    assert cp.main(["--root", str(root), "--report-dir", rep]) == 2  # non-final, no config: INCOMPLETE


def test_sn4_hash_policy(tmp_path):
    root = _tree(tmp_path)
    restricted = b"restricted asset bytes\n"
    rh = hashlib.sha256(restricted).hexdigest()
    approved = hashlib.sha256(b"weights").hexdigest()
    (root / "manifests").mkdir()
    (root / "manifests" / "model_runtime_manifest.json").write_text(json.dumps({"h": approved}), encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "REPRODUCIBILITY.md").write_text(f"hash {approved}\n", encoding="utf-8")
    local, _ = _local(tmp_path, blocked=[rh, approved], approved=[approved])
    assert cp.scan(root, local)["overall"] == "PASS"
    (root / "docs" / "OTHER.md").write_text(f"x {approved}\n", encoding="utf-8")
    rep = cp.scan(root, local)
    assert rep["checks"]["S6"]["status"] == "FAIL" and rep["checks"]["S6"]["findings"][0]["path"] == "docs/OTHER.md"
    (root / "docs" / "OTHER.md").write_text(f"x {rh}\n", encoding="utf-8")
    assert cp.scan(root, local)["checks"]["S6"]["findings"][0]["detail"] == "restricted hash"
    (root / "docs" / "OTHER.md").unlink()
    (root / "docs" / "COPY.md").write_bytes(restricted)
    assert cp.scan(root, local)["checks"]["S6"]["findings"][0]["detail"].startswith("file is byte-identical")


def test_sn5_identity_paths_shingles_claims(tmp_path):
    root = _tree(tmp_path)
    local, _ = _local(tmp_path)
    (root / "docs").mkdir()
    (root / "docs" / "A.md").write_text(f"ok\nby {TOKEN.lower()}\n", encoding="utf-8")
    (root / "docs" / "B.md").write_text(f"path {WIN_USER}\n", encoding="utf-8")
    (root / "docs" / "C.md").write_text("intro\nQuick brown fox, jumps over the lazy dog near the river\n", encoding="utf-8")
    (root / "examples" / "synthetic").mkdir(parents=True)
    (root / "examples" / "synthetic" / "README.md").write_text("Accuracy of this demo\n", encoding="utf-8")
    rep = cp.scan(root, local)
    c = rep["checks"]
    assert c["S4"]["findings"] == [{"path": "docs/A.md", "line": 2}]
    assert c["S3"]["findings"] == [{"path": "docs/B.md", "line": 1}]
    assert c["S5"]["findings"] == [{"path": "docs/C.md", "line": 2}]
    assert c["S7"]["findings"] == [{"path": "examples/synthetic/README.md", "line": 1}]
    assert rep["overall"] == "FAIL"


def test_sn6_report_never_contains_matched_text(tmp_path, capsys):
    root = _tree(tmp_path)
    _, cfg = _local(tmp_path)
    (root / "docs").mkdir()
    (root / "docs" / "A.md").write_text(f"{TOKEN} {WIN_USER} {CONTROL_TEXT}\n", encoding="utf-8")
    code = cp.main(["--root", str(root), "--config", str(cfg), "--report-dir", str(tmp_path / "rep")])
    printed = capsys.readouterr().out
    saved = next((tmp_path / "rep").rglob("scan_report.json")).read_text()
    assert code == 1
    for blob in (printed, saved):
        assert TOKEN not in blob and "someone" not in blob and "riverbank" not in blob


def test_sn7_hidden_ignored_untracked_files_found_without_git(tmp_path):
    root = _tree(tmp_path)
    local, _ = _local(tmp_path)
    (root / ".gitignore").write_text("__pycache__/\n*.toml\n", encoding="utf-8")
    (root / ".secret.csv").write_text("a,b\n", encoding="utf-8")
    (root / ".cache").mkdir()
    (root / ".cache" / "x.txt").write_text("x\n", encoding="utf-8")
    (root / "src" / "grc_mapper" / "__pycache__").mkdir()
    (root / "src" / "grc_mapper" / "__pycache__" / "m.pyc").write_bytes(b"\x00\x01")
    (root / "local.toml").write_text("[x]\n", encoding="utf-8")
    env = child_env(PATH=str(tmp_path / "no_tools_here"))
    code = ("import importlib.util, json, sys; s = importlib.util.spec_from_file_location('c', sys.argv[1]); "
            "m = importlib.util.module_from_spec(s); sys.modules['c'] = m; s.loader.exec_module(m); "
            "print(json.dumps(m.scan(__import__('pathlib').Path(sys.argv[2]), m.load_local_inputs(None))))")
    r = subprocess.run([sys.executable, "-c", code, str(REPO_ROOT / "tools" / "check_publishable.py"), str(root)],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    s1 = {f["path"] for f in json.loads(r.stdout)["checks"]["S1"]["findings"]}
    assert {".secret.csv", ".cache/x.txt", "src/grc_mapper/__pycache__/m.pyc", "local.toml"} <= s1
    assert not (root / ".git").exists()
    assert local  # local inputs not needed for S1


def test_sn8_scanner_never_calls_git():
    tree = ast.parse((REPO_ROOT / "tools" / "check_publishable.py").read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names} | {
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "subprocess" not in imported and not {"Popen", "system", "popen", "run"} & names
    consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert not any(c == "git" or c.startswith("git ") or c in ("ls-files", "--cached") for c in consts)


def test_sn9_links_and_bad_encoding(tmp_path):
    root = _tree(tmp_path)
    local, _ = _local(tmp_path)
    (root / "docs").mkdir()
    (root / "docs" / "latin1.md").write_bytes(b"caf\xe9\n")
    rep = cp.scan(root, local)
    assert {"path": "docs/latin1.md", "line": None, "detail": "not valid UTF-8"} in rep["checks"]["S2"]["findings"]
    target = tmp_path / "outside.md"
    target.write_text("x\n")
    try:
        os.symlink(target, root / "docs" / "link.md")
    except OSError:
        pytest.skip("symbolic-link creation is unavailable on this machine")
    rep = cp.scan(root, local)
    assert any(f["path"] == "docs/link.md" for f in rep["checks"]["S1"]["findings"])


@pytest.mark.skipif(os.name == "nt", reason="chmod-based unreadable files are POSIX-only")
def test_sn9b_unreadable_file_is_incomplete(tmp_path):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("running as root: file permissions are not enforced")
    root = _tree(tmp_path)
    local, _ = _local(tmp_path)
    p = root / "README.md"
    p.chmod(0)
    try:
        assert cp.scan(root, local)["overall"] in ("INCOMPLETE", "FAIL")
    finally:
        p.chmod(0o644)


def test_sn10_git_dir_counted_not_scanned(tmp_path):
    root = _tree(tmp_path)
    local, _ = _local(tmp_path)
    (root / ".git" / "objects").mkdir(parents=True)
    (root / ".git" / "objects" / "blob").write_bytes(b"\x00" * 10 + WIN_USER.encode())
    (root / ".gitmodules").write_text("[x]\n", encoding="utf-8")
    rep = cp.scan(root, local)
    assert rep["git_metadata"] == {"present": True, "files": 1, "bytes": 10 + len(WIN_USER), "content_scanned": False}
    assert [f["path"] for f in rep["checks"]["S1"]["findings"]] == [".gitmodules"]
    assert rep["checks"]["S3"]["status"] == "PASS"


def test_lock_mismatch_and_missing(tmp_path):
    root = _tree(tmp_path)
    local, _ = _local(tmp_path)
    (root / "requirements" / "grc-mvp.lock.txt").write_bytes(LOCK + b"# x\n")
    assert cp.scan(root, local)["checks"]["S8"]["status"] == "FAIL"
    shutil.rmtree(root / "requirements")
    assert cp.scan(root, local)["checks"]["S8"]["findings"][0]["detail"] == "lock copy missing"


def test_blocked_extensions_and_size(tmp_path):
    root = _tree(tmp_path)
    local, _ = _local(tmp_path)
    (root / "m.safetensors").write_bytes(b"x")
    (root / "docs").mkdir()
    (root / "docs" / "BIG.md").write_bytes(b"a" * (cp.MAX_BYTES + 1))
    rep = cp.scan(root, local)
    details = {(f["path"], f.get("detail")) for f in rep["checks"]["S2"]["findings"]}
    assert ("m.safetensors", "blocked extension") in details and ("docs/BIG.md", "larger than 1 MB") in details
