"""Unit tests: framework specs, multi-framework mapping, run snapshots and comparison, map-multi CLI,
U-IN-5 CLI coverage and the validate_env offline-ordering fix. Isolated test encoder; no real model."""

import csv
import importlib.abc
import io
import json
import subprocess
import sys

import pytest

from _support import REPO_ROOT, TestEncoder, child_env
from grc_mapper import cli
from grc_mapper.config import load_settings
from grc_mapper.data import ControlCatalog, Statement
from grc_mapper.errors import ConfigError, InputValidationError
from grc_mapper.frameworks import load_framework_catalogue, load_framework_spec, parse_framework_spec
from grc_mapper.history import (
    IncompatibleSnapshots, Snapshot, SnapshotRow, compare_snapshots, read_snapshot, review_queue, write_snapshot,
)
from grc_mapper.multiframework import FrameworkIndex, encode_statements, map_framework, map_multi
from grc_mapper.pipeline import build_control_index, map_statements

SECRET = "ZQX-SENTINEL wording that must never leak"
STMTS = [Statement(f"S{i:03d}", f"statement {i} about topic {i % 4} and words " + "w " * (i % 9)) for i in range(25)]


def _catalog_csv(tmp_path, name, rows, id_col="Control_ID", text_col="Control_Text"):
    p = tmp_path / name
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([id_col, text_col])
    w.writerows(rows)
    p.write_text(buf.getvalue(), encoding="utf-8")
    return p


def _spec(tmp_path, entries):
    p = tmp_path / "frameworks.json"
    p.write_text(json.dumps({"schema_version": 1, "frameworks": entries}), encoding="utf-8")
    return p


def _entry(fid, path, *, version="1", bands=(0.59, 0.49), weights=None, expected=None, id_col="Control_ID",
           text_col="Control_Text"):
    e = {"framework_id": fid, "framework_name": f"name {fid}", "framework_version": version,
         "catalogue_csv": str(path), "id_column": id_col, "text_column": text_col, "source": "unit test",
         "licence": "synthetic", "bands": {"high_min": bands[0], "review_min": bands[1], "basis": "test"}}
    if weights:
        e["aps_weights"] = weights
    if expected:
        e["expected_controls"] = expected
    return e


@pytest.fixture
def two_frameworks(tmp_path):
    a = _catalog_csv(tmp_path, "a.csv", [(f"A.{i}", f"alpha control {i} topic {i % 4}") for i in range(1, 8)])
    b = _catalog_csv(tmp_path, "b.csv", [(f"AC-{i}", f"beta requirement {i} area {i % 3}") for i in range(1, 12)]
                     + [("A.1", "beta item sharing an identifier string with framework A")])
    spec = _spec(tmp_path, [_entry("FW-A", a, version="2022", expected=7),
                            _entry("FW-B", b, version="5.2.0", bands=(0.3, 0.1), weights={"HIGH": 1, "REVIEW": 2, "ESCALATE": 3})])
    return spec


# ------------------------------------------------------------------ 1. catalogue loading and unique IDs

def test_spec_and_catalogue_loading(two_frameworks):
    specs = load_framework_spec(two_frameworks)
    assert [s.framework_id for s in specs] == ["FW-A", "FW-B"]
    cats = {s.framework_id: load_framework_catalogue(s) for s in specs}
    assert len(cats["FW-A"]) == 7 and len(cats["FW-B"]) == 12
    assert len(cats["FW-B"].control_ids) == len(set(cats["FW-B"].control_ids))


def test_duplicate_control_ids_rejected(tmp_path):
    p = _catalog_csv(tmp_path, "d.csv", [("X-1", "one text"), ("X-1", "two text")])
    spec = parse_framework_spec({"schema_version": 1, "frameworks": [_entry("D", p)]})[0]
    with pytest.raises(InputValidationError, match="duplicate control IDs"):
        load_framework_catalogue(spec)


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d["frameworks"].append(dict(d["frameworks"][0])), "duplicate framework_id"),
    (lambda d: d["frameworks"][0].update(framework_id="bad id"), "framework_id must match"),
    (lambda d: d["frameworks"][0].update(extra=1), "unknown keys"),
    (lambda d: d["frameworks"][0].pop("licence"), "missing keys"),
    (lambda d: d["frameworks"][0]["bands"].update(high_min=0.4), "review_min < high_min"),
    (lambda d: d["frameworks"][0]["bands"].pop("basis"), "bands must have exactly"),
    (lambda d: d.update(schema_version=2), "schema_version"),
    (lambda d: d["frameworks"][0].update(aps_weights={"HIGH": 1}), "exactly the keys"),
])
def test_spec_rejections(tmp_path, mutate, msg):
    p = _catalog_csv(tmp_path, "c.csv", [("X-1", "one text")])
    doc = {"schema_version": 1, "frameworks": [_entry("OK", p)]}
    mutate(doc)
    with pytest.raises(ConfigError, match=msg):
        parse_framework_spec(doc)


def test_expected_count_enforced(tmp_path):
    p = _catalog_csv(tmp_path, "c.csv", [("X-1", "one text"), ("X-2", "two text")])
    spec = parse_framework_spec({"schema_version": 1, "frameworks": [_entry("C", p, expected=93)]})[0]
    with pytest.raises(InputValidationError, match="expected 93 controls"):
        load_framework_catalogue(spec)


# ------------------------------------------------------------------ 3-6. independent indices, mapping, ranking, arithmetic

def _indices(spec_path, enc, settings):
    specs = load_framework_spec(spec_path)
    return [FrameworkIndex(s, build_control_index(load_framework_catalogue(s), enc, s.scoring_settings(settings)))
            for s in specs]


def test_independent_indices_and_namespaces(two_frameworks, enc, settings):
    fws = _indices(two_frameworks, enc, settings)
    a, b = fws
    assert a.index.embeddings.shape[0] == 7 and b.index.embeddings.shape[0] == 12
    assert "A.1" in a.index.catalog.control_ids and "A.1" in b.index.catalog.control_ids  # same string, separate namespaces
    res = map_multi(STMTS, fws, enc, settings)
    assert set(res) == {"FW-A", "FW-B"}
    assert all(c.control_id in set(a.index.catalog.control_ids) for r in res["FW-A"] for c in r.candidates)
    assert all(c.control_id in set(b.index.catalog.control_ids) for r in res["FW-B"] for c in r.candidates)


def test_multi_equals_single_framework_pipeline(two_frameworks, enc, settings):
    fws = _indices(two_frameworks, enc, settings)
    res = map_multi(STMTS, fws, enc, settings)
    for fw in fws:
        ref = map_statements(STMTS, fw.index, enc, fw.spec.scoring_settings(settings))
        assert res[fw.spec.framework_id] == ref


def test_embeddings_computed_once(two_frameworks, settings):
    e = TestEncoder()
    fws = _indices(two_frameworks, e, settings)
    before = len(e.calls)
    map_multi(STMTS, fws, e, settings)
    assert len(e.calls) - before == 1 and e.calls[-1][0] == len(STMTS)


def test_framework_specific_bands_weights_and_arithmetic(two_frameworks, enc, settings):
    fws = _indices(two_frameworks, enc, settings)
    res = map_multi(STMTS, fws, enc, settings)
    for r in res["FW-B"]:
        s = r.top1_similarity
        want = "HIGH" if s >= 0.3 else ("REVIEW" if s >= 0.1 else "ESCALATE")
        assert r.tier_code == want and r.drift_index == 1.0 - s
        assert r.aps == (1.0 - s) * {"HIGH": 1.0, "REVIEW": 2.0, "ESCALATE": 3.0}[want]
    for r in res["FW-A"]:
        s = r.top1_similarity
        assert r.tier_code == ("HIGH" if s >= 0.59 else ("REVIEW" if s >= 0.49 else "ESCALATE"))
        assert r.aps == (1.0 - s) * {"HIGH": 1.0, "REVIEW": 1.25, "ESCALATE": 1.5}[r.tier_code]


def test_framework_specific_top_k(two_frameworks, enc):
    s8 = load_settings({"top_k": 8}, env={})
    fws = _indices(two_frameworks, enc, s8)
    with pytest.raises(ConfigError, match="top_k must be between 1 and 7 for FW-A"):
        map_multi(STMTS[:2], fws, enc, s8)
    enc_once = encode_statements(STMTS[:3], enc, s8)
    rows = map_framework(enc_once, fws[1], s8)
    assert all(len(r.candidates) == 8 for r in rows)
    assert all([c.similarity for c in r.candidates] == sorted([c.similarity for c in r.candidates], reverse=True)
               for r in rows)


def test_mismatched_encoder_rejected(two_frameworks, enc, settings):
    fws = _indices(two_frameworks, enc, settings)
    other = TestEncoder(dim=8)
    with pytest.raises(ConfigError):
        map_framework(encode_statements(STMTS[:2], other, settings), fws[0], settings)


# ------------------------------------------------------------------ 9. duplicate / missing IDs

def test_duplicate_statement_ids_rejected(two_frameworks, enc, settings):
    fws = _indices(two_frameworks, enc, settings)
    with pytest.raises(InputValidationError, match="duplicate"):
        map_multi([STMTS[0], STMTS[0]], fws, enc, settings)
    with pytest.raises(InputValidationError):
        map_multi([], fws, enc, settings)
    with pytest.raises(ConfigError):
        map_multi(STMTS[:1], [fws[0], fws[0]], enc, settings)


# ------------------------------------------------------------------ 7-8. snapshots and comparison

META = {"framework_id": "FW", "framework_version": "1", "catalogue_sha256": "c" * 64, "model_weights_sha256": "m" * 64,
        "preprocessing": "p", "bands": {"high_min": 0.59, "review_min": 0.49}, "aps_weights": {"HIGH": 1.0}}


def _row(sid, sim, top1="X-1", text="t" * 64, tops=()):
    tier = "HIGH" if sim >= 0.59 else ("REVIEW" if sim >= 0.49 else "ESCALATE")
    w = {"HIGH": 1.0, "REVIEW": 1.25, "ESCALATE": 1.5}[tier]
    return SnapshotRow(sid, text, top1, sim, tier, 1 - sim, (1 - sim) * w, tuple(tops))


def test_compare_by_stable_id(tmp_path):
    prev = Snapshot(dict(META), (_row("B", 0.50), _row("A", 0.70, tops=("X-1", "X-2", "X-3")), _row("C", 0.40)))
    cur = Snapshot(dict(META), (_row("A", 0.6999, tops=("X-1", "X-4", "X-3")), _row("B", 0.60, top1="X-9"),
                                _row("D", 0.3), _row("C", 0.40, text="u" * 64)))
    rows, summary = compare_snapshots(prev, cur, materiality=1e-3)
    by = {r["statement_id"]: r for r in rows}
    assert by["A"]["status"] == "matched" and abs(by["A"]["delta_di"] - (0.3001 - 0.3)) < 1e-12
    assert by["A"]["top3_overlap"] == 2 and not by["A"]["material_change"]
    assert by["B"]["top1_changed"] and by["B"]["tier_transition"] == "REVIEW->HIGH" and not by["B"]["tier_worsened"]
    assert by["C"]["text_changed"] and by["C"]["material_change"] and by["C"]["top3_overlap"] is None
    assert by["D"]["status"] == "only_current"
    assert summary["n_matched"] == 3 and summary["only_current"] == 1 and summary["top1_changed"] == 1
    q = review_queue(rows)
    assert [r["statement_id"] for r in q] == ["C", "B"] and q[0]["review_status"] == "CHANGE_PENDING_HUMAN_REVIEW"


@pytest.mark.parametrize("key,value", [
    ("model_weights_sha256", "z" * 64), ("catalogue_sha256", "d" * 64), ("framework_version", "2"),
    ("framework_id", "OTHER"), ("preprocessing", "q"), ("bands", {"high_min": 0.6, "review_min": 0.5}),
    ("aps_weights", {"HIGH": 2.0}),
])
def test_incompatible_snapshots_rejected(key, value):
    prev = Snapshot(dict(META), (_row("A", 0.7),))
    cur = Snapshot({**META, key: value}, (_row("A", 0.7),))
    with pytest.raises(IncompatibleSnapshots, match=key):
        compare_snapshots(prev, cur)


def test_snapshot_validation_and_roundtrip(tmp_path):
    with pytest.raises(InputValidationError, match="duplicate"):
        compare_snapshots(Snapshot(dict(META), (_row("A", 0.7), _row("A", 0.6))), Snapshot(dict(META), ()))
    with pytest.raises(InputValidationError, match="lacks"):
        compare_snapshots(Snapshot({"framework_id": "FW"}, ()), Snapshot(dict(META), ()))
    s = Snapshot(dict(META), (_row("A", 0.7, tops=("X-1", "X-2")), _row("B", 0.3)))
    write_snapshot(s, tmp_path / "s.csv", tmp_path / "s.json")
    assert read_snapshot(tmp_path / "s.csv", tmp_path / "s.json") == s
    with pytest.raises(FileExistsError):
        write_snapshot(s, tmp_path / "s.csv", tmp_path / "s2.json")


# ------------------------------------------------------------------ map-multi CLI (2, 9, 10)

@pytest.fixture
def multi_env(tmp_path, monkeypatch, two_frameworks):
    stm = tmp_path / "statements.csv"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Statement_ID", "Statement_Text", "role"])
    for i, st in enumerate(STMTS[:10]):
        w.writerow([st.statement_id, (SECRET + " " if i == 0 else "") + st.text, "keep" if i < 8 else "drop"])
    stm.write_text(buf.getvalue(), encoding="utf-8")
    cfg = tmp_path / "local.toml"
    cfg.write_text("[runtime]\nbatch_size = 4\n", encoding="utf-8")

    class Report:
        status = "OK_FULL"
        runtime_files_verified = 8
        runtime_manifest_sha256 = "r" * 64
        full_manifest_sha256 = "f" * 64

    def fake_load(settings):
        e = TestEncoder()
        e.report = Report()
        return e

    monkeypatch.setattr(cli, "load_encoder", fake_load)
    monkeypatch.delenv("GRC_CONFIG", raising=False)
    return {"cfg": str(cfg), "stm": str(stm), "spec": str(two_frameworks), "tmp": tmp_path}


def _run(argv, capsysbinary):
    code = cli.main(argv)
    out = capsysbinary.readouterr()
    return code, out.out, out.err.decode("utf-8", "replace")


def test_map_multi_outputs_provenance_and_privacy(multi_env, capsysbinary):
    out_dir = multi_env["tmp"] / "runs"
    code, out, err = _run(["map-multi", "--config", multi_env["cfg"], "--statements", multi_env["stm"],
                           "--frameworks", multi_env["spec"], "--out", str(out_dir), "--select-column", "role",
                           "--select-value", "keep", "--top-k", "3"], capsysbinary)
    assert code == 0, err
    summary = json.loads(out)
    run = out_dir / summary["run_id"]
    assert sorted(p.name for p in run.iterdir()) == sorted([
        "results_FW-A.csv", "queue_FW-A.csv", "results_FW-B.csv", "queue_FW-B.csv", "statements.csv",
        "excluded.csv", "rejected.csv", "provenance.json"])
    prov = json.loads((run / "provenance.json").read_text())
    assert [f["framework_id"] for f in prov["frameworks"]] == ["FW-A", "FW-B"]
    assert [f["framework_version"] for f in prov["frameworks"]] == ["2022", "5.2.0"]
    assert all(len(f["catalogue_sha256"]) == 64 for f in prov["frameworks"])
    assert prov["inputs"]["rows_total"] == 10 and prov["inputs"]["rows_selected"] == 8 and prov["inputs"]["rows_excluded"] == 2
    assert prov["model"]["full_manifest_sha256"] == "f" * 64 and prov["model"]["model_weights_sha256"]
    for fid, ver in (("FW-A", "2022"), ("FW-B", "5.2.0")):
        rows = list(csv.DictReader(io.StringIO((run / f"results_{fid}.csv").read_text(encoding="utf-8"))))
        assert len(rows) == 8 and {r["framework_id"] for r in rows} == {fid} and {r["framework_version"] for r in rows} == {ver}
    exc = list(csv.DictReader(io.StringIO((run / "excluded.csv").read_text(encoding="utf-8"))))
    assert [e["statement_id"] for e in exc] == ["S008", "S009"] and {e["value"] for e in exc} == {"drop"}
    st = list(csv.DictReader(io.StringIO((run / "statements.csv").read_text(encoding="utf-8"))))
    assert len(st) == 10 and all(len(r["text_sha256"]) == 64 for r in st)
    for p in run.iterdir():
        assert SECRET not in p.read_text(encoding="utf-8")
    assert SECRET not in err


def test_map_multi_selection_errors(multi_env, capsysbinary):
    base = ["map-multi", "--config", multi_env["cfg"], "--statements", multi_env["stm"], "--frameworks",
            multi_env["spec"], "--out", str(multi_env["tmp"] / "o")]
    code, _, err = _run(base + ["--select-column", "role"], capsysbinary)
    assert code == 2 and "together" in err
    code, _, err = _run(base + ["--select-column", "role", "--select-value", "none"], capsysbinary)
    assert code == 4
    code, _, err = _run(base + ["--select-column", "nope", "--select-value", "x"], capsysbinary)
    assert code == 4 and "missing required columns" in err
    code, _, err = _run(base + ["--only", "FW-Z"], capsysbinary)
    assert code == 2 and "unknown framework IDs" in err
    code, _, err = _run(base[:-1] + [str(REPO_ROOT / "x")], capsysbinary)
    assert code == 2


# ------------------------------------------------------------------ U-IN-5: map-one / map-batch without OK_FULL

class _ImportSpy(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.seen = []

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("sentence_transformers", "transformers", "torch"):
            self.seen.append(name)
        return None


@pytest.fixture
def model_cfg(fake_model, tmp_path, monkeypatch):
    model, rt, full = fake_model
    ctrl = _catalog_csv(tmp_path, "ctrl.csv", [("A.1", "access control text"), ("A.2", "backup text"), ("A.3", "other")])
    stm = tmp_path / "s.csv"
    stm.write_text("Statement_ID,Statement_Text\nS1,users review access\n", encoding="utf-8")
    no_full = tmp_path / "nofull.toml"
    no_full.write_text(f"[paths]\nmodel_dir = '{model}'\nmodel_manifest = '{rt}'\ncontrols_csv = '{ctrl}'\n")
    bad_full = tmp_path / "badfull.toml"
    bad_full.write_text(no_full.read_text() + f"model_full_manifest = '{tmp_path / 'absent.json'}'\n")
    monkeypatch.delenv("GRC_CONFIG", raising=False)
    spy = _ImportSpy()
    monkeypatch.setattr(sys, "meta_path", [spy] + sys.meta_path)
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    return {"no_full": str(no_full), "bad_full": str(bad_full), "stm": str(stm), "tmp": tmp_path, "spy": spy}


@pytest.mark.parametrize("cmd", ["map-one", "map-batch"])
def test_u_in_5_production_commands_fail_closed_without_full_manifest(model_cfg, capsysbinary, cmd):
    extra = (["--text", "users review access rights"] if cmd == "map-one"
             else ["--statements", model_cfg["stm"], "--out", str(model_cfg["tmp"] / "o")])
    code, _, err = _run([cmd, "--config", model_cfg["no_full"], *extra], capsysbinary)
    assert code == 2 and "model_full_manifest" in err          # not configured: configuration error
    code, _, err = _run([cmd, "--config", model_cfg["bad_full"], *extra], capsysbinary)
    assert code == 3 and "full manifest" in err                # configured but unreadable: integrity error
    assert model_cfg["spy"].seen == []                         # model library never imported
    assert not (model_cfg["tmp"] / "o").exists()


# ------------------------------------------------------------------ F-01: validate_env sets offline variables first

def test_validate_env_sets_offline_env_before_ml_imports(tmp_path):
    code = (
        "import importlib.abc, json, os, sys\n"
        "seen = {}\n"
        "class Spy(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name in ('torch', 'sentence_transformers') and name not in seen:\n"
        "            seen[name] = os.environ.get('HF_HUB_OFFLINE'), os.environ.get('TRANSFORMERS_OFFLINE')\n"
        "            raise ImportError('blocked by test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Spy())\n"
        "for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'): os.environ.pop(k, None)\n"
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('ve', sys.argv[1]); m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['ve'] = m; spec.loader.exec_module(m)\n"
        "m.main(['--lock', sys.argv[2], '--config', sys.argv[3]])\n"
        "print(json.dumps(seen))\n"
    )
    cfg = tmp_path / "empty.toml"
    cfg.write_text("[runtime]\ndevice = 'cpu'\n", encoding="utf-8")
    env = child_env()
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    r = subprocess.run([sys.executable, "-c", code, str(REPO_ROOT / "tools" / "validate_env.py"),
                        str(REPO_ROOT / "requirements" / "grc-mvp.lock.txt"), str(cfg)],
                       env=env, capture_output=True, text=True)
    seen = json.loads(r.stdout.strip().splitlines()[-1])
    assert seen.get("torch") == ["1", "1"], r.stdout + r.stderr
