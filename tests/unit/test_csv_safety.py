"""Unit tests: CSV formula-injection guard and CSV serialisation of awkward identifiers and text.

Covers export (results and queue CSV, include_text), history (snapshots, comparison CSV) and the
map-multi side files (statements.csv, excluded.csv, rejected.csv). JSON keeps original values."""

import csv
import io
import json

import pytest

from _support import TestEncoder
from grc_mapper import cli
from grc_mapper.config import load_settings
from grc_mapper.csvcell import neutralise, restore
from grc_mapper.data import ControlCatalog, Statement
from grc_mapper.export import result_json, write_queue_csv, write_results_csv
from grc_mapper.history import Snapshot, SnapshotRow, compare_snapshots, read_snapshot, write_comparison_csv, write_snapshot
from grc_mapper.pipeline import build_control_index, map_statements

FORMULAS = ["=1+1", "+SUM(A1:A2)", "-2+3+cmd|' /C calc'!A0", "@SUM(A1)", "=HYPERLINK(\"http://x\",\"y\")",
            "\t=1", "\r=1", "'already-quoted"]
SAFE = ["R00001", "A.5.1", "AC-2(1)", "0.12345678", "-0.12345678", "+1", "1e-05", "-1.5E+3", ".5", "", "x=1"]
AWKWARD = ["a,b", 'say "hi"', "line1\nline2", "semi;colon", 'mix, "q"\r\nend']


def _is_number(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def _parse(data: bytes):
    return list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))


@pytest.mark.parametrize("v", FORMULAS)
def test_formula_cells_neutralised_and_reversible(v):
    out = neutralise(v)
    assert out == "'" + v and restore(out) == v


@pytest.mark.parametrize("v", SAFE + AWKWARD)
def test_numbers_and_plain_values_unchanged(v):
    assert neutralise(v) == v and restore(neutralise(v)) == v


def _scored(ids, texts=None):
    cat = ControlCatalog(control_ids=("C-1", "C-2", "C-3"), texts=("alpha control", "beta control", "gamma control"),
                         source_sha256="0" * 64, source_kind="research")
    enc = TestEncoder()
    s = load_settings({"batch_size": 4}, env={})
    idx = build_control_index(cat, enc, s)
    stmts = [Statement(i, (texts or {}).get(i, f"policy statement about access {n}")) for n, i in enumerate(ids)]
    return map_statements(stmts, idx, enc, s), {st.statement_id: st.text for st in stmts}


IDS = ["=1+1", "+x", "-x", "@x", "a,b", 'q"uote', "nl\nid", "R00001"]


def test_results_and_queue_csv_guarded_and_parseable(tmp_path):
    text = {"=1+1": "=cmd|' /C calc'!A0 policy text", "a,b": "text, with comma and \"quotes\"\nand a newline"}
    rows, texts = _scored(IDS, text)
    write_results_csv(rows, tmp_path / "r.csv", include_text=True, texts=texts, extra={"framework_id": "-FW"})
    write_queue_csv(rows, tmp_path / "q.csv")
    for name, n_extra in (("r.csv", 0), ("q.csv", 0)):
        recs = list(csv.DictReader(io.StringIO((tmp_path / name).read_text(encoding="utf-8"), newline="")))
        assert len(recs) == len(IDS) + n_extra
        got = {restore(r["statement_id"]) for r in recs}
        assert got == set(IDS)
        for r in recs:
            for k, v in r.items():
                assert not v or v[0] not in "=+-@\t\r\n" or _is_number(v), (name, k, v)
            float(r["top1_similarity"])
    recs = {restore(r["statement_id"]): r for r in csv.DictReader(io.StringIO((tmp_path / "r.csv").read_text(encoding="utf-8"), newline=""))}
    assert recs["=1+1"]["statement_id"] == "'=1+1" and recs["=1+1"]["statement_text"].startswith("'=cmd")
    assert recs["a,b"]["statement_text"] == text["a,b"] and recs["nl\nid"]["statement_id"] == "nl\nid"
    assert recs["R00001"]["statement_id"] == "R00001" and recs["R00001"]["framework_id"] == "'-FW"


def test_negative_similarity_stays_numeric(tmp_path):
    rows, _ = _scored(["N1"])
    r = rows[0]
    import dataclasses
    neg = dataclasses.replace(r, top1_similarity=-0.25, drift_index=1.25, aps=1.875)
    write_results_csv([neg], tmp_path / "n.csv")
    rec = next(csv.DictReader(io.StringIO((tmp_path / "n.csv").read_text(encoding="utf-8"), newline="")))
    assert rec["top1_similarity"] == "-0.25000000"


def test_json_keeps_original_values():
    rows, _ = _scored(["=1+1", "@x"])
    d = [result_json(r) for r in rows]
    assert [x["statement_id"] for x in d] == ["=1+1", "@x"]
    assert json.loads(json.dumps(d))[0]["statement_id"] == "=1+1"


def test_history_snapshot_roundtrip_and_comparison_guarded(tmp_path):
    meta = {"framework_id": "FW", "framework_version": "1", "catalogue_sha256": "c" * 64,
            "model_weights_sha256": "m" * 64, "preprocessing": "p", "bands": {"high_min": 0.59, "review_min": 0.49},
            "aps_weights": {"HIGH": 1.0}}
    ids = IDS + ["'lead", "-0.5"]
    rows = tuple(SnapshotRow(i, "t" * 64, "=C-1", 0.4, "ESCALATE", 0.6, 0.9, ("=C-1", "C,2")) for i in ids)
    s = Snapshot(meta, rows)
    write_snapshot(s, tmp_path / "s.csv", tmp_path / "s.json")
    raw = _parse((tmp_path / "s.csv").read_bytes())
    assert all(not c or c[0] not in "=+@" for row in raw[1:] for c in row)
    assert read_snapshot(tmp_path / "s.csv", tmp_path / "s.json") == s
    cur = Snapshot(meta, tuple(SnapshotRow(r.statement_id, "u" * 64, r.top1_control_id, 0.41, r.tier_code, 0.59,
                                           0.885, r.top_controls) for r in rows))
    cmp_rows, summary = compare_snapshots(s, cur)
    assert summary["n_matched"] == len(ids)
    write_comparison_csv(cmp_rows, tmp_path / "c.csv")
    recs = list(csv.DictReader(io.StringIO((tmp_path / "c.csv").read_text(encoding="utf-8"), newline="")))
    assert {restore(r["statement_id"]) for r in recs} == set(ids)
    assert all(r["previous_top1"] == "'=C-1" for r in recs)
    assert float(recs[0]["delta_di"]) == pytest.approx(-0.01)


def test_map_multi_side_files_safe(tmp_path, monkeypatch, capsysbinary):
    cat = tmp_path / "cat.csv"
    cat.write_text("Control_ID,Control_Text\nC-1,alpha control text\nC-2,beta control text\nC-3,gamma control text\n",
                   encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema_version": 1, "frameworks": [{
        "framework_id": "FW", "framework_name": "FW", "framework_version": "1", "catalogue_csv": str(cat),
        "id_column": "Control_ID", "text_column": "Control_Text", "source": "test", "licence": "test",
        "bands": {"high_min": 0.59, "review_min": 0.49, "basis": "test"},
        "aps_weights": {"HIGH": 1.0, "REVIEW": 1.25, "ESCALATE": 1.5}}]}), encoding="utf-8")
    ids = ["=1+1", "a,b", 'q"uote', "nl\nid", "@x", "R1"]
    roles = {"=1+1": "keep", "a,b": "drop, \"x\"", 'q"uote': "keep", "nl\nid": "=drop\nnow", "@x": "keep", "R1": "-1+2"}
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Statement_ID", "Statement_Text", "role"])
    for n, i in enumerate(ids):
        w.writerow([i, f"policy statement about access number {n}", roles[i]])
    stm = tmp_path / "s.csv"
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
    out = tmp_path / "runs"
    code = cli.main(["map-multi", "--config", str(cfg), "--statements", str(stm), "--frameworks", str(spec),
                     "--out", str(out), "--select-column", "role", "--select-value", "keep"])
    o = capsysbinary.readouterr()
    assert code == 0, o.err
    run = out / json.loads(o.out)["run_id"]
    st = list(csv.DictReader(io.StringIO((run / "statements.csv").read_text(encoding="utf-8"), newline="")))
    assert [restore(r["statement_id"]) for r in st] == ids and all(len(r["text_sha256"]) == 64 for r in st)
    assert {restore(r["statement_id"]): r["selected"] for r in st} == {i: str(roles[i] == "keep").lower() for i in ids}
    exc = list(csv.DictReader(io.StringIO((run / "excluded.csv").read_text(encoding="utf-8"), newline="")))
    assert {restore(e["statement_id"]): restore(e["value"]) for e in exc} == {i: roles[i] for i in ids if roles[i] != "keep"}
    assert {e["select_column"] for e in exc} == {"role"}
    assert [e["value"] for e in exc if restore(e["statement_id"]) == "nl\nid"] == ["'=drop\nnow"]
    assert (run / "rejected.csv").read_bytes() == b"statement_id,reason\n"
    res = list(csv.DictReader(io.StringIO((run / "results_FW.csv").read_text(encoding="utf-8"), newline="")))
    assert [restore(r["statement_id"]) for r in res] == [i for i in ids if roles[i] == "keep"]
    assert res[0]["statement_id"] == "'=1+1"
