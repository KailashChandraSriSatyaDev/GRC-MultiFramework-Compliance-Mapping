"""End-to-end CLI tests with the REAL production model (authoritative run: grc-mvp).

Synthetic statements only (written to a temporary directory) mapped against the real local control
catalogue. Outputs go to pytest's temporary directory, never into the repository.
"""

import csv
import io
import json
import os
import subprocess
import sys

import pytest

if not os.environ.get("GRC_CONFIG"):
    pytest.skip("GRC_CONFIG not set: integration tests need the local configuration", allow_module_level=True)

from _support import REPO_ROOT, child_env  # noqa: E402

NEW_STATEMENT = "All administrator accounts must use multi-factor authentication and be reviewed every quarter."
BATCH = [
    ("NEW-001", "Backups of the finance database are encrypted and restored as a test each month."),
    ("NEW-002", "Visitors must sign in at reception and be escorted in secure areas."),
    ("NEW-003", "Security incidents are reported to the service desk within one hour."),
    ("NEW-004", "Developers must not use production data in test environments."),
]


def _cli(*args, stdin=None):
    r = subprocess.run([sys.executable, "-m", "grc_mapper", *args], env=child_env(), capture_output=True,
                       input=stdin, cwd=str(REPO_ROOT), timeout=900)
    return r.returncode, r.stdout, r.stderr.decode("utf-8", "replace")


def test_e2e_verify_model_load():
    code, out, err = _cli("verify-model", "--load")
    d = json.loads(out)
    assert code == 0, err
    assert d["status"] == "OK_FULL" and d["load"]["embedding_dim"] == 384 and d["load"]["max_seq_length"] == 256


def test_e2e_verify_inputs_research_profile():
    code, out, err = _cli("verify-inputs", "--profile", "research")
    d = json.loads(out)
    assert code == 0, err
    assert d["controls"]["rows"] == 93 and d["statements"]["rows"] == 206 and d["statements"]["labels_checked"]


def test_e2e_map_one_new_statement_text_file_and_out(tmp_path):
    tf = tmp_path / "new.txt"
    tf.write_text(NEW_STATEMENT, encoding="utf-8")
    out_dir = tmp_path / "runs"
    code, out, err = _cli("map-one", "--text-file", str(tf), "--out", str(out_dir), "--statement-id", "NEW-ONE")
    assert code == 0, err
    d = json.loads(out)
    assert d["encoder_kind"] == "SBERT_PRODUCTION" and d["review_status"] == "CANDIDATE_PENDING_HUMAN_REVIEW"
    assert len(d["candidates"]) == 3 and d["tier_code"] in ("HIGH", "REVIEW", "ESCALATE")
    run = out_dir / d["run_id"]
    assert sorted(p.name for p in run.iterdir()) == ["provenance.json", "result.csv", "result.json"]
    assert (run / "result.json").read_bytes() == out
    for p in run.iterdir():
        assert "multi-factor" not in p.read_text(encoding="utf-8")
    print(f"map-one: top-1 {d['top1_control_id']} sim {d['top1_similarity']:.4f} tier {d['tier_code']}")


def test_e2e_map_one_stdin_equals_text_file(tmp_path):
    tf = tmp_path / "new.txt"
    tf.write_text(NEW_STATEMENT, encoding="utf-8")
    c1, o1, _ = _cli("map-one", "--text-file", str(tf))
    c2, o2, _ = _cli("map-one", "--stdin", stdin=NEW_STATEMENT.encode("utf-8"))
    a, b = json.loads(o1), json.loads(o2)
    assert c1 == c2 == 0
    assert [c["control_id"] for c in a["candidates"]] == [c["control_id"] for c in b["candidates"]]
    assert abs(a["top1_similarity"] - b["top1_similarity"]) <= 1e-6


def test_e2e_map_batch_and_single_consistency(tmp_path):
    import tomllib
    cfg = tomllib.loads(open(os.environ["GRC_CONFIG"], encoding="utf-8").read())
    id_col, text_col = cfg["columns"]["statement_id"], cfg["columns"]["statement_text"]
    csv_path = tmp_path / "batch.csv"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([id_col, text_col])
    w.writerows(BATCH)
    csv_path.write_text(buf.getvalue(), encoding="utf-8")
    code, out, err = _cli("map-batch", "--statements", str(csv_path), "--out", str(tmp_path / "b"))
    assert code == 0, err
    s = json.loads(out)
    run = tmp_path / "b" / s["run_id"]
    assert sorted(p.name for p in run.iterdir()) == ["provenance.json", "queue.csv", "results.csv"]
    rows = {r["statement_id"]: r for r in csv.DictReader(io.StringIO((run / "results.csv").read_text(encoding="utf-8")))}
    prov = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    assert prov["encoder_kind"] == "SBERT_PRODUCTION" and prov["model_files_verified"] == 8 and prov["n_statements"] == 4
    for sid, text in BATCH[:2]:
        c, o, _ = _cli("map-one", "--stdin", "--statement-id", sid, stdin=text.encode("utf-8"))
        d = json.loads(o)
        assert c == 0 and d["top1_control_id"] == rows[sid]["top1_control_id"]
        assert abs(d["top1_similarity"] - float(rows[sid]["top1_similarity"])) <= 1e-5
    assert all(text not in (run / "results.csv").read_text(encoding="utf-8") for _, text in BATCH)
    print("map-batch:", {k: (v["top1_control_id"], v["tier_code"]) for k, v in rows.items()})
