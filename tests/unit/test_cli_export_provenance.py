"""Unit tests: text input, FakeEncoder, safe logging, export, provenance and the CLI (no real model)."""

import ast
import csv
import io
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from _support import REPO_ROOT, TestEncoder, child_env
from grc_mapper import cli
from grc_mapper.config import load_settings
from grc_mapper.data import Statement
from grc_mapper.errors import ConfigError, InputValidationError, OutputWriteError
from grc_mapper.export import (
    RESULT_COLUMNS, check_output_location, create_run_dir, json_bytes, result_json, result_record, write_json,
    write_queue_csv, write_results_csv,
)
from grc_mapper.fake_encoder import FAKE_KIND, FakeEncoder
from grc_mapper.logging_setup import RedactingFilter
from grc_mapper.mapping import Candidate
from grc_mapper.provenance import DEMO_DISCLAIMER, DISCLAIMER, build_run_record, new_run_id
from grc_mapper.scoring import score_row
from grc_mapper.text import TokenProfile
from grc_mapper.textio import MAX_INPUT_BYTES, decode_input, read_stdin, read_text_file

SECRET = "ZQX-SENTINEL wording that must never leak"


# =============================================================== textio

def test_text_file_verbatim_and_bom(tmp_path):
    p = tmp_path / "s.txt"
    p.write_bytes(b"\xef\xbb\xbf  Line one\r\nline two \n")
    assert read_text_file(p) == "  Line one\r\nline two \n"
    p.write_bytes("caf\u00e9 policy".encode("utf-8"))
    assert read_text_file(p) == "caf\u00e9 policy"


def test_text_file_errors(tmp_path):
    bad = tmp_path / "b.txt"
    bad.write_bytes(b"caf\xe9")
    with pytest.raises(InputValidationError, match="not valid UTF-8"):
        read_text_file(bad)
    with pytest.raises(InputValidationError, match="not found: nope.txt"):
        read_text_file(tmp_path / "nope.txt")
    big = tmp_path / "big.txt"
    big.write_bytes(b"a" * (MAX_INPUT_BYTES + 1))
    with pytest.raises(InputValidationError, match="larger than"):
        read_text_file(big)


def test_stdin_bytes():
    assert read_stdin(io.BytesIO("abc \u00e9\n".encode("utf-8"))) == "abc \u00e9\n"
    with pytest.raises(InputValidationError, match="stdin"):
        read_stdin(io.BytesIO(b"\xff\xfe"))
    assert decode_input(b"", "x") == ""


# =============================================================== FakeEncoder

def test_fake_encoder_contract():
    fe = FakeEncoder()
    out = fe.encode(["a", "b", "a"])
    assert out.shape == (3, 384) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-6) and np.array_equal(out[0], out[2])
    assert fe.kind == FAKE_KIND == "FAKE_SHA256_DEMO"
    assert fe.token_profile(" ".join(["w"] * 300)) == TokenProfile(302, 256, True)


def test_fake_encoder_identical_across_processes():
    code = ("import hashlib; from grc_mapper.fake_encoder import FakeEncoder; "
            "print(hashlib.sha256(FakeEncoder().encode(['alpha','beta']).tobytes()).hexdigest())")
    outs = {subprocess.run([sys.executable, "-c", code], env=child_env(PYTHONHASHSEED=seed),
                           capture_output=True, text=True, check=True).stdout for seed in ("0", "1", "12345")}
    assert len(outs) == 1


def test_no_builtin_hash_in_package():
    for p in (REPO_ROOT / "src" / "grc_mapper").glob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "hash"]
        assert not calls, p.name


# =============================================================== logging

def _record(msg, *args):
    return logging.LogRecord("grc_mapper.x", logging.WARNING, __file__, 1, msg, args, None)


def test_redacting_filter():
    f = RedactingFilter()
    r = _record("item %s text %s", Statement("P1", SECRET), "x" * 300)
    f.filter(r)
    msg = r.getMessage()
    assert SECRET not in msg and "<statement P1>" in msg and "<redacted 300 chars>" in msg
    r2 = _record("y" * 500)
    f.filter(r2)
    assert r2.getMessage().startswith("<log message redacted")
    r3 = _record("short %s %d", "P1", 3)
    f.filter(r3)
    assert r3.getMessage() == "short P1 3"


# =============================================================== export / provenance

def _row(sid="P1", sim=0.41425902, k=3, prof=TokenProfile(5, 5, False)):
    cands = tuple(Candidate(i + 1, f"A.5.{i + 1}", sim - 0.01 * i) for i in range(k))
    return score_row(sid, cands, prof, "TEST_SHA256", load_settings(env={}))


def test_result_record_formats():
    rec = result_record(_row())
    assert list(rec) == list(RESULT_COLUMNS)
    assert rec["top1_similarity"] == "0.41425902" and rec["drift_index_r4"] == "0.5857" and rec["aps_r4"] == "0.8786"
    assert rec["aps_weight"] == "1.50" and rec["truncated"] == "false"
    one = result_record(_row(k=1))
    assert one["top2_control_id"] == "" and one["top3_similarity"] == ""


def test_csv_and_json_writers(tmp_path):
    rows = [_row("P1", 0.7), _row("P2", 0.3), _row("P3", 0.5)]
    written = []
    write_results_csv(rows, tmp_path / "r.csv", written=written)
    write_queue_csv(rows, tmp_path / "q.csv", written=written)
    data = write_json({"a": 1}, tmp_path / "x.json", written=written)
    assert written == ["r.csv", "q.csv", "x.json"]
    raw = (tmp_path / "r.csv").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r\n" not in raw
    q = list(csv.DictReader(io.StringIO((tmp_path / "q.csv").read_text(encoding="utf-8"))))
    assert [r["statement_id"] for r in q] == ["P2", "P3", "P1"] and [r["queue_rank"] for r in q] == ["1", "2", "3"]
    assert data == (tmp_path / "x.json").read_bytes() and data.endswith(b"\n")
    with pytest.raises(OutputWriteError, match="already written"):
        write_results_csv(rows, tmp_path / "r.csv", written=written)  # never overwrite


def test_include_text_column(tmp_path):
    rows = [_row("P1")]
    with pytest.raises(ConfigError):
        write_results_csv(rows, tmp_path / "a.csv", include_text=True)
    write_results_csv(rows, tmp_path / "b.csv", include_text=True, texts={"P1": SECRET})
    assert SECRET in (tmp_path / "b.csv").read_text(encoding="utf-8")
    write_results_csv(rows, tmp_path / "c.csv")
    assert SECRET not in (tmp_path / "c.csv").read_text(encoding="utf-8")


def test_output_location_and_run_dir(tmp_path):
    with pytest.raises(ConfigError, match="outside the public repository"):
        check_output_location(REPO_ROOT / "outputs")
    with pytest.raises(ConfigError):
        check_output_location(REPO_ROOT)
    d = create_run_dir(tmp_path / "out", "RUN1")
    assert d.is_dir()
    with pytest.raises(ConfigError, match="already exists"):
        create_run_dir(tmp_path / "out", "RUN1")


def test_provenance_has_no_text_and_redacts_paths(tmp_path):
    s = load_settings({"model_dir": str(tmp_path / "secret_dir" / "model")}, env={})
    rec = build_run_record(
        run_id=new_run_id(), mode="map-one", started_utc="t", settings=s, encoder_kind="SBERT_PRODUCTION",
        model_fingerprint="f" * 64, model_files_verified=8, catalog_source_kind="research", controls_sha256="0" * 64,
        rows=[_row(prof=TokenProfile(300, 256, True))], controls_truncated_ids=["A.5.9"], inputs={"input_mode": "stdin"},
    )
    blob = json.dumps(rec)
    assert "secret_dir" not in blob and rec["settings"]["paths"]["model_dir"] == "model"
    assert rec["disclaimer"] == DISCLAIMER and rec["demo_disclaimer"] is None
    assert rec["truncated_statement_ids"] == ["P1"] and rec["truncated_control_ids"] == ["A.5.9"]
    assert rec["review_status"] == "CANDIDATE_PENDING_HUMAN_REVIEW"
    assert "drift over time" not in blob.lower() and "not temporal drift" in DISCLAIMER


def test_result_json_contains_no_text():
    d = result_json(_row(), run_id="R")
    assert d["run_id"] == "R" and len(d["candidates"]) == 3
    assert json_bytes(d).decode("ascii")


# =============================================================== CLI (no real model)

CTRL_CSV = "Control_ID,Control_Text\nA.5.1,access rights review\nA.5.2,backup and restore of data\nA.5.3,incident reporting duties\n"
STMT_CSV = "Statement_ID,Statement_Text\nS1,users review access rights\nS2,backups are restored quarterly\n"


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Local config with synthetic controls; load_encoder patched to the isolated test encoder."""
    ctrl = tmp_path / "controls.csv"
    ctrl.write_text(CTRL_CSV, encoding="utf-8")
    stm = tmp_path / "statements.csv"
    stm.write_text(STMT_CSV, encoding="utf-8")
    cfg = tmp_path / "local.toml"
    cfg.write_text(f"[paths]\ncontrols_csv = '{ctrl}'\n", encoding="utf-8")

    class Report:
        runtime_files_verified = 8

    def fake_load(settings):
        e = TestEncoder()
        e.report = Report()
        return e

    monkeypatch.setattr(cli, "load_encoder", fake_load)
    monkeypatch.delenv("GRC_CONFIG", raising=False)
    return {"cfg": str(cfg), "stm": str(stm), "tmp": tmp_path}


def _run(argv, capsysbinary):
    code = cli.main(argv)
    out = capsysbinary.readouterr()
    return code, out.out, out.err.decode("utf-8", "replace")


def test_map_one_text_file_no_out_writes_nothing(cli_env, capsysbinary):
    tf = cli_env["tmp"] / "one.txt"
    tf.write_text(SECRET + " users review access rights", encoding="utf-8")
    before = sorted(p.name for p in cli_env["tmp"].iterdir())
    code, out, err = _run(["map-one", "--config", cli_env["cfg"], "--text-file", str(tf)], capsysbinary)
    assert code == 0 and sorted(p.name for p in cli_env["tmp"].iterdir()) == before
    d = json.loads(out)
    assert d["statement_id"] == "ADHOC-0001" and d["review_status"] == "CANDIDATE_PENDING_HUMAN_REVIEW"
    assert SECRET.encode() not in out and SECRET not in err


def test_map_one_out_writes_exactly_three_files(cli_env, capsysbinary):
    tf = cli_env["tmp"] / "one.txt"
    tf.write_text(SECRET + " backups are restored", encoding="utf-8")
    out_dir = cli_env["tmp"] / "out"
    code, out, err = _run(["map-one", "--config", cli_env["cfg"], "--text-file", str(tf), "--out", str(out_dir),
                           "--statement-id", "Q-7"], capsysbinary)
    assert code == 0
    runs = list(out_dir.iterdir())
    assert len(runs) == 1
    files = sorted(p.name for p in runs[0].iterdir())
    assert files == ["provenance.json", "result.csv", "result.json"]
    assert (runs[0] / "result.json").read_bytes() == out
    for f in files:
        blob = (runs[0] / f).read_text(encoding="utf-8")
        assert SECRET not in blob and str(cli_env["tmp"]) not in blob
    prov = json.loads((runs[0] / "provenance.json").read_text())
    assert prov["inputs"]["input_mode"] == "text-file" and prov["inputs"]["input_file_basename"] == "one.txt"
    assert len(prov["inputs"]["input_sha256"]) == 64 and prov["mode"] == "map-one"


def test_map_one_stdin_and_text_warning(cli_env, capsysbinary, monkeypatch):
    class FakeStdin:
        buffer = io.BytesIO("incident reporting duties for staff".encode("utf-8"))

    monkeypatch.setattr(sys, "stdin", FakeStdin())
    code, out, _ = _run(["map-one", "--config", cli_env["cfg"], "--stdin"], capsysbinary)
    assert code == 0 and json.loads(out)["top1_control_id"].startswith("A.5.")
    code, out, err = _run(["map-one", "--config", cli_env["cfg"], "--text", "access rights review"], capsysbinary)
    assert code == 0 and "shell history" in err


def test_map_one_rejections(cli_env, capsysbinary):
    with pytest.raises(SystemExit) as e:
        cli.main(["map-one", "--config", cli_env["cfg"], "--text", "a", "--stdin"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        cli.main(["map-one", "--config", cli_env["cfg"], "--text", "abc", "--include-text"])
    assert e.value.code == 2
    code, _, err = _run(["map-one", "--config", cli_env["cfg"], "--text", "abc def", "--statement-id", "bad id"], capsysbinary)
    assert code == 4
    code, _, err = _run(["map-one", "--config", cli_env["cfg"], "--text", "abc def", "--out", str(REPO_ROOT / "x")], capsysbinary)
    assert code == 2 and "outside the public repository" in err
    code, _, err = _run(["map-one", "--config", cli_env["cfg"], "--text", "   "], capsysbinary)
    assert code == 4


def test_map_batch(cli_env, capsysbinary):
    out_dir = cli_env["tmp"] / "batch"
    code, out, _ = _run(["map-batch", "--config", cli_env["cfg"], "--statements", cli_env["stm"], "--out", str(out_dir)],
                        capsysbinary)
    assert code == 0
    summary = json.loads(out)
    run = out_dir / summary["run_id"]
    assert sorted(p.name for p in run.iterdir()) == ["provenance.json", "queue.csv", "results.csv"]
    assert summary["n_statements"] == 2 and sum(summary["tier_counts"].values()) == 2
    header = (run / "results.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == list(RESULT_COLUMNS)
    code, out, _ = _run(["map-batch", "--config", cli_env["cfg"], "--statements", cli_env["stm"], "--out", str(out_dir),
                         "--include-text"], capsysbinary)
    run2 = out_dir / json.loads(out)["run_id"]
    assert "statement_text" in (run2 / "results.csv").read_text(encoding="utf-8").splitlines()[0]
    assert json.loads((run2 / "provenance.json").read_text())["include_text"] is True


def test_map_batch_research_profile_counts(cli_env, capsysbinary):
    code, _, err = _run(["map-batch", "--config", cli_env["cfg"], "--statements", cli_env["stm"], "--profile", "research",
                         "--out", str(cli_env["tmp"] / "o")], capsysbinary)
    assert code == 4 and "expected 93" in err


def test_verify_model_exit_codes(fake_model, tmp_path, capsysbinary, monkeypatch):
    monkeypatch.delenv("GRC_CONFIG", raising=False)
    model, rt, full = fake_model
    cfg_full = tmp_path / "f.toml"
    cfg_full.write_text(f"[paths]\nmodel_dir = '{model}'\nmodel_manifest = '{rt}'\nmodel_full_manifest = '{full}'\n")
    code, out, _ = _run(["verify-model", "--config", str(cfg_full)], capsysbinary)
    assert code == 0 and json.loads(out)["status"] == "OK_FULL"
    cfg_part = tmp_path / "p.toml"
    cfg_part.write_text(f"[paths]\nmodel_dir = '{model}'\nmodel_manifest = '{rt}'\n")
    code, out, _ = _run(["verify-model", "--config", str(cfg_part)], capsysbinary)
    assert code == 7 and json.loads(out)["status"] == "OK_PARTIAL"
    code, _, err = _run(["verify-model", "--config", str(cfg_part), "--load"], capsysbinary)
    assert code == 3 and "full model manifest required" in err
    (model / "config.json").write_bytes(b"tampered!")
    code, _, _ = _run(["verify-model", "--config", str(cfg_full)], capsysbinary)
    assert code == 3


def test_verify_inputs(cli_env, capsysbinary):
    code, out, _ = _run(["verify-inputs", "--config", cli_env["cfg"], "--statements", cli_env["stm"]], capsysbinary)
    d = json.loads(out)
    assert code == 0 and d["controls"]["rows"] == 3 and d["statements"]["rows"] == 2 and d["status"] == "OK"
    code, _, err = _run(["verify-inputs", "--config", cli_env["cfg"], "--statements", cli_env["stm"], "--profile",
                         "research"], capsysbinary)
    assert code == 2 and "columns.label" in err


def test_demo_stdout_and_out(tmp_path, capsysbinary):
    code, out, err = _run(["demo"], capsysbinary)
    d = json.loads(out)
    assert code == 0 and "SOFTWARE-PIPELINE DEMONSTRATION" in err
    assert len(d["rows"]) == 12 and all(r["demo_disclaimer"] == DEMO_DISCLAIMER for r in d["rows"])
    assert all(r["encoder_kind"] == "FAKE_SHA256_DEMO" for r in d["rows"])
    assert sorted(d["queue"]) == sorted(r["statement_id"] for r in d["rows"])
    code, out, err = _run(["demo", "--out", str(tmp_path / "demo")], capsysbinary)
    run = tmp_path / "demo" / json.loads(out)["run_id"]
    assert code == 0 and "DEMO COMPLETE: 12 synthetic statements mapped to 8 synthetic controls" in err
    assert f"output directory: {run.resolve()}" in err
    assert sorted(p.name for p in run.iterdir()) == ["provenance.json", "queue.csv", "results.csv"]
    rows = list(csv.DictReader(io.StringIO((run / "results.csv").read_text(encoding="utf-8"))))
    assert len(rows) == 12 and all(r["demo_disclaimer"] == DEMO_DISCLAIMER for r in rows)
    prov = json.loads((run / "provenance.json").read_text())
    assert prov["demo_disclaimer"] == DEMO_DISCLAIMER and prov["model_runtime_manifest_sha256"] is None


def test_demo_is_deterministic_and_never_loads_a_model(tmp_path):
    code = ("import sys, json, io, contextlib; from grc_mapper import cli; buf = io.BytesIO()\n"
            "class W:\n    buffer = buf\n    def flush(self): pass\n"
            "sys.stdout = W(); rc = cli.main(['demo'])\n"
            "d = json.loads(buf.getvalue()); sys.stdout = sys.__stdout__\n"
            "bad = [m for m in ('sentence_transformers','transformers','torch') if m in sys.modules]\n"
            "print(json.dumps({'rc': rc, 'bad': bad, 'rows': [(r['statement_id'], r['top1_control_id'], r['aps']) for r in d['rows']]}))")
    outs = [json.loads(subprocess.run([sys.executable, "-c", code], env=child_env(GRC_MODEL_DIR=str(tmp_path)),
                                      capture_output=True, text=True, check=True).stdout) for _ in range(2)]
    assert outs[0]["rc"] == 0 and outs[0]["bad"] == [] and outs[0]["rows"] == outs[1]["rows"]


def test_python_dash_m_entry_point(tmp_path):
    r = subprocess.run([sys.executable, "-m", "grc_mapper", "demo"], env=child_env(), capture_output=True, cwd=tmp_path)
    assert r.returncode == 0 and json.loads(r.stdout)["rows"]
    r = subprocess.run([sys.executable, "-m", "grc_mapper"], env=child_env(), capture_output=True, cwd=tmp_path)
    assert r.returncode == 2
