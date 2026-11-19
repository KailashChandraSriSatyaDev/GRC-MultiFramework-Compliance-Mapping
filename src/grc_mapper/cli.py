"""Command-line interface (Blueprint rev 3, section 3.8).

    python -m grc_mapper verify-model   [--config PATH] [--load]
    python -m grc_mapper verify-inputs  [--controls PATH] [--statements PATH] [--profile research|generic]
    python -m grc_mapper map-batch      --statements PATH --out DIR [--controls PATH] [--top-k N] [--include-text]
    python -m grc_mapper map-one        (--text-file PATH | --stdin | --text STR) [--out DIR] [--statement-id ID]
    python -m grc_mapper demo           [--out DIR]
    python -m grc_mapper map-multi      --statements PATH --frameworks SPEC.json --out DIR
                                        [--select-column COL --select-value VALUE] [--only ID[,ID]]

Exit codes: 0 ok, 1 unexpected error, 2 config/usage, 3 integrity, 4 input, 5 model,
6 write failure, 7 verification incomplete (verify-model only).

Every production-model command verifies the model directory to OK_FULL before loading it.
`demo` uses the hash-seeded FakeEncoder and the synthetic examples only; it never loads a model.
stdout carries JSON results (no policy or control text); diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, Settings, load_settings
from .data import load_control_catalog, load_statements, read_csv
from .errors import ConfigError, GrcError, IntegrityError
from .export import (
    _csv_bytes, check_output_location, create_run_dir, json_bytes, result_json, write_bytes, write_json,
    write_queue_csv, write_results_csv,
)
from .fake_encoder import FakeEncoder
from .frameworks import load_framework_catalogue, load_framework_spec
from .history import PREPROCESSING_TAG
from .integrity import load_full_manifest, load_runtime_manifest, sha256_file, verify_model_dir
from .multiframework import FrameworkIndex, map_multi
from .logging_setup import configure_logging
from .pipeline import build_control_index, load_encoder, map_single, map_statements
from .provenance import DEMO_DISCLAIMER, DISCLAIMER, build_run_record, library_versions, new_run_id, tier_counts, utc_now
from .scoring import priority_queue
from .errors import InputValidationError
from .text import text_fingerprint
from .textio import read_stdin, read_text_file

log = logging.getLogger("grc_mapper.cli")

SYNTHETIC_DIR = REPO_ROOT / "examples" / "synthetic"
DEMO_CONTROLS = SYNTHETIC_DIR / "demo_controls.csv"
DEMO_STATEMENTS = SYNTHETIC_DIR / "demo_statements.csv"
DEMO_COLUMNS = {"control_id": "Control_ID", "control_text": "Control_Text",
                "statement_id": "Statement_ID", "statement_text": "Statement_Text"}
RESEARCH_COUNTS = {"controls": 93, "statements": 206}
TEXT_ARG_WARNING = ("warning: --text puts the statement on the command line, where it can persist in "
                    "shell history and process listings; prefer --text-file or --stdin for real policy text")


# ------------------------------------------------------------------------------------------ helpers

def _emit(obj: Any) -> bytes:
    """Write JSON to stdout as raw bytes (LF line endings on every platform)."""
    data = json_bytes(obj)
    sys.stdout.flush()
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
    return data


def _settings(args: argparse.Namespace, **extra: Any) -> Settings:
    over: dict[str, Any] = {"config": getattr(args, "config", None)}
    for key in ("top_k", "batch_size", "device"):
        if getattr(args, key, None) is not None:
            over[key] = getattr(args, key)
    if getattr(args, "controls", None):
        over["controls_csv"] = args.controls
    over.update(extra)
    return load_settings(over)


def _load_research_catalog(s: Settings, profile: str):
    return load_control_catalog(
        s.require_path("controls_csv"),
        id_column=s.columns["control_id"],
        text_column=s.columns["control_text"],
        expected_count=RESEARCH_COUNTS["controls"] if profile == "research" else None,
        source_kind="research",
    )


# ------------------------------------------------------------------------------------------ commands

def cmd_verify_model(args: argparse.Namespace) -> int:
    s = _settings(args)
    runtime = load_runtime_manifest(s.require_path("model_manifest"))
    full_path = s.path("model_full_manifest")
    full = load_full_manifest(full_path) if full_path is not None else None
    report = verify_model_dir(s.require_path("model_dir"), runtime, full)
    out: dict[str, Any] = {
        "status": report.status,
        "runtime_files_verified": report.runtime_files_verified,
        "non_runtime": dict(report.non_runtime),
        "directories": list(report.directories),
        "runtime_manifest_sha256": report.runtime_manifest_sha256,
        "full_manifest_sha256": report.full_manifest_sha256,
    }
    if report.status != "OK_FULL":
        out["note"] = "verification incomplete: OK_PARTIAL is diagnostic only and never permits inference"
        if args.load:
            _emit(out)
            raise IntegrityError("full model manifest required for production-model inference")
        _emit(out)
        return 7
    if args.load:
        enc = load_encoder(s)
        out["load"] = {"offline": True, "device": s.device, "embedding_dim": enc.embedding_dim,
                       "max_seq_length": enc.max_seq_length, "post_load_reverification": "OK_FULL"}
    _emit(out)
    return 0


def cmd_verify_inputs(args: argparse.Namespace) -> int:
    s = _settings(args)
    label = s.columns["label"] if args.profile == "research" else None
    if args.profile == "research" and not label:
        raise ConfigError("the research profile needs columns.label in the configuration")
    cat = _load_research_catalog(s, args.profile)
    stmt_path = Path(args.statements) if args.statements else s.require_path("policies_csv")
    stmts, labels = load_statements(
        stmt_path, id_column=s.columns["statement_id"], text_column=s.columns["statement_text"],
        label_column=label, known_controls=cat if label else None,
        expected_count=RESEARCH_COUNTS["statements"] if args.profile == "research" else None,
    )
    _emit({
        "profile": args.profile,
        "controls": {"file": s.require_path("controls_csv").name, "rows": len(cat), "sha256": cat.source_sha256},
        "statements": {"file": stmt_path.name, "rows": len(stmts), "sha256": sha256_file(stmt_path),
                       "labels_checked": labels is not None,
                       "distinct_labels": len(set(labels.values())) if labels else None},
        "status": "OK",
    })
    return 0


def cmd_map_batch(args: argparse.Namespace) -> int:
    s = _settings(args)
    check_output_location(Path(args.out))
    started = utc_now()
    cat = _load_research_catalog(s, args.profile)
    stmt_path = Path(args.statements)
    stmts, _ = load_statements(stmt_path, id_column=s.columns["statement_id"], text_column=s.columns["statement_text"],
                               expected_count=RESEARCH_COUNTS["statements"] if args.profile == "research" else None)
    enc = load_encoder(s)
    idx = build_control_index(cat, enc, s)
    rows = map_statements(stmts, idx, enc, s)
    run_id = new_run_id()
    run_dir = create_run_dir(Path(args.out), run_id)
    written: list[str] = []
    texts = {st.statement_id: st.text for st in stmts} if args.include_text else None
    write_results_csv(rows, run_dir / "results.csv", include_text=args.include_text, texts=texts, written=written)
    write_queue_csv(rows, run_dir / "queue.csv", written=written)
    record = build_run_record(
        run_id=run_id, mode="map-batch", started_utc=started, settings=s, encoder_kind=idx.encoder_kind,
        model_fingerprint=idx.model_fingerprint, model_files_verified=enc.report.runtime_files_verified,
        catalog_source_kind=cat.source_kind, controls_sha256=cat.source_sha256, rows=rows,
        controls_truncated_ids=idx.truncated_control_ids,
        inputs={"statements_csv": stmt_path.name, "statements_csv_sha256": sha256_file(stmt_path), "profile": args.profile},
        include_text=args.include_text,
    )
    write_json(record, run_dir / "provenance.json", written=written)
    _emit({"run_id": run_id, "run_dir": run_dir.name, "files": written, "n_statements": len(rows),
           "tier_counts": tier_counts(rows), "review_status": "CANDIDATE_PENDING_HUMAN_REVIEW"})
    return 0


def cmd_map_one(args: argparse.Namespace) -> int:
    if args.text_file is not None:
        text, mode, extra_in = read_text_file(Path(args.text_file)), "text-file", {"input_file_basename": Path(args.text_file).name}
    elif args.stdin:
        text, mode, extra_in = read_stdin(), "stdin", {}
    else:
        print(TEXT_ARG_WARNING, file=sys.stderr)
        text, mode, extra_in = args.text, "text", {}
    s = _settings(args)
    if args.out is not None:
        check_output_location(Path(args.out))
    started = utc_now()
    cat = _load_research_catalog(s, "generic")
    enc = load_encoder(s)
    idx = build_control_index(cat, enc, s)
    r = map_single(text, idx, enc, s, statement_id=args.statement_id)
    run_id = new_run_id()
    data = _emit(result_json(r, run_id=run_id))
    if args.out is None:
        return 0
    run_dir = create_run_dir(Path(args.out), run_id)
    written: list[str] = []
    write_bytes(run_dir / "result.json", data, written)  # byte-identical to stdout
    write_results_csv([r], run_dir / "result.csv", written=written)
    digest, n_chars = text_fingerprint(text)
    record = build_run_record(
        run_id=run_id, mode="map-one", started_utc=started, settings=s, encoder_kind=idx.encoder_kind,
        model_fingerprint=idx.model_fingerprint, model_files_verified=enc.report.runtime_files_verified,
        catalog_source_kind=cat.source_kind, controls_sha256=cat.source_sha256, rows=[r],
        controls_truncated_ids=idx.truncated_control_ids,
        inputs={"input_mode": mode, "input_sha256": digest, "input_char_count": n_chars, **extra_in},
    )
    write_json(record, run_dir / "provenance.json", written=written)
    print(f"wrote {written} to run directory {run_id}", file=sys.stderr)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    print(DEMO_DISCLAIMER, file=sys.stderr)
    s = load_settings({"top_k": 3}, env={})  # demo ignores local configuration entirely
    if args.out is not None:
        check_output_location(Path(args.out))
    started = utc_now()
    enc = FakeEncoder()
    cat = load_control_catalog(DEMO_CONTROLS, id_column=DEMO_COLUMNS["control_id"],
                               text_column=DEMO_COLUMNS["control_text"], expected_count=8, source_kind="synthetic")
    stmts, _ = load_statements(DEMO_STATEMENTS, id_column=DEMO_COLUMNS["statement_id"],
                               text_column=DEMO_COLUMNS["statement_text"], expected_count=12)
    idx = build_control_index(cat, enc, s)
    rows = map_statements(stmts, idx, enc, s)
    run_id = new_run_id()
    _emit({
        "demo_disclaimer": DEMO_DISCLAIMER,
        "run_id": run_id,
        "rows": [{**result_json(r), "demo_disclaimer": DEMO_DISCLAIMER} for r in rows],
        "queue": [r.statement_id for r in priority_queue(rows)],
    })
    if args.out is None:
        return 0
    run_dir = create_run_dir(Path(args.out), run_id)
    written: list[str] = []
    extra = {"demo_disclaimer": DEMO_DISCLAIMER}
    write_results_csv(rows, run_dir / "results.csv", extra=extra, written=written)
    write_queue_csv(rows, run_dir / "queue.csv", extra=extra, written=written)
    record = build_run_record(
        run_id=run_id, mode="demo", started_utc=started, settings=s, encoder_kind=idx.encoder_kind,
        model_fingerprint=idx.model_fingerprint, model_files_verified=None, catalog_source_kind=cat.source_kind,
        controls_sha256=cat.source_sha256, rows=rows, controls_truncated_ids=idx.truncated_control_ids,
        inputs={"statements_csv": DEMO_STATEMENTS.name, "statements_csv_sha256": sha256_file(DEMO_STATEMENTS)},
        demo=True,
    )
    write_json(record, run_dir / "provenance.json", written=written)
    print(f"wrote {written} to run directory {run_id}", file=sys.stderr)
    print(f"DEMO COMPLETE: {len(rows)} synthetic statements mapped to {len(cat.control_ids)} synthetic controls; "
          f"bands {tier_counts(rows)}", file=sys.stderr)
    print(f"output directory: {run_dir.resolve()}", file=sys.stderr)
    return 0


MULTI_DISCLAIMER = (
    "Each framework is mapped and scored independently; scores from different frameworks are not comparable "
    "and are never ranked together. Review bands outside ISO/IEC 27001 reuse ISO-derived heuristic cut-offs "
    "unless the framework spec states otherwise; they are not calibrated for that framework. Candidate "
    "mappings across frameworks are not an equivalence mapping or a crosswalk."
)


def cmd_map_multi(args: argparse.Namespace) -> int:
    s = _settings(args)
    check_output_location(Path(args.out))
    started = utc_now()
    specs = load_framework_spec(Path(args.frameworks))
    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        unknown = sorted(set(wanted) - {sp.framework_id for sp in specs})
        if unknown:
            raise ConfigError(f"--only names unknown framework IDs {unknown}")
        specs = [sp for sp in specs if sp.framework_id in wanted]
    catalogues = {sp.framework_id: load_framework_catalogue(sp) for sp in specs}
    stmt_path = Path(args.statements)
    id_col, text_col = s.columns["statement_id"], s.columns["statement_text"]
    stmts, _ = load_statements(stmt_path, id_column=id_col, text_column=text_col)
    if (args.select_column is None) != (args.select_value is None):
        raise ConfigError("--select-column and --select-value must be given together")
    if args.select_column is not None:
        rows, _ = read_csv(stmt_path, [id_col, args.select_column])
        value = {r[id_col]: r[args.select_column] for r in rows}
        selected = [st for st in stmts if value[st.statement_id] == args.select_value]
        excluded = [(st.statement_id, value[st.statement_id]) for st in stmts if value[st.statement_id] != args.select_value]
        if not selected:
            raise InputValidationError(f"no rows with {args.select_column} == {args.select_value!r}")
    else:
        selected, excluded = stmts, []
    enc = load_encoder(s)
    fws = [FrameworkIndex(sp, build_control_index(catalogues[sp.framework_id], enc, sp.scoring_settings(s)))
           for sp in specs]
    results = map_multi(selected, fws, enc, s)
    sel_ids = [st.statement_id for st in selected]
    for fid, rows_ in results.items():
        if [r.statement_id for r in rows_] != sel_ids:
            raise GrcError(f"{fid}: result rows do not match the selected statements one-to-one")

    run_id = new_run_id()
    run_dir = create_run_dir(Path(args.out), run_id)
    written: list[str] = []
    fw_info = []
    for fw in fws:
        fid, ver = fw.spec.framework_id, fw.spec.framework_version
        extra = {"framework_id": fid, "framework_version": ver}
        write_results_csv(results[fid], run_dir / f"results_{fid}.csv", extra=extra, written=written)
        write_queue_csv(results[fid], run_dir / f"queue_{fid}.csv", extra=extra, written=written)
        info = fw.spec.public_info(fw.index.catalog)
        info.update({"n_results": len(results[fid]), "tier_counts": tier_counts(results[fid]),
                     "truncated_control_ids": list(fw.index.truncated_control_ids),
                     "results_file": f"results_{fid}.csv", "queue_file": f"queue_{fid}.csv"})
        fw_info.append(info)
    sel_set = set(sel_ids)
    stmt_rows = []
    for st in stmts:
        digest, n = text_fingerprint(st.text)
        stmt_rows.append({"statement_id": st.statement_id, "text_sha256": digest, "n_chars": str(n),
                          "selected": "true" if st.statement_id in sel_set else "false"})
    write_bytes(run_dir / "statements.csv",
                _csv_bytes(("statement_id", "text_sha256", "n_chars", "selected"), stmt_rows), written)
    write_bytes(run_dir / "excluded.csv",
                _csv_bytes(("statement_id", "select_column", "value"),
                           [{"statement_id": i, "select_column": args.select_column or "", "value": v}
                            for i, v in excluded]),
                written)
    write_bytes(run_dir / "rejected.csv", _csv_bytes(("statement_id", "reason"), []), written)  # validation is all-or-nothing
    runtime = load_runtime_manifest(s.require_path("model_manifest"))
    weights = next((e.sha256 for e in runtime.files if e.path == "model.safetensors"), None)
    record = {
        "schema_version": 1, "run_id": run_id, "mode": "map-multi", "package_version": __import__("grc_mapper").__version__,
        "encoder_kind": enc.kind, "model": {
            "verification_status": enc.report.status, "runtime_files_verified": enc.report.runtime_files_verified,
            "runtime_manifest_sha256": enc.report.runtime_manifest_sha256,
            "full_manifest_sha256": enc.report.full_manifest_sha256, "model_weights_sha256": weights},
        "preprocessing": PREPROCESSING_TAG,
        "inputs": {"statements_csv": stmt_path.name, "statements_csv_sha256": sha256_file(stmt_path),
                   "rows_total": len(stmts), "select_column": args.select_column, "select_value": args.select_value,
                   "rows_selected": len(selected), "rows_excluded": len(excluded), "rows_rejected": 0},
        "frameworks": fw_info, "settings": s.redacted(), "versions": library_versions(), "platform": sys.platform,
        "n_truncated_statements": sum(r.token_profile.truncated for r in next(iter(results.values()))),
        "include_text": False, "review_status": "CANDIDATE_PENDING_HUMAN_REVIEW",
        "started_utc": started, "finished_utc": utc_now(), "disclaimer": DISCLAIMER,
        "multi_framework_disclaimer": MULTI_DISCLAIMER,
    }
    write_json(record, run_dir / "provenance.json", written=written)
    _emit({"run_id": run_id, "run_dir": run_dir.name, "files": written, "rows_selected": len(selected),
           "rows_excluded": len(excluded), "frameworks": {i["framework_id"]: i["tier_counts"] for i in fw_info},
           "review_status": "CANDIDATE_PENDING_HUMAN_REVIEW"})
    return 0


# ------------------------------------------------------------------------------------------ parser

def _positive_int(v: str) -> int:
    try:
        n = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="local TOML configuration (overrides GRC_CONFIG)")
    common.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug (stderr)")

    p = argparse.ArgumentParser(prog="python -m grc_mapper",
                                description="Candidate mapping of policy statements to ISO/IEC 27001:2022 controls "
                                            "for human review.")
    sub = p.add_subparsers(dest="command", required=True)

    vm = sub.add_parser("verify-model", parents=[common], help="verify the model directory (OK_FULL / OK_PARTIAL)")
    vm.add_argument("--load", action="store_true", help="also load the model offline after OK_FULL")

    vi = sub.add_parser("verify-inputs", parents=[common], help="validate control and statement CSVs (no model)")
    vi.add_argument("--controls")
    vi.add_argument("--statements")
    vi.add_argument("--profile", choices=("research", "generic"), default="generic")

    mb = sub.add_parser("map-batch", parents=[common], help="map a CSV of statements")
    mb.add_argument("--statements", required=True)
    mb.add_argument("--out", required=True, help="output directory outside the public repository")
    mb.add_argument("--controls")
    mb.add_argument("--profile", choices=("research", "generic"), default="generic")
    mb.add_argument("--top-k", dest="top_k", type=_positive_int)
    mb.add_argument("--batch-size", dest="batch_size", type=_positive_int)
    mb.add_argument("--device")
    mb.add_argument("--include-text", action="store_true", help="add statement_text to results.csv")

    mo = sub.add_parser("map-one", parents=[common], help="map one statement")
    src = mo.add_mutually_exclusive_group(required=True)
    src.add_argument("--text-file", help="UTF-8 file holding the statement (recommended)")
    src.add_argument("--stdin", action="store_true", help="read the statement from stdin (UTF-8)")
    src.add_argument("--text", help="statement on the command line (synthetic / quick checks only)")
    mo.add_argument("--controls")
    mo.add_argument("--top-k", dest="top_k", type=_positive_int)
    mo.add_argument("--device")
    mo.add_argument("--out", help="also save result.json, result.csv and provenance.json under OUT/<run_id>")
    mo.add_argument("--statement-id", default="ADHOC-0001")

    dm = sub.add_parser("demo", parents=[common], help="software-pipeline demo with the FakeEncoder")
    dm.add_argument("--out")

    mm = sub.add_parser("map-multi", parents=[common], help="map statements against several framework catalogues")
    mm.add_argument("--statements", required=True)
    mm.add_argument("--frameworks", required=True, help="framework spec JSON (local; outside the repository)")
    mm.add_argument("--out", required=True, help="output directory outside the public repository")
    mm.add_argument("--select-column", help="keep only rows whose COLUMN equals --select-value (others listed in excluded.csv)")
    mm.add_argument("--select-value")
    mm.add_argument("--only", help="comma-separated framework IDs from the spec")
    mm.add_argument("--top-k", dest="top_k", type=_positive_int)
    mm.add_argument("--batch-size", dest="batch_size", type=_positive_int)
    mm.add_argument("--device")
    return p


COMMANDS = {
    "verify-model": cmd_verify_model,
    "verify-inputs": cmd_verify_inputs,
    "map-batch": cmd_map_batch,
    "map-one": cmd_map_one,
    "demo": cmd_demo,
    "map-multi": cmd_map_multi,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        return COMMANDS[args.command](args)
    except GrcError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # unexpected: report the type only
        print(f"error: unexpected {exc.__class__.__name__}", file=sys.stderr)
        return 1
