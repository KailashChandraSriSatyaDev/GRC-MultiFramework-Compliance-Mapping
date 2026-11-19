"""Result, queue and provenance writers (Blueprint rev 3, section 3.8).

- Statement text is never exported unless include_text=True, and real-data or text-bearing
  output is never written inside the public repository.
- CSV: UTF-8 without BOM, LF line endings; unrounded floats as %.8f, legacy 4-decimal values
  as %.4f. Every CSV cell passes through csvcell.neutralise (formula-injection guard; plain
  numbers are unchanged); quoting of commas, quotes and newlines is done by the csv module. JSON: UTF-8 without BOM, ASCII-escaped, LF, 2-space indent.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .csvcell import neutralise
from .errors import ConfigError, OutputWriteError
from .scoring import ScoredMapping, priority_queue

RESULT_COLUMNS = (
    "statement_id", "top1_control_id", "top1_similarity", "tier_code", "tier_label",
    "drift_index", "aps_weight", "aps", "drift_index_r4", "aps_r4",
    "top2_control_id", "top2_similarity", "top3_control_id", "top3_similarity",
    "n_tokens_full", "n_tokens_used", "truncated", "encoder_kind", "review_status",
)


def _f8(x: float) -> str:
    return f"{x:.8f}"


def _f4(x: float) -> str:
    return f"{x:.4f}"


def is_inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def check_output_location(out_dir: Path, repo_root: Path = REPO_ROOT) -> Path:
    """Refuse an output directory inside the public repository."""
    p = Path(out_dir)
    if is_inside(p, repo_root):
        raise ConfigError("output directory must be outside the public repository")
    return p


def create_run_dir(out_dir: Path, run_id: str, repo_root: Path = REPO_ROOT) -> Path:
    base = check_output_location(out_dir, repo_root)
    run_dir = base / run_id
    if run_dir.exists():
        raise ConfigError(f"run directory already exists: {run_id}")
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise OutputWriteError(f"cannot create run directory {run_id} ({exc.__class__.__name__})") from None
    return run_dir


def result_record(r: ScoredMapping) -> dict[str, str]:
    """One CSV row (strings) in RESULT_COLUMNS order."""
    c = r.candidates
    rec = {
        "statement_id": r.statement_id,
        "top1_control_id": r.top1_control_id,
        "top1_similarity": _f8(r.top1_similarity),
        "tier_code": r.tier_code,
        "tier_label": r.tier_label,
        "drift_index": _f8(r.drift_index),
        "aps_weight": f"{r.aps_weight:.2f}",
        "aps": _f8(r.aps),
        "drift_index_r4": _f4(r.drift_index_r4),
        "aps_r4": _f4(r.aps_r4),
        "top2_control_id": c[1].control_id if len(c) > 1 else "",
        "top2_similarity": _f8(c[1].similarity) if len(c) > 1 else "",
        "top3_control_id": c[2].control_id if len(c) > 2 else "",
        "top3_similarity": _f8(c[2].similarity) if len(c) > 2 else "",
        "n_tokens_full": str(r.token_profile.n_tokens_full),
        "n_tokens_used": str(r.token_profile.n_tokens_used),
        "truncated": "true" if r.token_profile.truncated else "false",
        "encoder_kind": r.encoder_kind,
        "review_status": r.review_status,
    }
    return rec


def result_json(r: ScoredMapping, *, run_id: str | None = None) -> dict[str, Any]:
    """JSON-ready dict of every ScoredMapping field except text (there is none to omit)."""
    d: dict[str, Any] = {}
    if run_id is not None:
        d["run_id"] = run_id
    d.update({
        "statement_id": r.statement_id,
        "top1_control_id": r.top1_control_id,
        "top1_similarity": r.top1_similarity,
        "tier_code": r.tier_code,
        "tier_label": r.tier_label,
        "drift_index": r.drift_index,
        "aps_weight": r.aps_weight,
        "aps": r.aps,
        "drift_index_r4": r.drift_index_r4,
        "aps_r4": r.aps_r4,
        "candidates": [{"rank": c.rank, "control_id": c.control_id, "similarity": c.similarity} for c in r.candidates],
        "n_tokens_full": r.token_profile.n_tokens_full,
        "n_tokens_used": r.token_profile.n_tokens_used,
        "truncated": r.token_profile.truncated,
        "encoder_kind": r.encoder_kind,
        "review_status": r.review_status,
    })
    return d


def json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _csv_bytes(header: Sequence[str], rows: Sequence[Mapping[str, str]]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(header), lineterminator="\n", extrasaction="raise")
    w.writeheader()
    for row in rows:
        w.writerow({k: neutralise(v) for k, v in row.items()})
    return buf.getvalue().encode("utf-8")


def write_bytes(path: Path, data: bytes, written: list[str]) -> None:
    """Write bytes to a new file (never overwrites); record its name in `written`."""
    try:
        with open(path, "xb") as fh:  # never overwrite
            fh.write(data)
    except OSError as exc:
        raise OutputWriteError(
            f"cannot write {Path(path).name} ({exc.__class__.__name__}); already written: {written}"
        ) from None
    written.append(Path(path).name)


def write_results_csv(
    rows: Sequence[ScoredMapping],
    path: Path,
    *,
    include_text: bool = False,
    texts: Mapping[str, str] | None = None,
    extra: Mapping[str, str] | None = None,
    written: list[str] | None = None,
) -> None:
    header = list(RESULT_COLUMNS) + (["statement_text"] if include_text else []) + list(extra or {})
    recs = []
    for r in rows:
        rec = result_record(r)
        if include_text:
            if texts is None:
                raise ConfigError("include_text requires the statement texts")
            rec["statement_text"] = texts[r.statement_id]
        rec.update(extra or {})
        recs.append(rec)
    write_bytes(Path(path), _csv_bytes(header, recs), written if written is not None else [])


def write_queue_csv(
    rows: Sequence[ScoredMapping],
    path: Path,
    *,
    extra: Mapping[str, str] | None = None,
    written: list[str] | None = None,
) -> None:
    header = ["queue_rank"] + list(RESULT_COLUMNS) + list(extra or {})
    recs = []
    for i, r in enumerate(priority_queue(rows), start=1):
        rec = {"queue_rank": str(i), **result_record(r), **(extra or {})}
        recs.append(rec)
    write_bytes(Path(path), _csv_bytes(header, recs), written if written is not None else [])


def write_json(obj: Any, path: Path, *, written: list[str] | None = None) -> bytes:
    data = json_bytes(obj)
    write_bytes(Path(path), data, written if written is not None else [])
    return data
