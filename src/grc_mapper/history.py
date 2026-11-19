"""Run snapshots and previous-run comparison of top-1 snapshot dissimilarity (DI).

A snapshot is one framework's results for one run, keyed by stable statement ID and carrying a
SHA-256 of each statement's text (never the text). check_compatible() requires these snapshot
metadata fields to be equal, otherwise IncompatibleSnapshots is raised and no comparison is produced:
framework_id, framework_version, catalogue_sha256, model_weights_sha256 (model.safetensors only),
preprocessing tag, bands and aps_weights. Dataset hash, top-k and library versions are recorded by
callers but not checked here.

Rows are matched by statement ID. IDs present in one snapshot only are reported as only_previous /
only_current and get no dDI. For matched IDs:
  dDI = DI_current - DI_previous  (change in top-1 snapshot dissimilarity between two compatible runs)
Positive: higher dissimilarity in the current run; negative: lower; zero: unchanged recorded value.
A text-hash difference does not block the comparison; it is flagged per row (text_changed) and ranks
first in the change review queue. dDI therefore means:
  - same text hash: a same-text inference rerun (dDI reflects numerical/reproducibility differences);
  - different text hash: a changed policy version scored by the same model and catalogue;
  - a different model or catalogue (or framework version, preprocessing or scoring rules) is
    incompatible and is never reported as a dDI result.
dDI alone does not establish temporal policy drift or compliance deterioration.
CSV cells are written through csvcell.neutralise (formula-injection guard) and read back with
csvcell.restore, so snapshot round trips are exact. Standard library only.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .csvcell import neutralise, restore
from .errors import ConfigError, InputValidationError

PREPROCESSING_TAG = "verbatim-text; L2-normalised embeddings; cosine via dot product; v1"
SEVERITY = {"HIGH": 0, "REVIEW": 1, "ESCALATE": 2}
COMPAT_KEYS = ("framework_id", "framework_version", "catalogue_sha256", "model_weights_sha256", "preprocessing",
               "bands", "aps_weights")


class IncompatibleSnapshots(ConfigError):
    """The two snapshots differ in model, catalogue, framework, preprocessing or scoring rules."""


@dataclass(frozen=True)
class SnapshotRow:
    statement_id: str
    text_sha256: str
    top1_control_id: str
    top1_similarity: float
    tier_code: str
    drift_index: float
    aps: float
    top_controls: tuple[str, ...] = ()  # ranked top-k IDs when available


@dataclass(frozen=True)
class Snapshot:
    meta: Mapping[str, Any]
    rows: tuple[SnapshotRow, ...] = field(default_factory=tuple)

    def by_id(self) -> dict[str, SnapshotRow]:
        return {r.statement_id: r for r in self.rows}


def validate_snapshot(s: Snapshot) -> None:
    missing = [k for k in COMPAT_KEYS if k not in s.meta]
    if missing:
        raise InputValidationError(f"snapshot meta lacks {missing}")
    ids = [r.statement_id for r in s.rows]
    if any(not i or i != i.strip() for i in ids):
        raise InputValidationError("snapshot has blank or padded statement IDs")
    dup = sorted({i for i, c in Counter(ids).items() if c > 1})
    if dup:
        raise InputValidationError(f"snapshot has duplicate statement IDs {dup[:20]}")
    for r in s.rows:
        if r.tier_code not in SEVERITY:
            raise InputValidationError(f"snapshot row {r.statement_id}: unknown tier {r.tier_code!r}")
        for v in (r.top1_similarity, r.drift_index, r.aps):
            if not math.isfinite(v):
                raise InputValidationError(f"snapshot row {r.statement_id}: non-finite value")


def check_compatible(prev: Snapshot, cur: Snapshot) -> dict[str, Any]:
    validate_snapshot(prev)
    validate_snapshot(cur)
    diffs = {k: {"previous": prev.meta.get(k), "current": cur.meta.get(k)}
             for k in COMPAT_KEYS if prev.meta.get(k) != cur.meta.get(k)}
    if diffs:
        raise IncompatibleSnapshots(f"snapshots are not comparable; differing fields: {sorted(diffs)}")
    return {k: cur.meta[k] for k in COMPAT_KEYS}


def compare_snapshots(prev: Snapshot, cur: Snapshot, *, materiality: float = 1e-3) -> tuple[list[dict], dict]:
    """Row-level comparison and summary. Raises IncompatibleSnapshots when not comparable."""
    compat = check_compatible(prev, cur)
    if not (materiality > 0):
        raise ConfigError("materiality must be > 0")
    p, c = prev.by_id(), cur.by_id()
    fid = cur.meta["framework_id"]
    out: list[dict] = []
    for sid in sorted(set(p) | set(c)):
        a, b = p.get(sid), c.get(sid)
        if a is None or b is None:
            out.append({"statement_id": sid, "framework_id": fid,
                        "status": "only_current" if a is None else "only_previous"})
            continue
        d_di = b.drift_index - a.drift_index
        overlap = len(set(a.top_controls) & set(b.top_controls)) if a.top_controls and b.top_controls else None
        tier_change = "unchanged" if a.tier_code == b.tier_code else f"{a.tier_code}->{b.tier_code}"
        row = {
            "statement_id": sid, "framework_id": fid, "status": "matched",
            "text_changed": a.text_sha256 != b.text_sha256,
            "previous_top1": a.top1_control_id, "current_top1": b.top1_control_id,
            "top1_changed": a.top1_control_id != b.top1_control_id,
            "top3_overlap": overlap,
            "previous_similarity": a.top1_similarity, "current_similarity": b.top1_similarity,
            "delta_similarity": b.top1_similarity - a.top1_similarity,
            "previous_di": a.drift_index, "current_di": b.drift_index, "delta_di": d_di,
            "previous_aps": a.aps, "current_aps": b.aps, "delta_aps": b.aps - a.aps,
            "previous_tier": a.tier_code, "current_tier": b.tier_code, "tier_transition": tier_change,
            "tier_worsened": SEVERITY[b.tier_code] > SEVERITY[a.tier_code],
        }
        row["material_change"] = bool(row["text_changed"] or row["top1_changed"] or tier_change != "unchanged"
                                      or abs(d_di) >= materiality)
        out.append(row)
    m = [r for r in out if r["status"] == "matched"]
    dd = sorted(abs(r["delta_di"]) for r in m)
    summary = {
        "framework_id": fid, "compatibility": compat, "materiality_abs_delta_di": materiality,
        "previous_meta": dict(prev.meta), "current_meta": dict(cur.meta),
        "n_previous": len(p), "n_current": len(c), "n_matched": len(m),
        "only_previous": sum(r["status"] == "only_previous" for r in out),
        "only_current": sum(r["status"] == "only_current" for r in out),
        "text_changed": sum(r["text_changed"] for r in m),
        "top1_changed": sum(r["top1_changed"] for r in m),
        "tier_transitions": dict(Counter(r["tier_transition"] for r in m)),
        "material_changes": sum(r["material_change"] for r in m),
        "max_abs_delta_di": dd[-1] if dd else None,
        "mean_abs_delta_di": (sum(dd) / len(dd)) if dd else None,
        "top3_overlap_available": any(r["top3_overlap"] is not None for r in m),
    }
    return out, summary


def review_queue(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Material changes first: text changed, tier worsened, top-1 changed, then largest |dDI|.

    A matched row is material when its text hash, top-1 control or band changed, or |dDI| >= materiality.
    """
    m = [dict(r) for r in rows if r.get("status") == "matched" and r.get("material_change")]
    m.sort(key=lambda r: (not r["text_changed"], not r["tier_worsened"], not r["top1_changed"],
                          -abs(r["delta_di"]), r["statement_id"]))
    for i, r in enumerate(m, start=1):
        r["review_rank"] = i
        r["review_status"] = "CHANGE_PENDING_HUMAN_REVIEW"
    return m


# --------------------------------------------------------------------------- persistence

SNAP_COLS = ("statement_id", "text_sha256", "top1_control_id", "top1_similarity", "tier_code", "drift_index", "aps",
             "top_controls")


def write_snapshot(s: Snapshot, csv_path: Path, meta_path: Path) -> None:
    validate_snapshot(s)
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(SNAP_COLS)
    for r in s.rows:
        w.writerow([neutralise(v) for v in (r.statement_id, r.text_sha256, r.top1_control_id, repr(r.top1_similarity),
                                            r.tier_code, repr(r.drift_index), repr(r.aps), "|".join(r.top_controls))])
    with open(csv_path, "x", encoding="utf-8", newline="") as f:
        f.write(buf.getvalue())
    with open(meta_path, "x", encoding="utf-8") as f:
        json.dump(dict(s.meta), f, indent=2, ensure_ascii=True)
        f.write("\n")


def read_snapshot(csv_path: Path, meta_path: Path) -> Snapshot:
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    rows = []
    for raw in csv.DictReader(io.StringIO(Path(csv_path).read_text(encoding="utf-8"), newline="")):
        d = {k: restore(v) if isinstance(v, str) else v for k, v in raw.items()}
        rows.append(SnapshotRow(d["statement_id"], d["text_sha256"], d["top1_control_id"], float(d["top1_similarity"]),
                                d["tier_code"], float(d["drift_index"]), float(d["aps"]),
                                tuple(x for x in d["top_controls"].split("|") if x)))
    s = Snapshot(meta, tuple(rows))
    validate_snapshot(s)
    return s


def write_comparison_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    cols = ["statement_id", "framework_id", "status", "text_changed", "previous_top1", "current_top1", "top1_changed",
            "top3_overlap", "previous_similarity", "current_similarity", "delta_similarity", "previous_di",
            "current_di", "delta_di", "previous_aps", "current_aps", "delta_aps", "previous_tier", "current_tier",
            "tier_transition", "tier_worsened", "material_change"]
    extra = [k for k in ("review_rank", "review_status") if rows and k in rows[0]]
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=(["review_rank"] if "review_rank" in extra else []) + cols
                       + (["review_status"] if "review_status" in extra else []), lineterminator="\n",
                       extrasaction="ignore", restval="")
    w.writeheader()
    for r in rows:
        w.writerow({k: neutralise(v) for k, v in r.items()})
    with open(path, "x", encoding="utf-8", newline="") as f:
        f.write(buf.getvalue())


def snapshot_meta_dict(**kw: Any) -> dict[str, Any]:
    missing = [k for k in COMPAT_KEYS if k not in kw]
    if missing:
        raise ConfigError(f"snapshot meta needs {missing}")
    return dict(kw)


__all__ = ["Snapshot", "SnapshotRow", "IncompatibleSnapshots", "check_compatible", "compare_snapshots",
           "review_queue", "write_snapshot", "read_snapshot", "write_comparison_csv", "PREPROCESSING_TAG",
           "snapshot_meta_dict", "asdict"]
