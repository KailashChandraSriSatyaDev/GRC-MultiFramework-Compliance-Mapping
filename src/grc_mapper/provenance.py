"""Run provenance records (Blueprint rev 3, section 3.8). Never contain policy or control text.

Recorded: run id, mode, package and library versions, encoder kind and verified manifest hash,
input file hashes (or the single-input fingerprint), redacted settings, counts, tier counts,
truncation counts and IDs, UTC timestamps, and the disclaimer.
"""

from __future__ import annotations

import platform
import secrets
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

from . import __version__
from .config import BAND_TERM, Settings
from .scoring import ScoredMapping

DISCLAIMER = (
    "Candidate mappings for human review. Not a compliance determination. "
    "DI (field drift_index) is the top-1 snapshot dissimilarity, 1 - similarity to the model's top-1 candidate "
    "in this run: not temporal drift, not evidence of policy change or non-compliance. "
    "APS = DI x tier weight is a heuristic audit-review priority score, not a validated or calibrated risk score; "
    "with the default weights its queue order equals ascending top-1 similarity. "
    "Review bands are heuristic similarity review bands; cosine similarity is not a probability."
)
DEMO_DISCLAIMER = (
    "SOFTWARE-PIPELINE DEMONSTRATION WITH A HASH-SEEDED FAKE ENCODER. "
    "Scores are not NLP inference and not research results."
)
_LIBS = ("numpy", "torch", "transformers", "sentence-transformers", "tokenizers", "safetensors")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + secrets.token_hex(4)


def library_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version(), "grc_mapper": __version__}
    for name in _LIBS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def tier_counts(rows: Sequence[ScoredMapping]) -> dict[str, int]:
    return {t: sum(r.tier_code == t for r in rows) for t in ("HIGH", "REVIEW", "ESCALATE")}


def build_run_record(
    *,
    run_id: str,
    mode: str,
    started_utc: str,
    settings: Settings,
    encoder_kind: str,
    model_fingerprint: str,
    model_files_verified: int | None,
    catalog_source_kind: str,
    controls_sha256: str,
    rows: Sequence[ScoredMapping],
    controls_truncated_ids: Sequence[str],
    inputs: Mapping[str, Any],
    include_text: bool = False,
    demo: bool = False,
) -> dict[str, Any]:
    truncated = [r.statement_id for r in rows if r.token_profile.truncated]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "mode": mode,
        "package_version": __version__,
        "encoder_kind": encoder_kind,
        "model_runtime_manifest_sha256": model_fingerprint if encoder_kind != "FAKE_SHA256_DEMO" else None,
        "model_files_verified": model_files_verified,
        "catalog_source_kind": catalog_source_kind,
        "controls_csv_sha256": controls_sha256,
        "inputs": dict(inputs),
        "settings": settings.redacted(),
        "band_term": BAND_TERM,
        "include_text": bool(include_text),
        "versions": library_versions(),
        "platform": sys.platform,
        "n_statements": len(rows),
        "tier_counts": tier_counts(rows),
        "n_truncated_statements": len(truncated),
        "truncated_statement_ids": truncated,
        "n_controls_truncated": len(controls_truncated_ids),
        "truncated_control_ids": list(controls_truncated_ids),
        "review_status": "CANDIDATE_PENDING_HUMAN_REVIEW",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "disclaimer": DISCLAIMER,
        "demo_disclaimer": DEMO_DISCLAIMER if demo else None,
    }
