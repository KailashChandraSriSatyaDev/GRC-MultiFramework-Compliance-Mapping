"""The single canonical mapping path (Blueprint rev 3, sections 2 and 3.3).

build_control_index -> map_statements -> (top_k, score_row) per statement.
`map_single` is exactly `map_statements([one statement])[0]`: batch and single-statement
mapping share one implementation.

All results are candidate mappings pending human review (review_status on every row).
Logs carry IDs and counts only, never policy or control text.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import Settings
from .data import ControlCatalog, Statement
from .errors import ConfigError, InputValidationError, IntegrityError
from .mapping import encode_texts, similarity_matrix, top_k
from .model import ENCODER_KIND as SBERT_KIND
from .model import load_verified_model, profile_texts
from .scoring import ScoredMapping, score_row
from .text import TokenProfile, validate_statement_text

log = logging.getLogger("grc_mapper.pipeline")
STATEMENT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@dataclass(frozen=True)
class ControlIndex:
    catalog: ControlCatalog
    embeddings: np.ndarray                  # float32 (n_controls, dim), L2-normalised
    token_profiles: tuple[TokenProfile, ...]
    encoder_kind: str
    model_fingerprint: str                  # sha256 of the verified runtime manifest, or the fake encoder kind

    @property
    def truncated_control_ids(self) -> tuple[str, ...]:
        return tuple(c for c, p in zip(self.catalog.control_ids, self.token_profiles) if p.truncated)


def load_encoder(settings: Settings):
    """Settings-based model load: OK_FULL verification, then offline load (model.load_verified_model)."""
    return load_verified_model(
        settings.require_path("model_dir"),
        runtime_manifest_path=settings.require_path("model_manifest"),
        full_manifest_path=settings.require_path("model_full_manifest"),
        device=settings.device,
    )


def _fingerprint(encoder) -> str:
    if encoder.kind == SBERT_KIND:
        report = getattr(encoder, "report", None)
        if report is None or not report.ok_full:
            raise IntegrityError("full model manifest required for production-model inference")
        return report.runtime_manifest_sha256
    return str(encoder.kind)


def build_control_index(catalog: ControlCatalog, encoder, settings: Settings) -> ControlIndex:
    """Profile and encode every control description once per run."""
    fingerprint = _fingerprint(encoder)
    if len(catalog) == 0:
        raise InputValidationError("control catalogue is empty")
    for cid, text in zip(catalog.control_ids, catalog.texts):
        validate_statement_text(text, item_id=cid)
    profiles = profile_texts(encoder, catalog.control_ids, catalog.texts, kind="control")
    emb = encode_texts(encoder, catalog.texts, settings.batch_size)
    log.info("control index built: %d controls, %d truncated, encoder %s",
             len(catalog), sum(p.truncated for p in profiles), encoder.kind)
    return ControlIndex(catalog, emb, profiles, str(encoder.kind), fingerprint)


def map_statements(
    statements: Sequence[Statement],
    index: ControlIndex,
    encoder,
    settings: Settings,
) -> list[ScoredMapping]:
    """Map statements to ranked candidate controls and score them. Output keeps input order."""
    if encoder.kind != index.encoder_kind or int(encoder.embedding_dim) != index.embeddings.shape[1]:
        raise ConfigError("encoder does not match the one used to build the control index")
    if _fingerprint(encoder) != index.model_fingerprint:
        raise ConfigError("encoder does not match the one used to build the control index")
    k = settings.top_k
    if not 1 <= k <= len(index.catalog):
        raise ConfigError(f"top_k must be between 1 and {len(index.catalog)}")
    items = list(statements)
    if not items:
        return []
    ids = [s.statement_id for s in items]
    if len(set(ids)) != len(ids):
        raise InputValidationError("duplicate statement IDs in input")
    texts = [validate_statement_text(s.text, item_id=s.statement_id) for s in items]

    profiles = profile_texts(encoder, ids, texts, kind="statement")
    emb = encode_texts(encoder, texts, settings.batch_size)
    sims = similarity_matrix(emb, index.embeddings)
    control_ids = index.catalog.control_ids
    rows = [
        score_row(sid, top_k(sims[i], k, control_ids), profiles[i], index.encoder_kind, settings)
        for i, sid in enumerate(ids)
    ]
    counts = {t: sum(r.tier_code == t for r in rows) for t in ("HIGH", "REVIEW", "ESCALATE")}
    log.info("mapped %d statements: %s; %d truncated", len(rows), counts, sum(p.truncated for p in profiles))
    return rows


def map_single(
    text: str,
    index: ControlIndex,
    encoder,
    settings: Settings,
    statement_id: str = "ADHOC-0001",
) -> ScoredMapping:
    """Map one statement through exactly the batch path."""
    if not isinstance(statement_id, str) or not STATEMENT_ID_RE.match(statement_id):
        raise InputValidationError("statement_id must match ^[A-Za-z0-9._-]{1,64}$")
    return map_statements([Statement(statement_id, text)], index, encoder, settings)[0]
