"""Multi-framework candidate mapping with shared statement embeddings.

Statements are validated, token-profiled and encoded ONCE; each framework then has its own control
index, top-k ranking, review bands, DI, APS and queue. The per-framework computation is the same as
pipeline.map_statements (similarity_matrix -> top_k -> score_row); a unit test asserts equality.
Framework identifiers are never merged and scores from different frameworks are never ranked together.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import Settings
from .data import Statement
from .errors import ConfigError, InputValidationError
from .frameworks import FrameworkSpec
from .mapping import encode_texts, similarity_matrix, top_k
from .model import profile_texts
from .pipeline import ControlIndex, _fingerprint
from .scoring import ScoredMapping, score_row
from .text import TokenProfile, validate_statement_text

log = logging.getLogger("grc_mapper.multiframework")


@dataclass(frozen=True)
class FrameworkIndex:
    spec: FrameworkSpec
    index: ControlIndex


@dataclass(frozen=True)
class EncodedStatements:
    statement_ids: tuple[str, ...]
    embeddings: np.ndarray
    token_profiles: tuple[TokenProfile, ...]
    encoder_kind: str
    model_fingerprint: str


def encode_statements(statements: Sequence[Statement], encoder, settings: Settings) -> EncodedStatements:
    items = list(statements)
    if not items:
        raise InputValidationError("no statements to map")
    ids = [s.statement_id for s in items]
    if len(set(ids)) != len(ids):
        raise InputValidationError("duplicate statement IDs in input")
    texts = [validate_statement_text(s.text, item_id=s.statement_id) for s in items]
    profiles = profile_texts(encoder, ids, texts, kind="statement")
    emb = encode_texts(encoder, texts, settings.batch_size)
    return EncodedStatements(tuple(ids), emb, profiles, str(encoder.kind), _fingerprint(encoder))


def map_framework(encoded: EncodedStatements, fw: FrameworkIndex, settings: Settings) -> list[ScoredMapping]:
    """One framework's candidate mappings for already-encoded statements."""
    idx = fw.index
    if encoded.encoder_kind != idx.encoder_kind or encoded.model_fingerprint != idx.model_fingerprint:
        raise ConfigError(f"statements and the {fw.spec.framework_id} index were built with different encoders")
    if encoded.embeddings.shape[1] != idx.embeddings.shape[1]:
        raise ConfigError(f"embedding dimension differs for {fw.spec.framework_id}")
    k = settings.top_k
    if not 1 <= k <= len(idx.catalog):
        raise ConfigError(f"top_k must be between 1 and {len(idx.catalog)} for {fw.spec.framework_id}")
    fw_settings = fw.spec.scoring_settings(settings)
    sims = similarity_matrix(encoded.embeddings, idx.embeddings)
    cids = idx.catalog.control_ids
    rows = [score_row(sid, top_k(sims[i], k, cids), encoded.token_profiles[i], idx.encoder_kind, fw_settings)
            for i, sid in enumerate(encoded.statement_ids)]
    log.info("%s: mapped %d statements against %d controls", fw.spec.framework_id, len(rows), len(cids))
    return rows


def map_multi(statements: Sequence[Statement], frameworks: Sequence[FrameworkIndex], encoder,
              settings: Settings) -> dict[str, list[ScoredMapping]]:
    """Separate candidate mappings for every framework, from one encoding pass."""
    ids = [f.spec.framework_id for f in frameworks]
    if not ids or len(set(ids)) != len(ids):
        raise ConfigError("frameworks must be non-empty with unique framework_id values")
    encoded = encode_statements(statements, encoder, settings)
    return {f.spec.framework_id: map_framework(encoded, f, settings) for f in frameworks}
