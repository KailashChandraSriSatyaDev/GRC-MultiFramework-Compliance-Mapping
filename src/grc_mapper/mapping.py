"""Encoding, similarity and deterministic top-k ranking (Blueprint rev 3, sections 3.3 and 3.5).

NumPy and the standard library only; never imports the model library.

Determinism:
- Similarity is the dot product of L2-normalised float32 embeddings (equal to cosine),
  accumulated in float64 and stored as float32 (see similarity_matrix).
- Candidates are ordered by (-similarity, catalogue position). The catalogue is in natural-sort
  order of Control_ID (data.load_control_catalog), so exact float ties go to the control that
  sorts first, and CSV row order never affects ranking. Nothing is rounded before ranking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import SIMILARITY_RANGE_TOLERANCE
from .errors import ConfigError, GrcError

NORM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class Candidate:
    rank: int            # 1-based
    control_id: str
    similarity: float    # float64 copy of the float32 dot product


def validate_embeddings(emb: Any, *, n: int, dim: int) -> np.ndarray:
    """Shape (n, dim), finite, unit-norm rows (|norm - 1| <= 1e-3). Returns a float32 array."""
    arr = np.asarray(emb)
    if arr.ndim != 2 or arr.shape != (n, dim):
        raise GrcError(f"embedding shape {arr.shape} != expected {(n, dim)}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise GrcError(f"embeddings must be floating point, not {arr.dtype}")
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise GrcError("embeddings contain NaN or Inf")
    if n:
        worst = float(np.max(np.abs(np.linalg.norm(arr.astype(np.float64), axis=1) - 1.0)))
        if worst > NORM_TOLERANCE:
            raise GrcError(f"embeddings are not unit-normalised (max |norm-1| = {worst:.2e})")
    return arr


def encode_texts(encoder: Any, texts: Sequence[str], batch_size: int) -> np.ndarray:
    """Encode with the given encoder and validate the result against its declared dimension."""
    items = list(texts)
    emb = encoder.encode(items, batch_size=batch_size)
    return validate_embeddings(emb, n=len(items), dim=int(encoder.embedding_dim))


def similarity_matrix(stmt_emb: np.ndarray, ctrl_emb: np.ndarray) -> np.ndarray:
    """float32 matrix (n_statements, n_controls) of dot products of the float32 embeddings.

    The products are accumulated in float64 and then rounded once to float32. A pure float32
    matrix product can differ in the last bit between a 1-row and an n-row call (BLAS picks
    different kernels and summation orders); float64 accumulation makes each similarity
    independent of how many statements are mapped together, so batch and single-statement
    mapping agree exactly for identical embeddings.
    """
    s = np.asarray(stmt_emb, dtype=np.float32)
    c = np.asarray(ctrl_emb, dtype=np.float32)
    if s.ndim != 2 or c.ndim != 2 or s.shape[1] != c.shape[1]:
        raise GrcError(f"embedding dimensions do not match: {s.shape} vs {c.shape}")
    sims = (s.astype(np.float64) @ c.astype(np.float64).T).astype(np.float32)
    if not np.all(np.isfinite(sims)):
        raise GrcError("similarity matrix contains NaN or Inf")
    lim = 1.0 + SIMILARITY_RANGE_TOLERANCE
    if sims.size and (float(sims.max()) > lim or float(sims.min()) < -lim):
        raise GrcError("similarity outside [-1, 1] beyond tolerance; embeddings are not normalised")
    return sims


def ranking_order(sim_row: np.ndarray) -> np.ndarray:
    """Indices sorted by (-similarity, position). Exact ties resolve to the lower position."""
    row = np.asarray(sim_row)
    if row.ndim != 1:
        raise GrcError("similarity row must be one-dimensional")
    positions = np.arange(row.shape[0])
    # np.lexsort sorts by the LAST key first: primary -similarity, secondary position.
    return np.lexsort((positions, -row.astype(np.float64)))


def top_k(sim_row: np.ndarray, k: int, control_ids: Sequence[str]) -> tuple[Candidate, ...]:
    """The k best candidates for one statement, deterministic under ties."""
    row = np.asarray(sim_row)
    if row.ndim != 1 or row.shape[0] != len(control_ids):
        raise GrcError(f"similarity row length {row.shape} != number of controls {len(control_ids)}")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(control_ids):
        raise ConfigError(f"top_k must be an integer between 1 and {len(control_ids)}")
    if not np.all(np.isfinite(row)):
        raise GrcError("similarity row contains NaN or Inf")
    order = ranking_order(row)[:k]
    return tuple(
        Candidate(rank=r + 1, control_id=control_ids[int(i)], similarity=float(row[int(i)]))
        for r, i in enumerate(order)
    )
