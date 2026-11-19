"""Review bands, top-1 snapshot dissimilarity (DI), Audit Priority Score and the priority queue
(Blueprint rev 3, sections 3.1-3.3, 3.7).

Definitions (research values preserved):
- Review band ("heuristic similarity review band"), decided on the UNROUNDED float64 top-1
  similarity: HIGH if sim >= 0.59; REVIEW if 0.49 <= sim < 0.59; ESCALATE if sim < 0.49.
  Cosine similarity is not a probability and the bands are not probability-calibrated.
- DI = 1 - top-1 similarity, where top-1 is the control the model ranks first (DI_top1):
  the top-1 snapshot dissimilarity of one statement in one run. The function and field keep the
  legacy name `drift_index`; the value is not temporal drift, not evidence that a policy changed,
  not a control failure, not non-compliance and not a calibrated risk probability. DI_mapped
  (dissimilarity to an independently assigned control) is NOT computed by this package.
- APS = DI x tier weight (default 1.00 / 1.25 / 1.50). A heuristic audit-review priority score that
  depends on the top-1 similarity alone (through DI and the band). It is not a validated risk score,
  not a calibrated probability and does not measure control criticality. With weights satisfying
  HIGH <= REVIEW <= ESCALATE (the defaults), the queue order equals ascending top-1 similarity;
  configuration accepts any positive weights, and other orderings of the weights break that equivalence.

Rounding: DI and APS are kept unrounded (float64) for ranking. Separate legacy-compatible
four-decimal values reproduce the research notebook exactly:
  drift_index_r4 = numpy.round(1 - sim, 4); aps_r4 = numpy.round(drift_index_r4 * weight, 4)
(sequential rounding, numpy semantics; Python's built-in round() differs on half-way cases).

Every result is a candidate mapping pending human review, never a compliance determination.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .config import (
    APS_WEIGHTS,
    REVIEW_STATUS,
    SIMILARITY_RANGE_TOLERANCE,
    TIER_DISPLAY,
    TIER_HIGH_MIN,
    TIER_REVIEW_MIN,
    Settings,
    TierCode,
)
from .errors import ConfigError, GrcError
from .mapping import Candidate
from .text import TokenProfile


def _check_sim(sim: object) -> float:
    if isinstance(sim, bool) or not isinstance(sim, (int, float, np.floating, np.integer)):
        raise GrcError("similarity must be a real number")
    x = float(sim)
    if not math.isfinite(x):
        raise GrcError("similarity must be finite")
    if abs(x) > 1.0 + SIMILARITY_RANGE_TOLERANCE:
        raise GrcError("similarity outside [-1, 1]")
    return x


def tier_code(sim: float, high_min: float = TIER_HIGH_MIN, review_min: float = TIER_REVIEW_MIN) -> TierCode:
    """HIGH if sim >= high_min; REVIEW if review_min <= sim < high_min; ESCALATE otherwise."""
    x = _check_sim(sim)
    if not review_min < high_min:
        raise ConfigError("review_min must be lower than high_min")
    if x >= high_min:
        return "HIGH"
    if x >= review_min:
        return "REVIEW"
    return "ESCALATE"


def drift_index(sim: float) -> float:
    """DI_top1 = 1 - top-1 similarity (float64, unrounded). Snapshot dissimilarity; legacy name, not temporal drift."""
    return 1.0 - _check_sim(sim)


def aps_weight(tier: str, weights: Mapping[str, float] = APS_WEIGHTS) -> float:
    if tier not in weights:
        raise ConfigError(f"no APS weight for tier {tier!r}")
    return float(weights[tier])


def aps(di: float, weight: float) -> float:
    """APS = DI x tier weight (float64, unrounded). Heuristic audit-review priority score."""
    return float(di) * float(weight)


def round4(x: float) -> float:
    """Legacy-compatible 4-decimal rounding: numpy.round on float64."""
    return float(np.round(np.float64(x), 4))


@dataclass(frozen=True)
class ScoredMapping:
    statement_id: str
    candidates: tuple[Candidate, ...]
    top1_control_id: str
    top1_similarity: float
    tier_code: TierCode
    tier_label: str
    drift_index: float
    aps_weight: float
    aps: float
    drift_index_r4: float
    aps_r4: float
    token_profile: TokenProfile
    encoder_kind: str
    review_status: str = REVIEW_STATUS


def score_row(
    statement_id: str,
    candidates: tuple[Candidate, ...],
    profile: TokenProfile,
    encoder_kind: str,
    settings: Settings,
) -> ScoredMapping:
    if not candidates or candidates[0].rank != 1:
        raise GrcError("candidates must be non-empty and start at rank 1")
    top = candidates[0]
    sim = _check_sim(top.similarity)
    tier = tier_code(sim, settings.high_min, settings.review_min)
    w = aps_weight(tier, settings.aps_weights)
    di = drift_index(sim)
    di4 = round4(1.0 - sim)
    return ScoredMapping(
        statement_id=statement_id,
        candidates=candidates,
        top1_control_id=top.control_id,
        top1_similarity=sim,
        tier_code=tier,
        tier_label=TIER_DISPLAY[tier],
        drift_index=di,
        aps_weight=w,
        aps=aps(di, w),
        drift_index_r4=di4,
        aps_r4=round4(di4 * w),
        token_profile=profile,
        encoder_kind=encoder_kind,
        review_status=REVIEW_STATUS,
    )


def queue_key(row: ScoredMapping) -> tuple[float, float, str]:
    """Descending unrounded APS, then ascending top-1 similarity, then statement_id."""
    return (-row.aps, row.top1_similarity, row.statement_id)


def priority_queue(rows: Sequence[ScoredMapping]) -> list[ScoredMapping]:
    """Audit-priority queue (stable sort on queue_key). Ranks never depend on rounding."""
    ids = [r.statement_id for r in rows]
    if len(set(ids)) != len(ids):
        raise GrcError("duplicate statement IDs in queue input")
    return sorted(rows, key=queue_key)
