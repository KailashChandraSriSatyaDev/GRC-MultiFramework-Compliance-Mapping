"""Verified, offline loading of the production sentence encoder (Blueprint rev 3, sections 3.3-3.5, 4.2-4.3).

Load sequence, in order:
  1. parse the public runtime manifest and the local full manifest;
  2. verify the model directory (sizes, SHA-256, closed layout) with the standard library;
  3. require OK_FULL (no flag, setting or environment variable relaxes this);
  4. set the offline environment variables;
  5. inside the application-level socket guard: import sentence_transformers and load with
     local_files_only=True and trust_remote_code=False;
  6. post-load checks: embedding dimension 384, max_seq_length 256, unit-norm finite probe
     embeddings;
  7. post-load re-verification: fresh SHA-256 of every runtime and expected non-runtime file
     against the full local manifest (OK_FULL again), plus unchanged size and mtime of every file.

`sentence_transformers` is imported only inside `load_verified_model`, after step 3.
Logs carry IDs and counts only, never text.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .errors import ConfigError, GrcError, ModelLoadError, OfflineViolation
from .integrity import (
    ManifestReport,
    load_full_manifest,
    load_runtime_manifest,
    require_full_verification,
    reverify_after_load,
    verify_model_dir,
)
from .offline import enforce_offline_env, network_blocked
from .text import TokenProfile, validate_statement_text

log = logging.getLogger("grc_mapper.model")

EXPECTED_EMBED_DIM = 384
EXPECTED_MAX_SEQ = 256
NORM_TOLERANCE = 1e-3
ENCODER_KIND = "SBERT_PRODUCTION"
# Neutral probe strings used only for the post-load norm check (no research text).
_PROBES = ("access review procedure", "backup schedule for servers")
_DEVICE_RE = re.compile(r"^(cpu|cuda|cuda:\d+)$")


@runtime_checkable
class Encoder(Protocol):
    kind: str
    embedding_dim: int
    max_seq_length: int

    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray: ...

    def token_profile(self, text: str) -> TokenProfile: ...


def check_embeddings(emb: np.ndarray, n: int, dim: int) -> np.ndarray:
    """Validate an embedding matrix: shape (n, dim), finite, rows of unit norm. Returns float32."""
    arr = np.asarray(emb)
    if arr.shape != (n, dim):
        raise GrcError(f"embedding shape {arr.shape} != expected {(n, dim)}")
    arr = arr.astype(np.float32, copy=False)
    if not np.all(np.isfinite(arr)):
        raise GrcError("embeddings contain NaN or Inf")
    if n:
        norms = np.linalg.norm(arr.astype(np.float64), axis=1)
        worst = float(np.max(np.abs(norms - 1.0)))
        if worst > NORM_TOLERANCE:
            raise GrcError(f"embeddings are not unit-normalised (max |norm-1| = {worst:.2e})")
    return arr


class SbertEncoder:
    """Encoder backed by a verified SentenceTransformer. Create it with `load_verified_model`."""

    kind = ENCODER_KIND

    def __init__(self, st_model: Any, report: ManifestReport) -> None:
        self._st = st_model
        self.report = report
        self.embedding_dim = int(st_model.get_sentence_embedding_dimension())
        self.max_seq_length = int(st_model.max_seq_length)

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ConfigError("batch_size must be a positive integer")
        items = list(texts)
        if not items:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        emb = self._st.encode(
            items,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # the model's own Normalize module already L2-normalises
            show_progress_bar=False,
        )
        return check_embeddings(emb, len(items), self.embedding_dim)

    def token_profile(self, text: str) -> TokenProfile:
        """Token counts from the model's real tokenizer, including special tokens.

        n_tokens_full: tokenizer IDs with add_special_tokens=True and truncation=False.
        n_tokens_used: length of input_ids from model.tokenize, the call encode() uses,
                       so it reflects the model's actual truncation.
        The text is not modified.
        """
        tok = self._st.tokenizer
        try:
            full = tok(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]
        except TypeError:  # tokenizer without a `verbose` argument
            full = tok(text, add_special_tokens=True, truncation=False)["input_ids"]
        features = self._st.tokenize([text])
        used = int(features["input_ids"].shape[1])
        return TokenProfile(len(full), used, len(full) > used)


def profile_texts(
    encoder: Encoder,
    item_ids: Sequence[str],
    texts: Sequence[str],
    *,
    kind: str,
) -> tuple[TokenProfile, ...]:
    """Profile every item (controls or statements). Logs truncated items by ID and counts only."""
    if len(item_ids) != len(texts):
        raise ValueError("item_ids and texts differ in length")
    profiles = []
    n_trunc = 0
    for item_id, text in zip(item_ids, texts):
        prof = encoder.token_profile(text)
        if prof.truncated:
            n_trunc += 1
            log.warning(
                "%s %s truncated by the model: %d tokens -> %d used",
                kind, item_id, prof.n_tokens_full, prof.n_tokens_used,
            )
        profiles.append(prof)
    log.info("%s token profiles: %d items, %d truncated", kind, len(profiles), n_trunc)
    return tuple(profiles)


def _require_path(value: Path | None, name: str) -> Path:
    if value is None:
        raise ConfigError(f"{name} is required")
    return Path(value)


def load_verified_model(
    model_dir: Path,
    *,
    runtime_manifest_path: Path,
    full_manifest_path: Path,
    device: str = "cpu",
) -> SbertEncoder:
    """Verify (OK_FULL required) and load the production model offline.

    WP2-B form with explicit paths. The Settings-based wrapper `load_verified_model(settings)`
    from the blueprint is added with config.py and delegates to this function.
    """
    model_dir = _require_path(model_dir, "model_dir")
    runtime_manifest_path = _require_path(runtime_manifest_path, "runtime manifest path")
    full_manifest_path = _require_path(full_manifest_path, "full manifest path")
    if not isinstance(device, str) or not _DEVICE_RE.match(device):
        raise ConfigError(f"device must be 'cpu', 'cuda' or 'cuda:N', not {device!r}")

    runtime = load_runtime_manifest(runtime_manifest_path)
    full = load_full_manifest(full_manifest_path)
    report = verify_model_dir(model_dir, runtime, full)
    require_full_verification(report)
    log.info("model verification %s: %d runtime files verified", report.status, report.runtime_files_verified)

    enforce_offline_env()
    try:
        with network_blocked():
            from sentence_transformers import SentenceTransformer  # lazy: only after OK_FULL

            st = SentenceTransformer(
                str(model_dir),
                device=device,
                local_files_only=True,
                trust_remote_code=False,
            )
            dim = st.get_sentence_embedding_dimension()
            if dim != EXPECTED_EMBED_DIM:
                raise ModelLoadError(f"embedding dimension {dim} != {EXPECTED_EMBED_DIM}")
            if st.max_seq_length != EXPECTED_MAX_SEQ:
                raise ModelLoadError(f"max_seq_length {st.max_seq_length} != {EXPECTED_MAX_SEQ}")
            encoder = SbertEncoder(st, report)
            for i, probe in enumerate(_PROBES):
                validate_statement_text(probe, item_id=f"probe-{i}")
            try:
                encoder.encode(list(_PROBES), batch_size=2)
            except GrcError as exc:
                raise ModelLoadError(f"post-load probe failed: {exc}") from None
    except (OfflineViolation, ModelLoadError, ConfigError):
        raise
    except Exception as exc:  # library errors are wrapped; message carries the type only
        raise ModelLoadError(f"model load failed ({exc.__class__.__name__}: {exc})") from exc

    reverify_after_load(model_dir, runtime, full, report)
    log.info("model loaded offline on %s: dim=%d max_seq_length=%d", device, dim, st.max_seq_length)
    return encoder
