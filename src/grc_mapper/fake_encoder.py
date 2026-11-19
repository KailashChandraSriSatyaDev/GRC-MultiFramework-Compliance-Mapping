"""FakeEncoder: a hash-seeded stand-in for the sentence encoder (Blueprint rev 3, section 3.6).

SOFTWARE-PIPELINE DEMONSTRATION ONLY. Each text maps to a pseudo-random unit vector seeded from
the first 8 bytes of SHA-256(text). Similarities between such vectors carry no semantic meaning,
so its scores are not NLP inference and not research results.

Python's built-in hash() is never used, so output is identical across processes and machines.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from .text import TokenProfile

FAKE_KIND = "FAKE_SHA256_DEMO"


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


class FakeEncoder:
    kind = FAKE_KIND

    def __init__(self, dim: int = 384, max_seq_length: int = 256) -> None:
        if dim < 2 or max_seq_length < 3:
            raise ValueError("dim must be >= 2 and max_seq_length >= 3")
        self.embedding_dim = int(dim)
        self.max_seq_length = int(max_seq_length)

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        out = np.empty((len(texts), self.embedding_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            v = np.random.default_rng(_seed(t)).standard_normal(self.embedding_dim)
            out[i] = (v / np.linalg.norm(v)).astype(np.float32)
        return out

    def token_profile(self, text: str) -> TokenProfile:
        n = len(text.split()) + 2  # +2 mimics the two special tokens
        used = min(n, self.max_seq_length)
        return TokenProfile(n, used, n > used)
