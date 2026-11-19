"""Shared test helpers (no test functions)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from grc_mapper.text import TokenProfile

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


def child_env(**extra: str) -> dict[str, str]:
    """Environment for subprocesses running repository code (no bytecode written)."""
    env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    env.update(extra)
    return env


class TestEncoder:
    """Isolated deterministic test encoder (SHA-256-seeded unit vectors; whitespace token surrogate)."""

    __test__ = False
    kind = "TEST_SHA256"

    def __init__(self, dim: int = 16, max_seq_length: int = 12) -> None:
        self.embedding_dim = dim
        self.max_seq_length = max_seq_length
        self.calls: list[tuple[int, int]] = []

    def encode(self, texts, *, batch_size):
        self.calls.append((len(texts), batch_size))
        out = np.empty((len(texts), self.embedding_dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:8], "little")
            v = np.random.default_rng(seed).standard_normal(self.embedding_dim)
            out[i] = (v / np.linalg.norm(v)).astype(np.float32)
        return out

    def token_profile(self, text):
        n = len(text.split()) + 2
        used = min(n, self.max_seq_length)
        return TokenProfile(n, used, n > used)


RUNTIME = {
    "1_Pooling/config.json": b'{"pooling_mode_mean_tokens": true}',
    "config.json": b'{"model_type": "bert"}',
    "config_sentence_transformers.json": b"{}",
    "model.safetensors": b"\x00\x01fake-weights\x02",
    "modules.json": b"[]",
    "sentence_bert_config.json": b'{"max_seq_length": 256}',
    "tokenizer.json": b"{}",
    "tokenizer_config.json": b"{}",
}
NON_RUNTIME = {
    "README.md": b"# synthetic readme\n",
    "eval/similarity_evaluation_train_dev_results.csv": b"epoch,steps\n",
}


def _entries(d):
    return [{"path": p, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()} for p, b in sorted(d.items())]


def make_fake_model(base: Path) -> tuple[Path, Path, Path]:
    """Synthetic model directory with matching runtime and full manifests."""
    model = base / "model"
    for rel, data in {**RUNTIME, **NON_RUNTIME}.items():
        p = model / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (model / "2_Normalize").mkdir()
    rt = base / "runtime.json"
    rt.write_text(json.dumps({"schema_version": 1, "model_name": "m", "description": "d", "files": _entries(RUNTIME)}))
    full = base / "full.json"
    full.write_text(json.dumps({
        "schema_version": 1, "model_name": "m", "local_only": True,
        "runtime_files": _entries(RUNTIME), "non_runtime_files": _entries(NON_RUNTIME),
    }))
    return model, rt, full
