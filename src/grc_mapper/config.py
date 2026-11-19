"""Settings, research constants and configuration resolution (Blueprint rev 3, sections 3.1 and 4.1).

Resolution order (first hit wins): CLI overrides > environment variables > the TOML file named
by GRC_CONFIG (or the `config` CLI override) > built-in non-path defaults.
Paths have NO defaults, except the runtime manifest, which defaults to the copy shipped in this
repository when that file exists. A required path that is missing raises ConfigError when it is
requested through `Settings.require_path`.

Standard library only.
"""

from __future__ import annotations

import math
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from .errors import ConfigError

# ----------------------------------------------------------------------------- research constants
TIER_HIGH_MIN: float = 0.59       # HIGH     : sim >= 0.59
TIER_REVIEW_MIN: float = 0.49     # REVIEW   : 0.49 <= sim < 0.59 ; ESCALATE : sim < 0.49
TierCode = Literal["HIGH", "REVIEW", "ESCALATE"]
TIER_CODES: tuple[str, ...] = ("HIGH", "REVIEW", "ESCALATE")
APS_WEIGHTS: Mapping[str, float] = MappingProxyType({"HIGH": 1.00, "REVIEW": 1.25, "ESCALATE": 1.50})
TIER_DISPLAY: Mapping[str, str] = MappingProxyType({
    "HIGH": "High similarity - candidate mapping",
    "REVIEW": "Moderate similarity - human review",
    "ESCALATE": "Low similarity - escalate",
})
EXPECTED_EMBED_DIM = 384
EXPECTED_MAX_SEQ = 256
DEFAULT_TOP_K = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = "cpu"
REVIEW_STATUS = "CANDIDATE_PENDING_HUMAN_REVIEW"
BAND_TERM = "heuristic similarity review band"
SIMILARITY_RANGE_TOLERANCE = 1e-5

# Repository copy of the public runtime manifest (default for paths.model_manifest).
REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_RUNTIME_MANIFEST = REPO_ROOT / "manifests" / "model_runtime_manifest.json"

# ----------------------------------------------------------------------------- schema
PATH_KEYS: dict[str, str] = {  # TOML [paths] key -> environment variable
    "model_dir": "GRC_MODEL_DIR",
    "model_manifest": "GRC_MODEL_MANIFEST",
    "model_full_manifest": "GRC_MODEL_FULL_MANIFEST",
    "controls_csv": "GRC_CONTROLS_CSV",
    "policies_csv": "GRC_POLICIES_CSV",
    "regression_dir": "GRC_REGRESSION_DIR",
    "regression_expectations": "GRC_REGRESSION_EXPECTATIONS",
    "restricted_hashes": "GRC_RESTRICTED_HASHES",
    "output_dir": "GRC_OUTPUT_DIR",
}
COLUMN_DEFAULTS: dict[str, str | None] = {
    "control_id": "Control_ID",
    "control_text": "Control_Text",
    "statement_id": "Statement_ID",
    "statement_text": "Statement_Text",
    "label": None,
}
ALLOWED_TOML: dict[str, set[str]] = {
    "paths": set(PATH_KEYS),
    "runtime": {"device", "batch_size", "top_k"},
    "columns": set(COLUMN_DEFAULTS),
    "bands": {"high_min", "review_min"},
    "aps_weights": set(TIER_CODES),
    "scanner": {"public_root", "scan_output_dir", "identity_tokens"},
}
CLI_KEYS = set(PATH_KEYS) | {"config", "device", "batch_size", "top_k"}
_DEVICE_RE = re.compile(r"^(cpu|cuda|cuda:\d+)$")
_PLACEHOLDER_RE = re.compile(r"<[A-Z_]+>")


@dataclass(frozen=True)
class Settings:
    paths: Mapping[str, Path | None]
    columns: Mapping[str, str | None]
    device: str = DEFAULT_DEVICE
    batch_size: int = DEFAULT_BATCH_SIZE
    top_k: int = DEFAULT_TOP_K
    high_min: float = TIER_HIGH_MIN
    review_min: float = TIER_REVIEW_MIN
    aps_weights: Mapping[str, float] = field(default_factory=lambda: APS_WEIGHTS)
    scanner_public_root: Path | None = None
    scanner_output_dir: Path | None = None
    identity_tokens: tuple[str, ...] = ()
    config_file: Path | None = None

    @property
    def bands_are_default(self) -> bool:
        return self.high_min == TIER_HIGH_MIN and self.review_min == TIER_REVIEW_MIN

    @property
    def weights_are_default(self) -> bool:
        return dict(self.aps_weights) == dict(APS_WEIGHTS)

    def path(self, key: str) -> Path | None:
        if key not in PATH_KEYS:
            raise ConfigError(f"unknown path setting {key!r}")
        return self.paths.get(key)

    def require_path(self, key: str) -> Path:
        p = self.path(key)
        if p is None:
            raise ConfigError(f"required path not configured: paths.{key} (env {PATH_KEYS[key]})")
        return p

    def redacted(self) -> dict[str, Any]:
        """Settings for provenance: paths reduced to basenames, no identity tokens."""
        return {
            "paths": {k: (v.name if v is not None else None) for k, v in self.paths.items()},
            "columns": dict(self.columns),
            "device": self.device,
            "batch_size": self.batch_size,
            "top_k": self.top_k,
            "bands": {"high_min": self.high_min, "review_min": self.review_min, "default": self.bands_are_default},
            "aps_weights": dict(self.aps_weights),
            "aps_weights_default": self.weights_are_default,
            "config_file": self.config_file.name if self.config_file is not None else None,
        }


# ----------------------------------------------------------------------------- validators

def validate_bands(high_min: object, review_min: object) -> tuple[float, float]:
    vals = []
    for name, v in (("high_min", high_min), ("review_min", review_min)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ConfigError(f"bands.{name} must be a finite number")
        vals.append(float(v))
    hi, rv = vals
    if not (-1.0 <= rv < hi <= 1.0):
        raise ConfigError("bands must satisfy -1 <= review_min < high_min <= 1")
    return hi, rv


def validate_weights(weights: Mapping[str, object]) -> Mapping[str, float]:
    if set(weights) != set(TIER_CODES):
        raise ConfigError(f"aps_weights must have exactly the keys {list(TIER_CODES)}")
    out = {}
    for k in TIER_CODES:
        v = weights[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) <= 0:
            raise ConfigError(f"aps_weights.{k} must be a finite number > 0")
        out[k] = float(v)
    return MappingProxyType(out)


def _positive_int(v: object, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ConfigError(f"{name} must be a positive integer")
    return v


def _device(v: object) -> str:
    if not isinstance(v, str) or not _DEVICE_RE.match(v):
        raise ConfigError("device must be 'cpu', 'cuda' or 'cuda:N'")
    return v


def _to_path(v: object, name: str) -> Path:
    if not isinstance(v, (str, os.PathLike)):
        raise ConfigError(f"{name} must be a path string")
    s = os.fspath(v)
    if not s.strip():
        raise ConfigError(f"{name} is empty")
    if _PLACEHOLDER_RE.search(s):
        raise ConfigError(f"{name} still contains a template placeholder")
    return Path(os.path.expanduser(os.path.expandvars(s)))


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path.name}") from None
    except OSError as exc:
        raise ConfigError(f"config file cannot be read: {path.name} ({exc.__class__.__name__})") from None
    try:
        doc = tomllib.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"config file is not valid UTF-8 TOML: {path.name} ({exc.__class__.__name__})") from None
    for section, body in doc.items():
        if section not in ALLOWED_TOML:
            raise ConfigError(f"unknown config section [{section}] in {path.name}")
        if not isinstance(body, dict):
            raise ConfigError(f"config section [{section}] must be a table")
        unknown = sorted(set(body) - ALLOWED_TOML[section])
        if unknown:
            raise ConfigError(f"unknown keys in [{section}] of {path.name}: {unknown}")
    return doc


# ----------------------------------------------------------------------------- resolution

def load_settings(
    cli_overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Resolve settings: CLI > environment > TOML (GRC_CONFIG) > non-path defaults."""
    cli = {k: v for k, v in (cli_overrides or {}).items() if v is not None}
    unknown_cli = sorted(set(cli) - CLI_KEYS)
    if unknown_cli:
        raise ConfigError(f"unknown CLI overrides: {unknown_cli}")
    environ = os.environ if env is None else env

    cfg_src = cli.get("config", environ.get("GRC_CONFIG"))
    cfg_path = _to_path(cfg_src, "config file") if cfg_src else None
    doc = _read_toml(cfg_path) if cfg_path is not None else {}
    t_paths = doc.get("paths", {})
    t_run = doc.get("runtime", {})
    t_cols = doc.get("columns", {})
    t_bands = doc.get("bands", {})
    t_w = doc.get("aps_weights")
    t_scan = doc.get("scanner", {})

    paths: dict[str, Path | None] = {}
    for key, env_name in PATH_KEYS.items():
        raw = cli.get(key)
        if raw is None:
            raw = environ.get(env_name) or None
        if raw is None:
            raw = t_paths.get(key)
        paths[key] = _to_path(raw, f"paths.{key}") if raw is not None else None
    if paths["model_manifest"] is None and REPO_RUNTIME_MANIFEST.is_file():
        paths["model_manifest"] = REPO_RUNTIME_MANIFEST

    columns: dict[str, str | None] = {}
    for key, default in COLUMN_DEFAULTS.items():
        v = t_cols.get(key, default)
        if v is not None and (not isinstance(v, str) or not v.strip()):
            raise ConfigError(f"columns.{key} must be a non-empty string")
        columns[key] = v

    device = _device(cli.get("device", environ.get("GRC_DEVICE") or t_run.get("device", DEFAULT_DEVICE)))
    batch_size = _positive_int(cli.get("batch_size", t_run.get("batch_size", DEFAULT_BATCH_SIZE)), "batch_size")
    top_k = _positive_int(cli.get("top_k", t_run.get("top_k", DEFAULT_TOP_K)), "top_k")
    high_min, review_min = validate_bands(t_bands.get("high_min", TIER_HIGH_MIN), t_bands.get("review_min", TIER_REVIEW_MIN))
    weights = validate_weights(t_w) if t_w is not None else APS_WEIGHTS

    tokens = t_scan.get("identity_tokens", [])
    if not isinstance(tokens, list) or not all(isinstance(t, str) and t.strip() for t in tokens):
        raise ConfigError("scanner.identity_tokens must be a list of non-empty strings")
    pr = t_scan.get("public_root")
    so = t_scan.get("scan_output_dir")

    return Settings(
        paths=MappingProxyType(paths),
        columns=MappingProxyType(columns),
        device=device,
        batch_size=batch_size,
        top_k=top_k,
        high_min=high_min,
        review_min=review_min,
        aps_weights=weights,
        scanner_public_root=_to_path(pr, "scanner.public_root") if pr is not None else None,
        scanner_output_dir=_to_path(so, "scanner.scan_output_dir") if so is not None else None,
        identity_tokens=tuple(tokens),
        config_file=cfg_path,
    )
