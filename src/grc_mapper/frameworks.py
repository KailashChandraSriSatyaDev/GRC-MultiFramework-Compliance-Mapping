"""Framework catalogue specifications for multi-framework candidate mapping.

A framework spec file (JSON, kept outside the repository because it holds local paths) lists one
entry per framework. Each framework keeps its own identifier namespace, catalogue, review bands and
APS weights; scores from different frameworks are never combined into one ranking.

    {"schema_version": 1, "frameworks": [
      {"framework_id": "ISO27001-2022", "framework_name": "ISO/IEC 27001", "framework_version": "2022",
       "catalogue_csv": "<path>", "id_column": "Control_ID", "text_column": "Control_Text",
       "expected_controls": 93, "source": "...", "licence": "...",
       "bands": {"high_min": 0.59, "review_min": 0.49, "basis": "..."},
       "aps_weights": {"HIGH": 1.0, "REVIEW": 1.25, "ESCALATE": 1.5}}, ...]}

Standard library only (plus the package's validated loaders).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .config import APS_WEIGHTS, Settings, validate_bands, validate_weights
from .data import ControlCatalog, load_control_catalog
from .errors import ConfigError
from .integrity import sha256_file

FRAMEWORK_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,40}$")
REQUIRED = ("framework_id", "framework_name", "framework_version", "catalogue_csv", "id_column", "text_column",
            "source", "licence", "bands")
ALLOWED = set(REQUIRED) | {"expected_controls", "aps_weights", "title_column", "notes"}


@dataclass(frozen=True)
class FrameworkSpec:
    framework_id: str
    framework_name: str
    framework_version: str
    catalogue_csv: Path
    id_column: str
    text_column: str
    source: str
    licence: str
    high_min: float
    review_min: float
    band_basis: str
    aps_weights: Mapping[str, float]
    expected_controls: int | None = None
    notes: str | None = None

    def scoring_settings(self, base: Settings) -> Settings:
        """The base settings with this framework's bands and weights."""
        return dataclasses.replace(base, high_min=self.high_min, review_min=self.review_min,
                                   aps_weights=self.aps_weights)

    def public_info(self, catalogue: ControlCatalog | None = None) -> dict[str, Any]:
        """Metadata for provenance and exports (catalogue file name only, never its path or text)."""
        d = {
            "framework_id": self.framework_id, "framework_name": self.framework_name,
            "framework_version": self.framework_version, "catalogue_file": self.catalogue_csv.name,
            "id_column": self.id_column, "text_column": self.text_column, "source": self.source,
            "licence": self.licence, "bands": {"high_min": self.high_min, "review_min": self.review_min,
                                               "basis": self.band_basis},
            "aps_weights": dict(self.aps_weights), "expected_controls": self.expected_controls, "notes": self.notes,
        }
        if catalogue is not None:
            d["catalogue_sha256"] = catalogue.source_sha256
            d["n_controls"] = len(catalogue)
        return d


def _expand(p: object, where: str) -> Path:
    if not isinstance(p, str) or not p.strip():
        raise ConfigError(f"{where} must be a non-empty path string")
    return Path(os.path.expanduser(os.path.expandvars(p)))


def _text(d: Mapping[str, Any], key: str, where: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ConfigError(f"{where}.{key} must be a non-empty string")
    return v


def parse_framework_spec(doc: Mapping[str, Any]) -> list[FrameworkSpec]:
    if not isinstance(doc, Mapping) or doc.get("schema_version") != 1:
        raise ConfigError("framework spec: schema_version must be 1")
    items = doc.get("frameworks")
    if not isinstance(items, list) or not items:
        raise ConfigError("framework spec: 'frameworks' must be a non-empty list")
    out: list[FrameworkSpec] = []
    for i, d in enumerate(items):
        where = f"frameworks[{i}]"
        if not isinstance(d, Mapping):
            raise ConfigError(f"{where} must be an object")
        unknown = sorted(set(d) - ALLOWED)
        missing = [k for k in REQUIRED if k not in d]
        if unknown or missing:
            raise ConfigError(f"{where}: unknown keys {unknown}, missing keys {missing}")
        fid = _text(d, "framework_id", where)
        if not FRAMEWORK_ID_RE.match(fid):
            raise ConfigError(f"{where}.framework_id must match {FRAMEWORK_ID_RE.pattern}")
        bands = d["bands"]
        if not isinstance(bands, Mapping) or set(bands) != {"high_min", "review_min", "basis"}:
            raise ConfigError(f"{where}.bands must have exactly high_min, review_min, basis")
        hi, rv = validate_bands(bands["high_min"], bands["review_min"])
        basis = _text(bands, "basis", f"{where}.bands")
        weights = validate_weights(d["aps_weights"]) if "aps_weights" in d else APS_WEIGHTS
        exp = d.get("expected_controls")
        if exp is not None and (isinstance(exp, bool) or not isinstance(exp, int) or exp < 1):
            raise ConfigError(f"{where}.expected_controls must be a positive integer")
        out.append(FrameworkSpec(
            framework_id=fid, framework_name=_text(d, "framework_name", where),
            framework_version=_text(d, "framework_version", where),
            catalogue_csv=_expand(d["catalogue_csv"], f"{where}.catalogue_csv"),
            id_column=_text(d, "id_column", where), text_column=_text(d, "text_column", where),
            source=_text(d, "source", where), licence=_text(d, "licence", where),
            high_min=hi, review_min=rv, band_basis=basis, aps_weights=MappingProxyType(dict(weights)),
            expected_controls=exp, notes=d.get("notes") if isinstance(d.get("notes"), str) else None,
        ))
    ids = [f.framework_id for f in out]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"framework spec: duplicate framework_id values {sorted({x for x in ids if ids.count(x) > 1})}")
    return out


def load_framework_spec(path: Path) -> list[FrameworkSpec]:
    p = Path(path)
    try:
        doc = json.loads(p.read_bytes().decode("utf-8-sig"))
    except FileNotFoundError:
        raise ConfigError(f"framework spec not found: {p.name}") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"framework spec unreadable: {p.name} ({exc.__class__.__name__})") from None
    return parse_framework_spec(doc)


def load_framework_catalogue(spec: FrameworkSpec) -> ControlCatalog:
    """Validated catalogue (unique IDs, verbatim texts, natural-sort order) for one framework."""
    cat = load_control_catalog(spec.catalogue_csv, id_column=spec.id_column, text_column=spec.text_column,
                               expected_count=spec.expected_controls, source_kind="research")
    if sha256_file(spec.catalogue_csv) != cat.source_sha256:  # file changed between read and hash
        raise ConfigError(f"catalogue changed while loading: {spec.catalogue_csv.name}")
    return cat
