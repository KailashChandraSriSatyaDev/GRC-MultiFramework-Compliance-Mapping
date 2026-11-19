"""CSV loading with schema and content validation (Blueprint rev 3, section 3.4).

Standard library only. Errors name the file (basename), columns, row numbers and IDs, never text.
Control catalogues are returned in natural-sort order of Control_ID (A.5.2 < A.5.10), so file
row order never affects ranking. Statements keep file order.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .errors import ConfigError, InputValidationError
from .integrity import sha256_bytes
from .text import validate_statement_text

SourceKind = Literal["research", "synthetic"]
_MAX_LISTED = 20


@dataclass(frozen=True)
class ControlCatalog:
    control_ids: tuple[str, ...]
    texts: tuple[str, ...]
    source_sha256: str
    source_kind: SourceKind

    def __post_init__(self) -> None:
        if len(self.control_ids) != len(self.texts):
            raise ValueError("ControlCatalog ids and texts differ in length")

    def __len__(self) -> int:
        return len(self.control_ids)


@dataclass(frozen=True)
class Statement:
    statement_id: str
    text: str

    def __repr__(self) -> str:  # never expose text through repr (logs, tracebacks)
        return f"Statement(statement_id={self.statement_id!r}, text=<{len(self.text)} chars>)"


def natural_key(identifier: str) -> tuple:
    """Natural sort key: digit runs compare numerically ('A.5.2' < 'A.5.10')."""
    parts = re.split(r"(\d+)", identifier)
    key = tuple((0, int(p), "") if p.isdigit() else (1, 0, p.casefold()) for p in parts if p != "")
    return key + ((2, 0, identifier),)


def _list(items: Sequence[str]) -> str:
    shown = list(items[:_MAX_LISTED])
    more = len(items) - len(shown)
    return repr(shown) + (f" (+{more} more)" if more > 0 else "")


def read_csv(path: Path, required_columns: Sequence[str]) -> tuple[list[dict[str, str]], str]:
    """Read a UTF-8 CSV (BOM tolerated). Returns (rows, sha256 of the file bytes)."""
    p = Path(path)
    try:
        data = p.read_bytes()
    except FileNotFoundError:
        raise InputValidationError(f"file not found: {p.name}") from None
    except OSError as exc:
        raise InputValidationError(f"cannot read {p.name} ({exc.__class__.__name__})") from None
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputValidationError(f"{p.name} is not valid UTF-8 (byte offset {exc.start})") from None

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration:
        raise InputValidationError(f"{p.name} is empty (no header row)") from None
    except csv.Error as exc:
        raise InputValidationError(f"{p.name}: CSV parse error in header ({exc})") from None
    if not header or any(not h.strip() for h in header):
        raise InputValidationError(f"{p.name}: header has blank column names")
    dupes = sorted({h for h in header if header.count(h) > 1})
    if dupes:
        raise InputValidationError(f"{p.name}: duplicate column names {dupes}")
    missing = [c for c in required_columns if c not in header]
    if missing:
        raise InputValidationError(f"{p.name}: missing required columns {missing}")

    rows: list[dict[str, str]] = []
    bad_width: list[str] = []
    try:
        for line_no, values in enumerate(reader, start=2):
            if not values:  # csv yields [] for a completely empty line
                continue
            if len(values) != len(header):
                bad_width.append(str(line_no))
                continue
            rows.append(dict(zip(header, values)))
    except csv.Error as exc:
        raise InputValidationError(f"{p.name}: CSV parse error near line {reader.line_num} ({exc})") from None
    if bad_width:
        raise InputValidationError(f"{p.name}: rows with the wrong number of fields at lines {_list(bad_width)}")
    return rows, sha256_bytes(data)


def _validate_ids(rows: list[dict[str, str]], column: str, fname: str, what: str) -> list[str]:
    ids: list[str] = []
    blank: list[str] = []
    padded: list[str] = []
    for i, row in enumerate(rows, start=2):
        v = row[column]
        if not v.strip():
            blank.append(str(i))
        elif v != v.strip():
            padded.append(v.strip())
        ids.append(v)
    if blank:
        raise InputValidationError(f"{fname}: blank {what} at lines {_list(blank)}")
    if padded:
        raise InputValidationError(f"{fname}: {what} with leading/trailing whitespace {_list(padded)}")
    counts: dict[str, int] = {}
    for x in ids:
        counts[x] = counts.get(x, 0) + 1
    dup = sorted((x for x, n in counts.items() if n > 1), key=natural_key)
    if dup:
        raise InputValidationError(f"{fname}: duplicate {what} {_list(dup)}")
    return ids


def _check_count(n: int, expected: int | None, fname: str, what: str) -> None:
    if expected is not None and n != expected:
        raise InputValidationError(f"{fname}: expected {expected} {what}, found {n}")


def load_control_catalog(
    path: Path,
    *,
    id_column: str = "Control_ID",
    text_column: str = "Control_Text",
    expected_count: int | None = None,
    source_kind: SourceKind = "research",
) -> ControlCatalog:
    """Load and validate a control catalogue. Texts are kept verbatim."""
    if source_kind not in ("research", "synthetic"):
        raise ConfigError(f"source_kind must be 'research' or 'synthetic', not {source_kind!r}")
    fname = Path(path).name
    rows, digest = read_csv(path, [id_column, text_column])
    if not rows:
        raise InputValidationError(f"{fname}: no control rows")
    ids = _validate_ids(rows, id_column, fname, "control IDs")
    texts = [validate_statement_text(r[text_column], item_id=cid) for r, cid in zip(rows, ids)]
    _check_count(len(ids), expected_count, fname, "controls")
    order = sorted(range(len(ids)), key=lambda k: natural_key(ids[k]))
    return ControlCatalog(
        control_ids=tuple(ids[k] for k in order),
        texts=tuple(texts[k] for k in order),
        source_sha256=digest,
        source_kind=source_kind,
    )


def load_statements(
    path: Path,
    *,
    id_column: str,
    text_column: str,
    label_column: str | None = None,
    known_controls: ControlCatalog | None = None,
    expected_count: int | None = None,
) -> tuple[list[Statement], dict[str, str] | None]:
    """Load and validate statements (file order kept).

    Returns (statements, labels or None). Labels are validated only when `label_column` is
    given, and then `known_controls` is required. The file hash for provenance is taken
    separately with integrity.sha256_file.
    """
    if label_column is not None and known_controls is None:
        raise ConfigError("known_controls is required when label_column is given")
    fname = Path(path).name
    required = [id_column, text_column] + ([label_column] if label_column else [])
    rows, _digest = read_csv(path, required)
    if not rows:
        raise InputValidationError(f"{fname}: no statement rows")
    ids = _validate_ids(rows, id_column, fname, "statement IDs")
    statements = [
        Statement(sid, validate_statement_text(r[text_column], item_id=sid)) for r, sid in zip(rows, ids)
    ]
    _check_count(len(statements), expected_count, fname, "statements")

    labels: dict[str, str] | None = None
    if label_column is not None:
        assert known_controls is not None
        known = set(known_controls.control_ids)
        labels = {}
        blank: list[str] = []
        unknown: list[str] = []
        for r, sid in zip(rows, ids):
            lab = r[label_column]
            if not lab.strip() or lab != lab.strip():
                blank.append(sid)
            elif lab not in known:
                unknown.append(sid)
            labels[sid] = lab
        if blank:
            raise InputValidationError(f"{fname}: blank or padded labels for statements {_list(blank)}")
        if unknown:
            raise InputValidationError(f"{fname}: labels not in the control catalogue for statements {_list(unknown)}")
    return statements, labels
