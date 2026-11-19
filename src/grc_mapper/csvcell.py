"""Spreadsheet-safe CSV cells (formula-injection guard). Standard library only.

A CSV cell that starts with `=`, `+`, `-`, `@`, tab, carriage return or line feed can be run as a
formula when the file is opened in a spreadsheet. `neutralise` prefixes such a cell with one `'`
unless the whole cell is a plain decimal number (so `-0.12345678` stays numeric). A cell that
already starts with `'` is prefixed too, which makes the transformation reversible: `restore`
removes exactly one leading `'`, and restore(neutralise(v)) == v for every string.

Only CSV output is affected. JSON exports keep the original values.
"""

from __future__ import annotations

import re

FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r", "\n")
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def neutralise(value: object) -> str:
    v = "" if value is None else str(value)
    if v and (v[0] in FORMULA_TRIGGERS or v[0] == "'") and not _NUMBER.fullmatch(v):
        return "'" + v
    return v


def restore(value: str) -> str:
    return value[1:] if value.startswith("'") else value


__all__ = ["FORMULA_TRIGGERS", "neutralise", "restore"]
