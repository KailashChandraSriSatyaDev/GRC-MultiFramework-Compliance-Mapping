"""Reading a single statement for `map-one` (Blueprint rev 3, section 3.8).

Input is decoded as strict UTF-8. A leading UTF-8 byte-order mark is removed; nothing else is
changed (no stripping, no newline conversion). Errors name the source, never the text.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import BinaryIO

from .errors import InputValidationError

MAX_INPUT_BYTES = 1 << 20  # 1 MiB: one statement, not a document
_BOM = b"\xef\xbb\xbf"


def decode_input(data: bytes, source: str) -> str:
    if len(data) > MAX_INPUT_BYTES:
        raise InputValidationError(f"{source}: input larger than {MAX_INPUT_BYTES} bytes")
    if data.startswith(_BOM):
        data = data[len(_BOM):]
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InputValidationError(f"{source}: input is not valid UTF-8 (byte offset {exc.start})") from None


def read_text_file(path: Path) -> str:
    p = Path(path)
    try:
        if p.stat().st_size > MAX_INPUT_BYTES:
            raise InputValidationError(f"{p.name}: input larger than {MAX_INPUT_BYTES} bytes")
        data = p.read_bytes()
    except FileNotFoundError:
        raise InputValidationError(f"text file not found: {p.name}") from None
    except IsADirectoryError:
        raise InputValidationError(f"text file is a directory: {p.name}") from None
    except OSError as exc:
        raise InputValidationError(f"cannot read text file {p.name} ({exc.__class__.__name__})") from None
    return decode_input(data, p.name)


def read_stdin(stream: BinaryIO | None = None) -> str:
    src = sys.stdin.buffer if stream is None else stream
    data = src.read(MAX_INPUT_BYTES + 1)
    return decode_input(data, "stdin")
