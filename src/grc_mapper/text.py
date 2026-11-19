"""Text validation and the TokenProfile contract (Blueprint rev 3, sections 3.2, 3.4, 3.5).

Validation never changes the text: the object passed in is returned unchanged, so the model
receives exactly the stored string (verbatim parity with the research notebooks).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .errors import InputValidationError

# A Unicode letter: a word character that is neither a digit nor an underscore.
_LETTER = re.compile(r"[^\W\d_]")


def validate_statement_text(text: object, *, item_id: str) -> str:
    """Return `text` itself if it is valid model input; raise InputValidationError otherwise.

    Rules: must be a `str`; must not be empty or whitespace-only; must contain at least one
    letter. The error names the item ID only, never the text.
    """
    if not isinstance(text, str):
        raise InputValidationError(f"text for {item_id!r} is not a string ({type(text).__name__})")
    if not text.strip():
        raise InputValidationError(f"text for {item_id!r} is empty or whitespace-only")
    if not _LETTER.search(text):
        raise InputValidationError(f"text for {item_id!r} contains no letters")
    return text


@dataclass(frozen=True)
class TokenProfile:
    """Token counts for one input.

    n_tokens_full: tokenizer IDs with special tokens and no truncation.
    n_tokens_used: length of the IDs the model actually receives (model.tokenize).
    truncated:     n_tokens_full > n_tokens_used.
    """

    n_tokens_full: int
    n_tokens_used: int
    truncated: bool

    def __post_init__(self) -> None:
        for name in ("n_tokens_full", "n_tokens_used"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"TokenProfile.{name} must be a non-negative int")
        if self.n_tokens_used > self.n_tokens_full:
            raise ValueError("TokenProfile.n_tokens_used cannot exceed n_tokens_full")
        if self.truncated != (self.n_tokens_full > self.n_tokens_used):
            raise ValueError("TokenProfile.truncated must equal n_tokens_full > n_tokens_used")


def text_fingerprint(text: str) -> tuple[str, int]:
    """(sha256 of the UTF-8 bytes, character count). For provenance; contains no text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), len(text)
