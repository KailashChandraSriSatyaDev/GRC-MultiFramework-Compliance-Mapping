"""Logging to stderr with a redaction filter (Blueprint rev 3, sections 2 and 3.8).

grc_mapper code logs IDs and counts only. As a second line of defence the filter replaces any
`Statement` argument, and any string argument longer than 120 characters, with a placeholder.
No log file is written.
"""

from __future__ import annotations

import logging
import sys

from .data import Statement

MAX_ARG_CHARS = 120
_HANDLER_NAME = "grc_mapper_stderr"


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple):
            record.args = tuple(self._clean(a) for a in args)
        elif isinstance(args, dict):
            record.args = {k: self._clean(v) for k, v in args.items()}
        if isinstance(record.msg, str) and len(record.msg) > 400:
            record.msg = f"<log message redacted: {len(record.msg)} chars>"
            record.args = ()
        return True

    @staticmethod
    def _clean(a: object) -> object:
        if isinstance(a, Statement):
            return f"<statement {a.statement_id}>"
        if isinstance(a, str) and len(a) > MAX_ARG_CHARS:
            return f"<redacted {len(a)} chars>"
        return a


def configure_logging(verbosity: int = 0) -> logging.Logger:
    """Attach one stderr handler to the `grc_mapper` logger. Idempotent."""
    logger = logging.getLogger("grc_mapper")
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logger.setLevel(level)
    for h in logger.handlers:
        if h.get_name() == _HANDLER_NAME:
            h.setLevel(level)
            return logger
    h = logging.StreamHandler(sys.stderr)
    h.set_name(_HANDLER_NAME)
    h.setLevel(level)
    h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    h.addFilter(RedactingFilter())
    logger.addHandler(h)
    for noisy in ("sentence_transformers", "transformers", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    return logger
