"""Exception hierarchy for grc_mapper.

Every error carries the process exit code the CLI will use (Blueprint rev 3, section 3.8):
0 ok, 2 config, 3 integrity, 4 input, 5 model, 6 write failure,
7 verification incomplete (used by `verify-model` only; never raised as an exception).

Error messages name files, columns, IDs and counts. They never contain policy or control text.
"""

from __future__ import annotations


class GrcError(Exception):
    """Base class for all grc_mapper errors."""

    exit_code: int = 1


class ConfigError(GrcError):
    """Missing or invalid configuration (paths, settings, arguments)."""

    exit_code = 2


class IntegrityError(GrcError):
    """Model directory or manifest failed verification, or full verification is missing."""

    exit_code = 3


class InputValidationError(GrcError):
    """Input file, schema, identifier or text failed validation."""

    exit_code = 4


class ModelLoadError(GrcError):
    """The verified model could not be loaded or failed its post-load checks."""

    exit_code = 5


class OfflineViolation(ModelLoadError):
    """A network connection was attempted while the application-level socket guard was active."""

    exit_code = 5


class OutputWriteError(GrcError):
    """An output file could not be written."""

    exit_code = 6
