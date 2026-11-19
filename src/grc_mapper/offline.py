"""Offline controls for model loading.

Three measures are used together (Blueprint rev 3, section 4.3):

1. `enforce_offline_env()` sets the Hugging Face offline environment variables before
   `sentence_transformers` is imported.
2. The model is loaded with `local_files_only=True` and `trust_remote_code=False` (see model.py).
3. `network_blocked()` is an APPLICATION-LEVEL SOCKET GUARD. Inside this Python process it
   replaces `socket.socket.connect`, `socket.socket.connect_ex` and `socket.create_connection`
   with functions that raise `OfflineViolation`, and restores the originals on exit.

Scope of the guard: it does NOT cover child processes, native libraries that open sockets
without going through Python's `socket` module, or any OS-level traffic. It is not
OS-level network isolation. Because it patches the `socket` module process-wide, it also
affects other threads while it is active.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager

from .errors import OfflineViolation

OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}

GUARD_MESSAGE = (
    "network connection attempted while the application-level socket guard was active "
    "(model loading must be offline)"
)


def enforce_offline_env(environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Set the offline environment variables and return the values that were applied."""
    target = os.environ if environ is None else environ
    for key, value in OFFLINE_ENV.items():
        target[key] = value
    return dict(OFFLINE_ENV)


def _blocked(*_args: object, **_kwargs: object) -> None:
    raise OfflineViolation(GUARD_MESSAGE)


@contextmanager
def network_blocked() -> Iterator[None]:
    """Application-level socket guard; see the module docstring for its scope.

    Nested use is safe: each level saves what it found and restores it in `finally`.
    """
    saved_connect = socket.socket.connect
    saved_connect_ex = socket.socket.connect_ex
    saved_create_connection = socket.create_connection
    socket.socket.connect = _blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = _blocked  # type: ignore[method-assign]
    socket.create_connection = _blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = saved_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = saved_connect_ex  # type: ignore[method-assign]
        socket.create_connection = saved_create_connection  # type: ignore[assignment]
