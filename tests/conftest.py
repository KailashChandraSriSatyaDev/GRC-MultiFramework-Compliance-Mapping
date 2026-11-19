"""Test configuration: markers by directory, shared fixtures.

Run from the repository root (pyproject.toml puts src/ on the path):
    python -m pytest -m unit
    python -m pytest -m integration     # needs GRC_CONFIG and the verified local model
    python -m pytest -m regression      # needs GRC_CONFIG and staged local regression fixtures
Set PYTHONDONTWRITEBYTECODE=1 so no __pycache__ appears inside the repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from _support import TestEncoder, make_fake_model  # noqa: E402


def pytest_collection_modifyitems(config, items):
    for item in items:
        parts = Path(str(item.fspath)).resolve().relative_to(TESTS_DIR).parts
        if parts and parts[0] in ("unit", "integration", "regression"):
            item.add_marker(getattr(pytest.mark, parts[0]))


@pytest.fixture
def enc():
    return TestEncoder()


@pytest.fixture
def settings():
    from grc_mapper.config import load_settings
    return load_settings(env={})


@pytest.fixture
def fake_model(tmp_path):
    return make_fake_model(tmp_path)
