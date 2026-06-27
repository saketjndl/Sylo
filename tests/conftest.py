"""Shared test fixtures for the Sylo SDK test suite."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

import sylo
from sylo.config import reset_config
from sylo.storage.local_store import LocalStorage


@pytest.fixture(autouse=True)
def _clean_config():
    """Reset global config before and after each test."""
    reset_config()
    # Clear any env vars that might interfere
    env_vars = [
        "SYLO_API_KEY",
        "SYLO_PROJECT",
        "SYLO_ENVIRONMENT",
        "SYLO_STORAGE",
        "SYLO_REDIS_URL",
        "LURO_API_KEY",
        "LURO_PROJECT",
        "LURO_ENVIRONMENT",
        "LURO_STORAGE",
        "LURO_REDIS_URL",
    ]
    saved = {}
    for var in env_vars:
        if var in os.environ:
            saved[var] = os.environ.pop(var)
    yield
    reset_config()
    # Restore env vars
    for var, val in saved.items():
        os.environ[var] = val


@pytest.fixture
def tmp_storage_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for local storage tests."""
    storage_dir = tmp_path / "sylo_test_storage"
    storage_dir.mkdir()
    return storage_dir


@pytest.fixture
def local_storage(tmp_storage_dir: Path) -> LocalStorage:
    """Provide a LocalStorage instance backed by a temp directory."""
    return LocalStorage(root_dir=tmp_storage_dir)


@pytest.fixture
def init_sylo(tmp_storage_dir: Path):
    """Initialize Sylo with local storage in a temp directory.

    Returns a callable so tests can customize init params.
    """

    def _init(**kwargs):
        defaults = {
            "project": "test-project",
            "environment": "development",
            "storage": "local",
        }
        defaults.update(kwargs)
        sylo.init(**defaults)
        # Override the storage root to use temp dir
        from sylo.config import get_config
        return get_config()

    return _init


@pytest.fixture
def init_luro(init_sylo):
    """Backwards compatibility fixture alias."""
    return init_sylo
