"""Shared pytest fixtures for the test suite."""

import shutil
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Clean up __pycache__ directories before running tests.

    This prevents stale bytecode issues that can cause import errors.
    """
    root = Path(config.rootdir)
    for pycache in root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)
    for pyc in root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
