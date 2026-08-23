"""Shared plumbing for the suite.

``tests/unit/`` holds one file per scope, covering that scope's contract.
``tests/integration/**`` builds whole alternatives and runs the engine.
``tests/docs/`` executes the code blocks in the README and ``docs/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
DOCS_DIR = REPO_ROOT / "docs"


@pytest.fixture
def openpyxl() -> object:
    return pytest.importorskip("openpyxl")


@pytest.fixture
def agg_matplotlib() -> object:
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    return mpl


@pytest.fixture
def pandas() -> object:
    return pytest.importorskip("pandas")
