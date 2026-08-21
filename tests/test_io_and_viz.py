"""The optional output adapters: Excel reporting and plots."""

from __future__ import annotations

import pytest


def test_the_io_package_exposes_only_the_writer() -> None:
    import carbitrage.io as io_package

    assert callable(io_package.write_excel)
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = io_package.read_excel  # type: ignore[attr-defined]
