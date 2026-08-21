"""The optional output adapters: Excel reporting and plots."""

from __future__ import annotations

import pytest


def test_the_reporting_package_exposes_only_the_writer() -> None:
    import carbitrage.reporting as reporting_package

    assert callable(reporting_package.write_excel)
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = reporting_package.read_excel  # type: ignore[attr-defined]
