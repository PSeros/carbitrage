"""The optional output adapters: Excel reporting and plots.

Every name in the package is resolved lazily, so that reaching for a plot does
not require the ``excel`` extra and vice versa.
"""

from __future__ import annotations

import pytest

import carbitrage.reporting as reporting_package

LAZY = ["monte_carlo_plot", "one_way_plot", "ranking_plot", "tornado_plot", "write_excel"]


def test_the_package_advertises_exactly_the_adapters() -> None:
    assert list(reporting_package.__all__) == LAZY
    assert dir(reporting_package) == LAZY


@pytest.mark.parametrize("name", LAZY)
def test_every_advertised_adapter_resolves_to_something_callable(name: str) -> None:
    assert callable(getattr(reporting_package, name))


def test_anything_else_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = reporting_package.read_excel  # type: ignore[attr-defined]
