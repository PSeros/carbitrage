"""The optional output adapters: Excel reporting and plots.

Every name in the package is resolved lazily, so that reaching for a plot does
not require the ``excel`` extra and vice versa.
"""

from __future__ import annotations

import numpy as np
import pytest

import carbitrage.reporting as reporting_package
from carbitrage.errors import CarbitrageError
from carbitrage.sensitivity import MonteCarlo

LAZY = [
    "difference_plot",
    "npv_density_plot",
    "one_way_plot",
    "ranking_plot",
    "tornado_plot",
    "write_excel",
]


def test_the_package_advertises_exactly_the_adapters() -> None:
    assert list(reporting_package.__all__) == LAZY
    assert dir(reporting_package) == LAZY


@pytest.mark.parametrize("name", LAZY)
def test_every_advertised_adapter_resolves_to_something_callable(name: str) -> None:
    assert callable(getattr(reporting_package, name))


def test_anything_else_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = reporting_package.read_excel  # type: ignore[attr-defined]


# -------------------------------------------------------------- the plots


def simulation(*names: str) -> MonteCarlo:
    """A simulation built by hand, so the plots are tested without a full run."""
    rng = np.random.default_rng(0)
    return MonteCarlo(
        names=names,
        npv={name: rng.normal(-20_000.0 - 500.0 * i, 400.0, 128) for i, name in enumerate(names)},
        params=("price",),
        draws=rng.normal(20_000.0, 500.0, (128, 1)),
    )


def test_a_difference_needs_a_named_pair_once_there_are_more_than_two() -> None:
    from carbitrage.reporting import difference_plot

    with pytest.raises(CarbitrageError, match="no one difference to draw"):
        difference_plot(simulation("a", "b", "c"))


def test_a_difference_takes_the_only_pair_there_is() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    from carbitrage.reporting import difference_plot

    ax = difference_plot(simulation("a", "b"))
    assert "a" in ax.get_xlabel()
    assert "b" in ax.get_xlabel()
    plt.close(ax.figure)


def test_a_named_pair_is_drawn_out_of_a_wider_simulation() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    from carbitrage.reporting import difference_plot

    ax = difference_plot(simulation("a", "b", "c"), between=("a", "c"))
    assert "a" in ax.get_xlabel()
    assert "c" in ax.get_xlabel()
    plt.close(ax.figure)


def test_a_difference_is_drawn_between_exactly_two() -> None:
    from carbitrage.reporting import difference_plot

    with pytest.raises(CarbitrageError, match="exactly two alternatives"):
        difference_plot(simulation("a", "b", "c"), between=("a", "b", "c"))


def test_the_density_plot_draws_one_curve_per_alternative() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    from carbitrage.reporting import npv_density_plot

    ax = npv_density_plot(simulation("a", "b", "c"))
    labels = [line.get_label() for line in ax.get_lines()]
    assert [label for label in labels if not label.startswith("_")] == ["a", "b", "c"]
    plt.close(ax.figure)


def test_the_density_plot_says_the_overlap_is_not_a_probability() -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    from carbitrage.reporting import npv_density_plot

    ax = npv_density_plot(simulation("a", "b"))
    assert "not a probability" in ax.get_title()
    plt.close(ax.figure)


def test_the_density_plot_refuses_an_alternative_it_does_not_carry() -> None:
    from carbitrage.reporting import npv_density_plot

    with pytest.raises(CarbitrageError, match="not in this simulation"):
        npv_density_plot(simulation("a", "b"), names=("a", "z"))
