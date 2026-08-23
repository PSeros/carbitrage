"""Parameter overrides, grids, switch points, tornados and Monte Carlo."""

from __future__ import annotations

import numpy as np
import pytest

from carbitrage import Alternative, Case, Timeline, Vehicle
from carbitrage.acquisition import Purchase
from carbitrage.context import Usage
from carbitrage.energy import Petrol
from carbitrage.errors import CarbitrageError
from carbitrage.params import Uncertain
from carbitrage.residual import GeometricDecline
from carbitrage.sensitivity import LogNormal, Normal, Range, Triangular, Uniform

# ------------------------------------------------------------------ params


PRICE = Uncertain(20_000.0, "sticker_price")


@pytest.fixture
def marked() -> Case:
    """Two alternatives, one of them holding a marked price."""

    def car(name: str, price: float) -> Vehicle:
        return Vehicle(
            name,
            price=price,
            energy=Petrol(consumption=7.0, price=1.80),
            residual=GeometricDecline(0.15),
        )

    return Case(
        alternatives=(
            Alternative(car("a", PRICE), Purchase(), label="a"),
            Alternative(car("b", 21_000.0), Purchase(), label="b"),
        ),
        timeline=Timeline(horizon_years=3, periods_per_year=12, rate=0.03),
        usage=Usage(annual_km=12_000.0),
    )


def test_a_study_takes_the_mark_itself_not_a_path(marked: Case) -> None:
    """The point of a mark: no string, and nothing to look up."""
    grid = marked.run().one_way(PRICE, [18_000.0, 20_000.0, 22_000.0])
    assert grid.param == "sticker_price"
    assert grid.winners == ("a", "a", "b")


def test_a_grid_is_headed_by_the_label(marked: Case) -> None:
    grid = marked.run().one_way(PRICE, [18_000.0, 22_000.0])
    assert grid.to_markdown().startswith("| sticker_price |")


def test_a_switch_point_reports_the_label(marked: Case) -> None:
    found = marked.run().switch_point_report(PRICE, ("a", "b"))
    assert found.param == "sticker_price"
    assert "sticker_price switches the answer at" in found.describe()


def test_a_tornado_bar_is_named_by_the_label(marked: Case) -> None:
    bars = marked.run().tornado({PRICE: Range(18_000.0, 22_000.0)}).bars
    assert [bar.param for bar in bars] == ["sticker_price"]


def test_a_marked_parameter_can_be_swept_on_both_axes(marked: Case) -> None:
    """Two-way sweeps chain overrides, so the mark has to survive the first."""
    grid = marked.run().two_way(PRICE, [18_000.0, 22_000.0], "annual_km", [5_000.0, 30_000.0])
    assert grid.row_param == "sticker_price"
    assert grid.values.shape == (2, 2)


# ------------------------------------------------------------------- grids


# ----------------------------------------------------------- switch points


# ---------------------------------------------------------------- tornado


def test_a_range_must_be_ordered() -> None:
    with pytest.raises(CarbitrageError, match="exceeds high"):
        Range(5.0, 1.0)


# ----------------------------------------------------------- distributions


def test_uniform_inverse_cdf_spans_its_range() -> None:
    u = np.array([0.0, 0.5, 1.0])
    assert Uniform(10.0, 20.0).ppf(u) == pytest.approx([10.0, 15.0, 20.0])


def test_normal_inverse_cdf_is_symmetric() -> None:
    normal = Normal(mu=5.0, sigma=2.0)
    assert normal.ppf(np.array([0.5]))[0] == pytest.approx(5.0, abs=1e-9)
    assert normal.ppf(np.array([0.975]))[0] == pytest.approx(5.0 + 2.0 * 1.959964, abs=1e-6)


def test_lognormal_from_mean_cv_reproduces_its_mean() -> None:
    dist = LogNormal.from_mean_cv(mean=0.99, cv=0.25)
    draws = dist.ppf(np.random.default_rng(0).uniform(1e-9, 1 - 1e-9, 200_000))
    assert float(np.mean(draws)) == pytest.approx(0.99, rel=0.02)
    assert dist.mean() == pytest.approx(0.99, rel=1e-9)
    assert np.all(draws > 0)


def test_triangular_respects_its_bounds_and_mode() -> None:
    dist = Triangular(low=0.8, mode=1.0, high=1.6)
    draws = dist.ppf(np.random.default_rng(1).uniform(0, 1, 50_000))
    assert draws.min() >= 0.8
    assert draws.max() <= 1.6
    assert float(np.mean(draws)) == pytest.approx(dist.mean(), rel=0.02)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: Normal(0.0, -1.0), "sigma must not be negative"),
        (lambda: Uniform(2.0, 1.0), "low must be below high"),
        (lambda: Triangular(1.0, 0.5, 2.0), "low <= mode <= high"),
        (lambda: LogNormal.from_mean_cv(0.0, 0.2), "mean must be positive"),
    ],
)
def test_distributions_validate_their_parameters(factory, match: str) -> None:
    with pytest.raises(CarbitrageError, match=match):
        factory()


# ---------------------------------------------------------- monte carlo


# ------------------------------------------------- the result-level surface
