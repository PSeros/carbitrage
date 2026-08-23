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
from carbitrage.sensitivity.spec import _band

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


BANDED = Uncertain(20_000.0, "banded_price", Range(19_000.0, 21_500.0))


def case_with(price: Uncertain) -> Case:
    """The two-alternative comparison, with ``price`` on the first one."""

    def car(name: str, sticker: float) -> Vehicle:
        return Vehicle(
            name,
            price=sticker,
            energy=Petrol(consumption=7.0, price=1.80),
            residual=GeometricDecline(0.15),
        )

    return Case(
        alternatives=(
            Alternative(car("a", price), Purchase(), label="a"),
            Alternative(car("b", 21_000.0), Purchase(), label="b"),
        ),
        timeline=Timeline(horizon_years=3, periods_per_year=12, rate=0.03),
        usage=Usage(annual_km=12_000.0),
    )


def test_a_switch_point_says_where_it_falls_against_the_declared_band() -> None:
    found = case_with(BANDED).run().switch_point(BANDED, ("a", "b"))
    assert found is not None
    assert found.band == (19_000.0, 21_500.0)
    assert found.is_plausible is True
    assert "inside the 19,000.00 to 21,500.00 you called plausible" in found.describe()


def test_a_crossing_outside_the_band_is_still_found_and_reported() -> None:
    """The two questions stay separable: it can flip, and it flips out of reach."""
    tight = Uncertain(20_000.0, "banded_price", Range(19_500.0, 20_100.0))
    found = case_with(tight).run().switch_point("banded_price", ("a", "b"))
    assert found is not None
    assert found.value == pytest.approx(21_000.0, abs=1.0)  # found anyway, outside the band
    assert found.is_plausible is False
    assert "possible but not plausible" in found.describe()


def test_an_undeclared_parameter_keeps_the_plain_sentence(marked: Case) -> None:
    found = marked.run().switch_point(PRICE, ("a", "b"))
    assert found is not None
    assert found.band is None
    assert found.is_plausible is None
    assert found.describe().endswith("wins.")


def test_a_bounded_spread_is_plausible_all_the_way_to_its_edges() -> None:
    assert _band(Range(0.85, 1.35), base=0.99) == (0.85, 1.35)
    assert _band(Triangular(800.0, 1_700.0, 4_000.0), base=1_700.0) == (800.0, 4_000.0)


def test_an_unbounded_spread_is_read_at_its_deciles() -> None:
    """A normal's support is the whole line, so the support cannot be the band."""
    low, high = _band(Normal(mu=1_700.0, sigma=300.0), base=1_700.0)
    decile = 1.2815515655446004  # z at the 90th percentile
    assert low == pytest.approx(1_700.0 - decile * 300.0, abs=0.01)
    assert high == pytest.approx(1_700.0 + decile * 300.0, abs=0.01)


def test_a_relative_range_is_read_against_the_base() -> None:
    assert _band(Range(0.75, 1.25, relative=True), base=12_000.0) == (9_000.0, 15_000.0)


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
