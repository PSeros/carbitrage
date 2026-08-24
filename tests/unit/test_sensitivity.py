"""Parameter overrides, grids, switch points, tornados and Monte Carlo."""

from __future__ import annotations

import numpy as np
import pytest

from carbitrage import Alternative, Case, Timeline, Vehicle
from carbitrage.acquisition import Purchase
from carbitrage.cashflow import Component
from carbitrage.context import Usage
from carbitrage.energy import Petrol
from carbitrage.errors import CarbitrageError
from carbitrage.params import Uncertain, spreads
from carbitrage.residual import GeometricDecline
from carbitrage.sensitivity import (
    LogNormal,
    MonteCarlo,
    Normal,
    Range,
    Triangular,
    Uniform,
)
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


def test_a_bare_name_sweeps_the_range_it_declared() -> None:
    """The declaration is already on the parameter; restating it invites a mismatch."""
    bar = case_with(BANDED).run().tornado([BANDED]).bars[0]
    assert bar.endpoints() == ("19,000.00", "21,500.00")


def test_a_declared_relative_range_stays_relative() -> None:
    """Flattening it would erase the differences an alias exists to preserve."""
    relative = Uncertain(20_000.0, "banded_price", Range(0.9, 1.1, relative=True))
    bar = case_with(relative).run().tornado([relative]).bars[0]
    assert bar.relative is True
    assert bar.endpoints() == ("x0.9", "x1.1")
    assert (bar.low_value, bar.high_value) == (18_000.0, 22_000.0)


def test_a_declared_distribution_is_swept_across_its_band() -> None:
    declared = Uncertain(20_000.0, "banded_price", Triangular(18_000.0, 20_000.0, 23_000.0))
    bar = case_with(declared).run().tornado([declared]).bars[0]
    assert (bar.low_value, bar.high_value) == (18_000.0, 23_000.0)


def test_an_explicit_range_overrides_the_declaration() -> None:
    """The caller asked a specific question; the declaration does not narrow it."""
    bar = case_with(BANDED).run().tornado({BANDED: Range(10_000.0, 30_000.0)}).bars[0]
    assert (bar.low_value, bar.high_value) == (10_000.0, 30_000.0)


def test_an_unlisted_tornado_ranks_everything_the_case_declares() -> None:
    """The case already says what is uncertain; listing it again invites a mismatch."""
    bars = case_with(BANDED).run().tornado().bars
    assert [bar.param for bar in bars] == ["banded_price"]
    assert (bars[0].low_value, bars[0].high_value) == (19_000.0, 21_500.0)


def test_an_unlisted_tornado_needs_the_case_to_declare_something(marked: Case) -> None:
    with pytest.raises(CarbitrageError, match="no parameter in this case declares a spread"):
        marked.run().tornado()


def test_an_undeclared_parameter_still_falls_back_to_the_default_range(marked: Case) -> None:
    bar = marked.run().tornado([PRICE], default_range=Range(0.5, 1.5, relative=True)).bars[0]
    assert (bar.low_value, bar.high_value) == (10_000.0, 30_000.0)


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
        (lambda: LogNormal(12_000.0, 1_000.0), "parameters are those of the underlying normal"),
        (lambda: LogNormal(0.0, 1e300), "parameters are those of the underlying normal"),
    ],
)
def test_distributions_validate_their_parameters(factory, match: str) -> None:
    with pytest.raises(CarbitrageError, match=match):
        factory()


# ---------------------------------------------------------- monte carlo


def test_a_bare_name_samples_the_distribution_it_declared() -> None:
    declared = Uncertain(20_000.0, "banded_price", Triangular(18_000.0, 20_000.0, 23_000.0))
    simulation = case_with(declared).run().monte_carlo([declared], between=("a", "b"), n=64, seed=0)
    assert simulation.params == ("banded_price",)
    assert simulation.draws.min() >= 18_000.0
    assert simulation.draws.max() <= 23_000.0


def test_a_mapping_can_defer_one_parameter_and_state_another() -> None:
    declared = Uncertain(20_000.0, "banded_price", Triangular(18_000.0, 20_000.0, 23_000.0))
    simulation = (
        case_with(declared)
        .run()
        .monte_carlo(
            {declared: None, "annual_km": Uniform(8_000.0, 16_000.0)},
            between=("a", "b"),
            n=64,
            seed=0,
        )
    )
    assert simulation.params == ("banded_price", "annual_km")
    assert simulation.draws[:, 1].min() >= 8_000.0


def test_an_unlisted_simulation_samples_everything_the_case_declares() -> None:
    declared = Uncertain(20_000.0, "banded_price", Triangular(18_000.0, 20_000.0, 23_000.0))
    simulation = case_with(declared).run().monte_carlo(between=("a", "b"), n=64, seed=0)
    assert simulation.params == ("banded_price",)
    assert simulation.draws.min() >= 18_000.0


def test_an_unlisted_simulation_passes_over_what_it_cannot_sample() -> None:
    """A range was not offered to the simulation, so it is skipped, not refused."""
    both = case_with(BANDED)  # BANDED declares a Range
    both = both.__class__(
        alternatives=both.alternatives,
        timeline=both.timeline,
        usage=Usage(annual_km=Uncertain(12_000.0, "km", Uniform(8_000.0, 16_000.0))),
    )
    simulation = both.run().monte_carlo(between=("a", "b"), n=32, seed=0)
    assert simulation.params == ("km",)


def test_an_unlisted_simulation_needs_the_case_to_declare_a_distribution(marked: Case) -> None:
    with pytest.raises(CarbitrageError, match="declares a distribution"):
        marked.run().monte_carlo(between=("a", "b"), n=8)


def test_declared_spreads_are_collected_by_family() -> None:
    case = case_with(BANDED)
    assert spreads(case) == {"banded_price": Range(19_000.0, 21_500.0)}
    assert spreads(case, kind=Normal) == {}
    assert spreads(case, kind=Range) == {"banded_price": Range(19_000.0, 21_500.0)}


def test_a_parameter_declaring_nothing_cannot_be_left_to_its_declaration(marked: Case) -> None:
    with pytest.raises(CarbitrageError, match="sticker_price declares no spread"):
        marked.run().monte_carlo([PRICE], between=("a", "b"), n=8)


def test_a_declared_range_is_refused_rather_than_read_as_uniform() -> None:
    """A range says where the value lies, not how likely each value in it is."""
    with pytest.raises(CarbitrageError, match="how likely each value"):
        case_with(BANDED).run().monte_carlo([BANDED], between=("a", "b"), n=8)


# ------------------------------------------------ monte carlo across N

SAMPLED = Uncertain(20_000.0, "sampled_price", Triangular(18_000.0, 20_000.0, 23_000.0))


@pytest.fixture
def three() -> Case:
    """Three alternatives, the first of them holding a sampled price."""

    def car(name: str, sticker: float | Uncertain) -> Vehicle:
        return Vehicle(
            name,
            price=sticker,
            energy=Petrol(consumption=7.0, price=1.80),
            residual=GeometricDecline(0.15),
        )

    return Case(
        alternatives=(
            Alternative(car("a", SAMPLED), Purchase(), label="a"),
            Alternative(car("b", 21_000.0), Purchase(), label="b"),
            Alternative(car("c", 25_000.0), Purchase(), label="c"),
        ),
        timeline=Timeline(horizon_years=3, periods_per_year=12, rate=0.03),
        usage=Usage(annual_km=12_000.0),
    )


def test_a_simulation_carries_every_alternative_by_default(three: Case) -> None:
    simulation = three.run().monte_carlo(n=64, seed=0)
    assert simulation.names == ("a", "b", "c")
    assert set(simulation.npv) == {"a", "b", "c"}
    assert all(values.size == 64 for values in simulation.npv.values())


def test_between_narrows_the_simulation_and_orders_it(three: Case) -> None:
    simulation = three.run().monte_carlo(between=("c", "a"), n=64, seed=0)
    assert simulation.names == ("c", "a")


def test_narrowing_does_not_change_what_the_shared_alternatives_do(three: Case) -> None:
    """The draws are the parameters', not the alternatives': same seed, same trials."""
    everything = three.run().monte_carlo(n=64, seed=3)
    narrowed = three.run().monte_carlo(between=("a", "b"), n=64, seed=3)
    assert np.allclose(everything.npv["a"], narrowed.npv["a"])
    assert np.allclose(everything.npv["b"], narrowed.npv["b"])


def test_win_shares_are_a_distribution_over_the_alternatives(three: Case) -> None:
    simulation = three.run().monte_carlo(n=128, seed=0)
    share = simulation.win_share()
    assert set(share) == {"a", "b", "c"}
    assert sum(share.values()) == pytest.approx(1.0)


def test_regret_is_never_negative_and_zero_for_the_trial_winner(three: Case) -> None:
    simulation = three.run().monte_carlo(n=64, seed=0)
    regret = simulation.regret()
    stacked = np.column_stack([regret[name] for name in simulation.names])
    assert stacked.min() >= 0.0
    assert np.allclose(stacked.min(axis=1), 0.0)


def test_expected_regret_ranks_exactly_as_expected_value_does(three: Case) -> None:
    """E[max] is the same constant for everyone, so the two orderings coincide."""
    simulation = three.run().monte_carlo(n=128, seed=0)
    by_value = sorted(simulation.names, key=lambda n: -simulation.expected_npv()[n])
    by_regret = sorted(simulation.names, key=lambda n: simulation.expected_regret()[n])
    assert by_value == by_regret
    assert simulation.best_by_expected_value() == by_value[0]


def test_a_probability_is_read_from_the_paired_columns(three: Case) -> None:
    simulation = three.run().monte_carlo(n=64, seed=0)
    paired = np.mean(simulation.npv["a"] > simulation.npv["b"])
    assert simulation.probability("a", "b") == pytest.approx(paired)
    assert simulation.probability("b", "a") == pytest.approx(1.0 - paired)


def test_an_alternative_beaten_everywhere_is_named_as_droppable(three: Case) -> None:
    simulation = three.run().monte_carlo(n=64, seed=0)
    assert simulation.win_share()["c"] == 0.0
    assert "c" in simulation.never_best()


def test_a_pairwise_matrix_covers_every_ordered_pair(three: Case) -> None:
    simulation = three.run().monte_carlo(n=64, seed=0)
    matrix = simulation.pairwise()
    assert len(matrix) == 6
    assert matrix[("a", "b")] == pytest.approx(1.0 - matrix[("b", "a")])


def test_a_simulation_reports_both_decision_rules(three: Case) -> None:
    simulation = three.run().monte_carlo(n=64, seed=0)
    text = simulation.describe()
    assert "Highest expected value" in text
    assert "Regret at the 95th percentile" in text
    assert "| Alternative | Expected |" in simulation.to_markdown()


def test_a_name_the_simulation_does_not_carry_is_refused(three: Case) -> None:
    simulation = three.run().monte_carlo(between=("a", "b"), n=8, seed=0)
    with pytest.raises(CarbitrageError, match="not in this simulation"):
        simulation.probability("a", "c")


def test_between_needs_two_alternatives_to_compare(three: Case) -> None:
    with pytest.raises(CarbitrageError, match="at least two alternatives"):
        three.run().monte_carlo(between=("a",), n=8, seed=0)


def test_between_refuses_a_name_the_case_does_not_hold(three: Case) -> None:
    with pytest.raises(CarbitrageError, match="not an alternative in this case"):
        three.run().monte_carlo(between=("a", "z"), n=8, seed=0)


def test_between_refuses_the_same_alternative_twice(three: Case) -> None:
    with pytest.raises(CarbitrageError, match="names 'a' twice"):
        three.run().monte_carlo(between=("a", "a"), n=8, seed=0)


# ---------------------------------------------- components under uncertainty


@pytest.fixture
def two_sources(three: Case) -> Case:
    """The three-way case with mileage sampled too, so two components move apart."""
    return Case(
        alternatives=three.alternatives,
        timeline=three.timeline,
        usage=Usage(annual_km=Uncertain(12_000.0, "km", Uniform(8_000.0, 18_000.0))),
    )


def test_every_trial_s_components_sum_to_that_trial_s_npv(three: Case) -> None:
    simulation = three.run().monte_carlo(n=64, seed=0)
    for name in simulation.names:
        assert np.allclose(simulation.breakdown[name].sum(axis=1), simulation.npv[name])


def test_the_components_are_carried_in_the_order_the_enum_declares(three: Case) -> None:
    simulation = three.run().monte_carlo(n=32, seed=0)
    carried = set(simulation.components)
    assert list(simulation.components) == [c for c in Component if c in carried]
    assert Component.ACQUISITION in simulation.components


def test_a_component_column_is_paired_with_the_total_row_by_row(three: Case) -> None:
    simulation = three.run().monte_carlo(n=64, seed=0)
    acquisition = simulation.component_npv("a", Component.ACQUISITION)
    assert acquisition.size == simulation.n
    assert np.corrcoef(acquisition, simulation.draws[:, 0])[0, 1] < -0.99


def test_expected_components_sum_to_the_expected_npv(three: Case) -> None:
    simulation = three.run().monte_carlo(n=64, seed=0)
    expected = simulation.expected_npv()
    for name, parts in simulation.expected_breakdown().items():
        assert sum(parts.values()) == pytest.approx(expected[name], rel=1e-9, abs=1e-6)


def test_the_frame_indexes_alternatives_then_statistics_low_to_high(three: Case, pandas) -> None:
    simulation = three.run().monte_carlo(n=64, seed=0)
    frame = simulation.breakdown_frame()
    assert list(frame.columns.names) == ["alternative", "statistic"]
    assert list(frame.columns) == [
        (name, label) for name in simulation.names for label in ("p5", "mean", "p95")
    ]


def test_the_mean_sits_where_the_median_would(three: Case, pandas) -> None:
    simulation = three.run().monte_carlo(n=64, seed=0)
    frame = simulation.breakdown_frame(levels=(5, 25, 50, 75, 95))
    labels = [label for _, label in frame.columns[: len(frame.columns) // 3]]
    assert labels == ["p5", "p25", "mean", "p50", "p75", "p95"]


def test_the_frame_rows_are_the_components_and_a_final_total(three: Case, pandas) -> None:
    simulation = three.run().monte_carlo(n=32, seed=0)
    frame = simulation.breakdown_frame()
    assert frame.index.name == "component"
    assert list(frame.index) == [c.name for c in simulation.components] + ["NPV"]


def test_the_mean_column_sums_down_to_the_total(two_sources: Case, pandas) -> None:
    """Expectation is linear, so this one column adds up exactly."""
    simulation = two_sources.run().monte_carlo(n=128, seed=0)
    column = simulation.breakdown_frame()[("a", "mean")]
    assert column.drop("NPV").sum() == pytest.approx(column["NPV"], rel=1e-9, abs=1e-6)


def test_the_percentile_columns_do_not_sum_down_to_the_total(two_sources: Case, pandas) -> None:
    """Two components rarely hit their bad end together, so the total is narrower."""
    simulation = two_sources.run().monte_carlo(n=256, seed=0)
    column = simulation.breakdown_frame()[("a", "p5")]
    assert column.drop("NPV").sum() < column["NPV"]


def test_a_component_no_draw_reaches_reads_the_same_in_every_column(three: Case, pandas) -> None:
    simulation = three.run().monte_carlo(n=64, seed=0)
    row = simulation.breakdown_frame().loc[Component.ENERGY.name, "a"]
    assert row["p5"] == pytest.approx(row["mean"]) == pytest.approx(row["p95"])


def test_component_percentiles_cover_every_component(three: Case) -> None:
    simulation = three.run().monte_carlo(n=64, seed=0)
    percentiles = simulation.component_percentiles("a", (5, 95))
    assert set(percentiles) == set(simulation.components)
    acquisition = percentiles[Component.ACQUISITION]
    assert acquisition[5.0] < acquisition[95.0]


def test_rounding_the_frame_to_whole_units_gives_integers(three: Case, pandas) -> None:
    simulation = three.run().monte_carlo(n=32, seed=0)
    frame = simulation.breakdown_frame(decimals=0)
    assert all(pandas.api.types.is_integer_dtype(dtype) for dtype in frame.dtypes)


def test_an_empty_set_of_levels_leaves_the_mean_alone(three: Case, pandas) -> None:
    simulation = three.run().monte_carlo(n=32, seed=0)
    frame = simulation.breakdown_frame(levels=())
    assert list(frame.columns) == [(name, "mean") for name in simulation.names]


def test_a_level_outside_the_percentile_scale_is_refused(three: Case, pandas) -> None:
    simulation = three.run().monte_carlo(n=32, seed=0)
    with pytest.raises(CarbitrageError, match=r"percentile levels lie in \[0, 100\]"):
        simulation.breakdown_frame(levels=(5, 120))


def test_a_component_no_alternative_carries_is_refused(three: Case) -> None:
    simulation = three.run().monte_carlo(n=32, seed=0)
    with pytest.raises(CarbitrageError, match="appears in no alternative"):
        simulation.component_npv("a", Component.LEASE)


def test_a_simulation_built_by_hand_says_it_has_no_breakdown() -> None:
    bare = MonteCarlo(
        names=("a", "b"),
        npv={"a": np.zeros(4), "b": np.ones(4)},
        params=("price",),
        draws=np.zeros((4, 1)),
    )
    with pytest.raises(CarbitrageError, match="carries no component breakdown"):
        bare.expected_breakdown()


# ------------------------------------------------- the result-level surface
