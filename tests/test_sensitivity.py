"""Parameter overrides, grids, switch points, tornados and Monte Carlo."""

from __future__ import annotations

import numpy as np
import pytest

from carbitrage import params
from carbitrage.analysis import Case
from carbitrage.errors import CarbitrageError
from carbitrage.sensitivity import (
    LogNormal,
    Normal,
    Range,
    SwitchPoint,
    Triangular,
    Uniform,
    advantage,
    monte_carlo,
    npv_of,
    one_way,
    switch_point,
    switch_point_report,
    tornado,
    two_way,
)
from tests.fixtures import workbook_base_case as wb


@pytest.fixture(scope="module")
def case() -> Case:
    return Case(
        alternatives=tuple(wb.all_alternatives()),
        timeline=wb.timeline(),
        usage=wb.usage(),
        household=wb.household(),
        incumbent=wb.incumbent(),
    )


# ------------------------------------------------------------------ params


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("discount_rate", 0.03),
        ("annual_km", 12_000.0),
        ("energy_escalation", 0.02),
        ("lpg_price", 0.99),
        ("petrol_price", 2.10),
        ("home_electricity_price", 0.30),
        ("taxable_income", 55_000.0),
    ],
)
def test_aliases_read_the_workbook_inputs(case: Case, alias: str, expected: float) -> None:
    assert params.get_param(case, alias) == pytest.approx(expected)


def test_an_alias_can_address_several_places_at_once(case: Case) -> None:
    """The autogas price occurs in the chain's first leg and in the incumbent."""
    paths = params.resolve(case, "lpg_price")
    assert len(paths) > 1
    updated = params.set_param(case, "lpg_price", 1.35)
    for path in paths:
        assert params._get_path(updated, path) == pytest.approx(1.35)


def test_an_ambiguous_alias_refuses_to_report_one_base_value(case: Case) -> None:
    """The three vehicles depreciate at 15 %, 13 % and 11 %; there is no single rate."""
    with pytest.raises(CarbitrageError, match="no single base value"):
        params.get_param(case, "residual_rate")


def test_scaling_preserves_modelled_differences(case: Case) -> None:
    scaled = params.scale_param(case, "residual_rate", 2.0)
    for path in params.resolve(case, "residual_rate"):
        before = params._get_path(case, path)
        assert params._get_path(scaled, path) == pytest.approx(before * 2.0)


def test_dotted_paths_reach_inside_a_chain(case: Case) -> None:
    path = "alternatives[A4 Incumbent then EV].legs[LPG incumbent].acquisition.upfront_extra"
    assert params.get_param(case, path) == pytest.approx(wb.REPAIR_COST)
    assert params.get_param(params.set_param(case, path, 6_000.0), path) == 6_000.0


def test_overriding_leaves_the_original_untouched(case: Case) -> None:
    before = params.get_param(case, "annual_km")
    params.set_param(case, "annual_km", 30_000)
    assert params.get_param(case, "annual_km") == before


def test_bool_fields_stay_boolean(case: Case) -> None:
    path = (
        "alternatives[A4 Incumbent then EV].legs[Hyundai Inster (deferred)]"
        ".incentives[0].available"
    )
    updated = params.set_param(case, path, 0)
    assert params._get_path(updated, path) is False


def test_integer_fields_stay_integral(case: Case) -> None:
    updated = params.set_param(case, "household.children", 2.0)
    assert params._get_path(updated, "household.children") == 2
    assert isinstance(params._get_path(updated, "household.children"), int)


def test_unknown_names_are_reported_helpfully(case: Case) -> None:
    with pytest.raises(CarbitrageError, match="has no field"):
        params.get_param(case, "timeline.not_a_field")


def test_unknown_alternative_names_are_reported_helpfully(case: Case) -> None:
    with pytest.raises(CarbitrageError, match="has no item named"):
        params.get_param(case, "alternatives[nope].vehicle.price")


def test_describe_parameters_lists_the_addressable_tree(case: Case) -> None:
    described = params.describe_parameters(case)
    assert "timeline.rate" in described
    assert "alternatives[A1 EV bought new].vehicle.price" in described


# ------------------------------------------------------------------- grids


def test_one_way_records_every_alternative(case: Case) -> None:
    grid = one_way(case, "annual_km", [5_000, 12_000, 30_000])
    assert grid.npv.shape == (3, 5)
    assert grid.names == tuple(wb.TARGETS)


def test_low_mileage_favours_the_lease(case: Case) -> None:
    """Below the contractual allowance the lease carries no excess-mileage charge.

    Above it, the charge accrues on every kilometre and across both contracts,
    which is what turns the lease from the cheapest option into the dearest.
    """
    grid = one_way(case, "annual_km", [5_000, 12_000, 30_000])
    assert grid.winners == (wb.A2, wb.A1, wb.A1)


def test_one_way_reflects_the_base_case_at_the_base_value(case: Case) -> None:
    grid = one_way(case, "annual_km", [12_000])
    assert grid.npv[0, 0] == pytest.approx(wb.TARGETS[wb.A1].npv, abs=1.0)


def test_more_mileage_costs_more_for_every_alternative(case: Case) -> None:
    grid = one_way(case, "annual_km", [5_000, 30_000])
    assert np.all(grid.npv[1] < grid.npv[0])


def test_one_way_detects_a_change_of_winner(case: Case) -> None:
    grid = one_way(case, "discount_rate", [0.0, 0.05, 0.10, 0.20])
    changes = grid.winner_changes()
    assert changes
    assert changes[0][1] == wb.A1


def test_one_way_needs_values(case: Case) -> None:
    with pytest.raises(CarbitrageError, match="at least one value"):
        one_way(case, "annual_km", [])


def test_two_way_matches_the_workbook_layout(case: Case) -> None:
    """Sensitivitaet table 1: mileage down the side, autogas price across the top."""
    grid = two_way(
        case,
        "annual_km",
        [5_000, 10_000, 15_000],
        "lpg_price",
        [0.85, 0.99, 1.22],
        metric=advantage(wb.A1, wb.A4),
    )
    assert grid.values.shape == (3, 3)
    # The advantage of buying now grows with mileage and with the gas price.
    assert np.all(np.diff(grid.values, axis=0) > 0)
    assert np.all(np.diff(grid.values, axis=1) > 0)


def test_two_way_centre_cell_reproduces_the_base_case(case: Case) -> None:
    grid = two_way(
        case, "annual_km", [12_000], "lpg_price", [0.99], metric=advantage(wb.A1, wb.A4)
    )
    expected = wb.TARGETS[wb.A1].npv - wb.TARGETS[wb.A4].npv
    assert grid.values[0, 0] == pytest.approx(expected, abs=1.0)


def test_two_way_needs_both_axes(case: Case) -> None:
    with pytest.raises(CarbitrageError, match="at least one value on each axis"):
        two_way(case, "annual_km", [], "lpg_price", [1.0])


# ----------------------------------------------------------- switch points


def test_switch_point_finds_the_break_even_repair_cost(case: Case) -> None:
    """Cross-check against Sensitivitaet C38.

    The workbook computes 594.94 with an annual closed form and states that this
    approximation differs from its own monthly grid.  Solving on the monthly
    engine gives about 570, and the two agreeing this closely is the point of
    the check.
    """
    path = "alternatives[A4 Incumbent then EV].legs[LPG incumbent].acquisition.upfront_extra"
    found = switch_point(case, path, (wb.A1, wb.A4), bounds=(0.0, 10_000.0))
    assert found is not None
    assert found.value == pytest.approx(594.94, abs=50.0)
    assert found.favoured_below == wb.A4
    assert found.favoured_above == wb.A1


def test_the_solved_switch_point_actually_equalises_the_two(case: Case) -> None:
    path = "alternatives[A4 Incumbent then EV].legs[LPG incumbent].acquisition.upfront_extra"
    found = switch_point(case, path, (wb.A1, wb.A4), bounds=(0.0, 10_000.0))
    assert found is not None
    at = params.set_param(case, path, found.value).run()
    assert at[wb.A1].npv == pytest.approx(at[wb.A4].npv, abs=1e-3)


def test_no_switch_point_for_mileage_matching_the_workbook(case: Case) -> None:
    """Sensitivitaet C37 reports the break-even mileage as unreachable.

    Buying now wins at every mileage, so there is nothing to solve for, and the
    library says so instead of returning a spurious root.
    """
    report = switch_point_report(case, "annual_km", (wb.A1, wb.A4))
    assert not isinstance(report, SwitchPoint)
    assert "never flips" in report.reason
    assert switch_point(case, "annual_km", (wb.A1, wb.A4)) is None


def test_switch_point_locates_the_lease_crossover(case: Case) -> None:
    """The lease stops paying just above its 10,000 km allowance."""
    found = switch_point(case, "annual_km", (wb.A1, wb.A2), bounds=(1_000.0, 40_000.0))
    assert found is not None
    assert wb.LEASE_INCLUDED_KM < found.value < 12_000.0
    assert found.favoured_below == wb.A2
    assert found.favoured_above == wb.A1
    at = params.set_param(case, "annual_km", found.value).run()
    assert at[wb.A1].npv == pytest.approx(at[wb.A2].npv, abs=1e-3)


def test_the_discount_rate_has_a_switch_point(case: Case) -> None:
    """Waiting is worth more when capital is dear, so a high enough rate flips it."""
    found = switch_point(case, "discount_rate", (wb.A1, wb.A4), bounds=(0.0, 0.30))
    assert found is not None
    assert 0.05 < found.value < 0.25
    assert found.favoured_below == wb.A1
    assert found.favoured_above == wb.A4


def test_switch_point_describes_itself_readably(case: Case) -> None:
    found = switch_point(case, "discount_rate", (wb.A1, wb.A4), bounds=(0.0, 0.30))
    assert found is not None
    assert "switches the answer at" in found.describe()
    assert wb.A1 in found.describe()


def test_switch_point_rejects_reversed_bounds(case: Case) -> None:
    with pytest.raises(CarbitrageError, match="bounds must be increasing"):
        switch_point(case, "annual_km", (wb.A1, wb.A4), bounds=(100.0, 10.0))


def test_the_scan_survives_values_the_model_rejects(case: Case) -> None:
    """A share above 1 is invalid; the search skips it rather than crashing."""
    report = switch_point_report(
        case, "home_share", (wb.A1, wb.A4), bounds=(0.0, 5.0)
    )
    assert report is not None  # either outcome is fine; it must not raise


# ---------------------------------------------------------------- tornado


def test_tornado_ranks_by_absolute_swing(case: Case) -> None:
    result = tornado(
        case,
        {
            "annual_km": Range(5_000, 30_000),
            "lpg_price": Range(0.85, 1.35),
            "discount_rate": Range(0.0, 0.08),
        },
        metric=advantage(wb.A1, wb.A4),
    )
    swings = [bar.swing for bar in result.bars]
    assert swings == sorted(swings, reverse=True)
    assert all(swing >= 0 for swing in swings)


def test_tornado_base_metric_is_the_unperturbed_case(case: Case) -> None:
    result = tornado(case, ["annual_km"], metric=advantage(wb.A1, wb.A4))
    expected = wb.TARGETS[wb.A1].npv - wb.TARGETS[wb.A4].npv
    assert result.base_metric == pytest.approx(expected, abs=1.0)


def test_tornado_accepts_relative_ranges_for_ambiguous_aliases(case: Case) -> None:
    result = tornado(
        case, {"residual_rate": Range(0.7, 1.4, relative=True)}, metric=advantage(wb.A1, wb.A4)
    )
    bar = result.bars[0]
    assert bar.relative
    assert bar.endpoints() == ("x0.7", "x1.4")
    assert bar.swing > 0


def test_tornado_defaults_to_a_relative_range(case: Case) -> None:
    result = tornado(case, ["annual_km"], metric=advantage(wb.A1, wb.A4))
    assert result.bars[0].low_value == pytest.approx(12_000 * 0.75)
    assert result.bars[0].high_value == pytest.approx(12_000 * 1.25)


def test_tornado_needs_parameters(case: Case) -> None:
    with pytest.raises(CarbitrageError, match="at least one parameter"):
        tornado(case, [])


def test_tornado_rejects_an_unknown_parameter(case: Case) -> None:
    with pytest.raises(CarbitrageError):
        tornado(case, ["not_a_parameter"])


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


@pytest.fixture(scope="module")
def simulation(case: Case):
    return monte_carlo(
        case,
        {
            "lpg_price": Triangular(0.80, 0.99, 1.40),
            "home_electricity_price": Triangular(0.22, 0.30, 0.45),
            "annual_km": Uniform(8_000, 18_000),
        },
        between=(wb.A1, wb.A4),
        n=300,
        seed=7,
    )


def test_monte_carlo_returns_the_distribution_of_the_difference(simulation) -> None:
    assert simulation.n == 300
    assert simulation.differences.shape == (300,)
    difference = simulation.npv[wb.A1] - simulation.npv[wb.A4]
    assert simulation.differences == pytest.approx(difference)


def test_monte_carlo_reports_a_probability(simulation) -> None:
    p = simulation.probability_a_beats_b()
    assert 0.0 <= p <= 1.0
    assert p > 0.9  # the base case advantage is large and robust


def test_monte_carlo_percentiles_are_ordered(simulation) -> None:
    q = simulation.percentiles((5, 50, 95))
    assert q[5.0] <= q[50.0] <= q[95.0]


def test_monte_carlo_describes_itself(simulation) -> None:
    assert "beats" in simulation.describe()
    assert "percentile" in simulation.describe()


def test_monte_carlo_is_reproducible(case: Case) -> None:
    spec = {"lpg_price": Uniform(0.8, 1.3)}
    a = monte_carlo(case, spec, between=(wb.A1, wb.A4), n=40, seed=99)
    b = monte_carlo(case, spec, between=(wb.A1, wb.A4), n=40, seed=99)
    assert a.differences == pytest.approx(b.differences)


def test_correlation_widens_the_spread_of_the_difference(case: Case) -> None:
    """Correlated energy prices move the alternatives apart, not together.

    The incumbent burns gas and the electric car burns electricity.  When the
    two prices rise together the difference between the alternatives is *less*
    affected than when they move independently, so the spread narrows.  Either
    way, assuming independence gives the wrong answer, which is the point.
    """
    spec = {
        "lpg_price": Normal(0.99, 0.20),
        "home_electricity_price": Normal(0.30, 0.06),
    }
    independent = monte_carlo(case, spec, between=(wb.A1, wb.A4), n=400, seed=3)
    correlated = monte_carlo(
        case,
        spec,
        between=(wb.A1, wb.A4),
        n=400,
        seed=3,
        correlation=[[1.0, 0.8], [0.8, 1.0]],
    )
    assert np.std(correlated.differences) != pytest.approx(
        float(np.std(independent.differences)), rel=0.05
    )


def test_correlated_draws_have_the_requested_correlation(case: Case) -> None:
    sim = monte_carlo(
        case,
        {"lpg_price": Normal(0.99, 0.2), "home_electricity_price": Normal(0.30, 0.06)},
        between=(wb.A1, wb.A4),
        n=3_000,
        seed=11,
        correlation=[[1.0, 0.7], [0.7, 1.0]],
    )
    observed = float(np.corrcoef(sim.draws.T)[0, 1])
    assert observed == pytest.approx(0.7, abs=0.06)


@pytest.mark.parametrize(
    ("matrix", "match"),
    [
        ([[1.0, 0.0]], "must be 2x2"),
        ([[1.0, 0.5], [0.4, 1.0]], "must be symmetric"),
        ([[2.0, 0.0], [0.0, 1.0]], "ones on the diagonal"),
        ([[1.0, 1.5], [1.5, 1.0]], "not positive definite"),
    ],
)
def test_correlation_matrices_are_validated(case: Case, matrix, match: str) -> None:
    with pytest.raises(CarbitrageError, match=match):
        monte_carlo(
            case,
            {"lpg_price": Normal(0.99, 0.2), "home_electricity_price": Normal(0.30, 0.06)},
            between=(wb.A1, wb.A4),
            n=5,
            correlation=matrix,
        )


def test_monte_carlo_validates_its_inputs(case: Case) -> None:
    with pytest.raises(CarbitrageError, match="at least one distribution"):
        monte_carlo(case, {}, between=(wb.A1, wb.A4))
    with pytest.raises(CarbitrageError, match="n must be positive"):
        monte_carlo(case, {"lpg_price": Uniform(0.8, 1.2)}, between=(wb.A1, wb.A4), n=0)


# ------------------------------------------------- the result-level surface


def test_the_result_exposes_the_same_analysis(case: Case) -> None:
    result = case.run()
    assert result.switch_point("annual_km", (wb.A1, wb.A4)) is None
    assert result.tornado(["annual_km"]).bars
    assert result.one_way("annual_km", [10_000, 14_000]).npv.shape == (2, 5)


def test_analysis_without_a_case_explains_itself() -> None:
    from carbitrage.result import ComparisonResult

    bare = ComparisonResult(evaluations=case_evaluations())
    with pytest.raises(CarbitrageError, match="needs the inputs"):
        bare.tornado(["annual_km"])


def case_evaluations():
    from carbitrage import compare

    return compare(
        [wb.alternative_a1()], wb.timeline(), usage=wb.usage(), household=wb.household()
    ).evaluations


def test_npv_of_metric_reads_one_alternative(case: Case) -> None:
    grid = one_way(case, "annual_km", [12_000], metric=npv_of(wb.A1))
    assert grid.metric[0] == pytest.approx(wb.TARGETS[wb.A1].npv, abs=1.0)
