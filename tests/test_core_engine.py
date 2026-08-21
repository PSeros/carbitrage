"""Acquisition modes, chains, and the comparison result.

These cover the behaviours the workbook does not exercise: financing, the
unequal-lives guard, the double-counting warning, and the guards around IRR.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from carbitrage import (
    Alternative,
    ComparisonResult,
    Component,
    Financed,
    Household,
    Lease,
    Purchase,
    ReplacementChain,
    Timeline,
    Usage,
    Vehicle,
    compare,
    optimal_replacement_age,
)
from carbitrage.errors import (
    CarbitrageError,
    DoubleCountingWarning,
    UnequalLivesError,
)
from carbitrage.residual import GeometricDecline
from tests.fixtures import workbook_base_case as wb


@pytest.fixture
def tl() -> Timeline:
    return wb.timeline()


@pytest.fixture
def usage() -> Usage:
    return wb.usage()


@pytest.fixture
def household() -> Household:
    return wb.household()


def _short_leg() -> Alternative:
    return Alternative(
        vehicle=wb.incumbent_vehicle(),
        acquisition=Purchase(already_owned=True),
        life_years=2,
        disposes_incumbent=False,
        label="incumbent only",
    )


# ------------------------------------------------------------- unequal lives


def test_unequal_lives_without_a_chain_raises(tl: Timeline, usage: Usage) -> None:
    with pytest.raises(UnequalLivesError, match="shorter than the 6-year horizon"):
        compare([wb.alternative_a1(), _short_leg()], tl, usage=usage, incumbent=wb.incumbent())


def test_the_error_names_both_remedies(tl: Timeline, usage: Usage) -> None:
    with pytest.raises(UnequalLivesError) as excinfo:
        compare([wb.alternative_a1(), _short_leg()], tl, usage=usage)
    message = str(excinfo.value)
    assert "ReplacementChain" in message
    assert "allow_unequal_lives" in message


def test_unequal_lives_can_be_opted_into(tl: Timeline, usage: Usage) -> None:
    result = compare(
        [wb.alternative_a1(), _short_leg()],
        tl,
        usage=usage,
        incumbent=wb.incumbent(),
        allow_unequal_lives=True,
    )
    assert len(result.evaluations) == 2


def test_a_chain_covers_the_horizon_and_is_accepted(tl: Timeline, usage: Usage) -> None:
    result = compare(
        [wb.alternative_a1(), wb.alternative_a4()],
        tl,
        usage=usage,
        household=wb.household(),
        incumbent=wb.incumbent(),
    )
    assert result.best().name == wb.A1


def test_an_empty_comparison_raises(tl: Timeline) -> None:
    with pytest.raises(UnequalLivesError, match="nothing to compare"):
        compare([], tl)


# ------------------------------------------------------------------- chains


def test_a_chain_needs_at_least_two_legs() -> None:
    with pytest.raises(CarbitrageError, match="at least two legs"):
        ReplacementChain(wb.alternative_a1())


def test_every_leg_but_the_last_needs_a_life() -> None:
    with pytest.raises(UnequalLivesError, match="has no life_years"):
        ReplacementChain(wb.alternative_a1(), wb.alternative_a1())


def test_a_leg_that_reaches_the_horizon_leaves_no_room(tl: Timeline, usage: Usage) -> None:
    long_leg = replace(_short_leg(), life_years=6)
    chain = ReplacementChain(long_leg, wb.alternative_a1())
    with pytest.raises(UnequalLivesError, match="already reaches the horizon"):
        compare([chain], tl, usage=usage, incumbent=wb.incumbent())


def test_the_successor_price_is_escalated_to_its_purchase_date(
    tl: Timeline, usage: Usage, household: Household
) -> None:
    """Annahmen C81: buying the same car in two years costs 1.5 % p.a. more."""
    result = compare(
        [wb.alternative_a4()], tl, usage=usage, household=household, incumbent=wb.incumbent()
    )
    line = next(
        d for d in result.detail(wb.A4) if d.description.startswith("Purchase price of Hyundai")
    )
    expected = -wb.SUCCESSOR_PRICE * tl.discount_factor(24)
    assert line.pv == pytest.approx(expected, abs=0.01)


def test_escalation_of_the_successor_can_be_switched_off(
    tl: Timeline, usage: Usage, household: Household
) -> None:
    keep = replace(_short_leg(), label="LPG incumbent")
    chain = ReplacementChain(
        keep, wb.alternative_a1(), label="flat", escalate_successor_price=False
    )
    result = compare(
        [chain], tl, usage=usage, household=household, incumbent=wb.incumbent()
    )
    line = next(
        d for d in result.detail("flat") if d.description.startswith("Purchase price of Hyundai")
    )
    assert line.pv == pytest.approx(-wb.EV_PRICE * tl.discount_factor(24), abs=0.01)


def test_a_later_leg_cannot_resell_the_incumbent(
    tl: Timeline, usage: Usage, household: Household
) -> None:
    """The successor inherits an incumbent that is already gone."""
    result = compare(
        [wb.alternative_a4()], tl, usage=usage, household=household, incumbent=wb.incumbent()
    )
    disposals = [d for d in result.detail(wb.A4) if d.label is Component.DISPOSAL]
    assert disposals == []


# -------------------------------------------------------------- acquisition


def test_financing_at_the_discount_rate_equals_a_cash_purchase(
    tl: Timeline, usage: Usage
) -> None:
    """The timing of the money is neutral only when the two rates agree."""
    cash = Alternative(vehicle=wb.ev_vehicle(), acquisition=Purchase(), label="cash")
    loan = Alternative(
        vehicle=wb.ev_vehicle(),
        acquisition=Financed(loan_rate=tl.rate, term_months=72),
        label="loan",
    )
    result = compare([cash, loan], tl, usage=usage)
    assert result["loan"].npv == pytest.approx(result["cash"].npv, abs=1e-6)


def test_borrowing_above_the_discount_rate_costs_money(tl: Timeline, usage: Usage) -> None:
    cash = Alternative(vehicle=wb.ev_vehicle(), acquisition=Purchase(), label="cash")
    dear = Alternative(
        vehicle=wb.ev_vehicle(),
        acquisition=Financed(loan_rate=0.09, term_months=72),
        label="dear",
    )
    result = compare([cash, dear], tl, usage=usage)
    assert result["dear"].npv < result["cash"].npv


def test_borrowing_below_the_discount_rate_is_worth_something(
    tl: Timeline, usage: Usage
) -> None:
    cash = Alternative(vehicle=wb.ev_vehicle(), acquisition=Purchase(), label="cash")
    cheap = Alternative(
        vehicle=wb.ev_vehicle(),
        acquisition=Financed(loan_rate=0.0, term_months=72),
        label="cheap",
    )
    result = compare([cash, cheap], tl, usage=usage)
    assert result["cheap"].npv > result["cash"].npv


def test_a_lease_chains_contracts_across_the_horizon(
    tl: Timeline, usage: Usage, household: Household
) -> None:
    """Two 36-month contracts over six years, the second at the escalated rate."""
    result = compare(
        [wb.alternative_a2()], tl, usage=usage, household=household, incumbent=wb.incumbent()
    )
    rents = [d for d in result.detail(wb.A2) if d.description.startswith("Lease rent")]
    assert len(rents) == 2
    assert "contract 2" in rents[1].description


def test_excess_mileage_settles_at_each_term_end(
    tl: Timeline, usage: Usage, household: Household
) -> None:
    """Annahmen C52: 2,000 km over, at 0.12 EUR, for three years is 720 EUR a term."""
    result = compare(
        [wb.alternative_a2()], tl, usage=usage, household=household, incumbent=wb.incumbent()
    )
    settlements = [d for d in result.detail(wb.A2) if d.description.startswith("Excess mileage")]
    assert len(settlements) == 2
    assert settlements[0].pv == pytest.approx(-720.0 * tl.discount_factor(36), abs=1e-6)
    assert settlements[1].pv == pytest.approx(-720.0 * tl.discount_factor(72), abs=1e-6)


def test_mileage_inside_the_allowance_settles_nothing(tl: Timeline, household: Household) -> None:
    result = compare(
        [wb.alternative_a2()],
        tl,
        usage=Usage(annual_km=8_000),
        household=household,
        incumbent=wb.incumbent(),
    )
    assert not [d for d in result.detail(wb.A2) if d.description.startswith("Excess mileage")]


def test_a_subsidised_lease_rate_plus_an_incentive_warns() -> None:
    with pytest.warns(DoubleCountingWarning, match="counted twice"):
        Alternative(
            vehicle=wb.ev_vehicle(),
            acquisition=Lease(monthly_rate=99.0, term_months=36, rate_includes_subsidy=True),
            incentives=wb.ev_incentives(),
        )


def test_a_subsidised_lease_rate_alone_does_not_warn() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DoubleCountingWarning)
        Alternative(
            vehicle=wb.ev_vehicle(),
            acquisition=Lease(monthly_rate=99.0, term_months=36, rate_includes_subsidy=True),
        )


def test_an_open_calculation_lease_is_rejected_rather_than_approximated() -> None:
    with pytest.raises(CarbitrageError, match="residual risk on the lessor"):
        Lease(monthly_rate=199.0, residual_risk_borne_by="lessee")


# ------------------------------------------------- optimal replacement age


def test_replacement_age_reports_cost_as_positive(tl: Timeline, usage: Usage) -> None:
    table = optimal_replacement_age(
        wb.alternative_a1(), timeline=tl, candidates=[4, 6], usage=usage,
        household=wb.household(),
    )
    assert all(value > 0 for value in table.eac)
    assert all(value < 0 for value in table.pv)


def test_rising_running_costs_produce_an_interior_optimum(
    tl: Timeline, usage: Usage
) -> None:
    """The classical U-shape needs costs that actually grow with age."""
    rising = replace(tl, cost_escalation=0.12)
    table = optimal_replacement_age(
        wb.alternative_a1(),
        timeline=rising,
        candidates=[3, 4, 6, 8, 10, 14],
        usage=usage,
        household=wb.household(),
    )
    assert table.best_age not in (3, 14)  # neither end of the range
    assert table.best_eac == min(table.eac)


def test_replacement_age_needs_candidates(tl: Timeline) -> None:
    with pytest.raises(CarbitrageError, match="at least one candidate"):
        optimal_replacement_age(wb.alternative_a1(), timeline=tl, candidates=[])


def test_replacement_age_rejects_non_positive_ages(tl: Timeline) -> None:
    with pytest.raises(CarbitrageError, match="must be positive"):
        optimal_replacement_age(wb.alternative_a1(), timeline=tl, candidates=[0, 5])


# ------------------------------------------------------------------ results


@pytest.fixture
def result(tl: Timeline, usage: Usage, household: Household) -> ComparisonResult:
    return compare(
        wb.all_alternatives(), tl, usage=usage, household=household, incumbent=wb.incumbent()
    )


def test_duplicate_names_are_rejected(tl: Timeline, usage: Usage) -> None:
    twice = replace(wb.alternative_a1(), label="same")
    with pytest.raises(CarbitrageError, match="names must be unique"):
        compare([twice, replace(twice)], tl, usage=usage)


def test_cost_per_km_without_mileage_raises(tl: Timeline) -> None:
    result = compare([wb.alternative_a3()], tl)
    with pytest.raises(CarbitrageError, match="needs a positive Usage"):
        result[wb.A3].cost_per_km()


def test_an_immaterial_lead_is_reported_as_such(tl: Timeline, usage: Usage) -> None:
    a = Alternative(vehicle=wb.ev_vehicle(), acquisition=Purchase(), label="a")
    nearly = replace(wb.ev_vehicle(), price=wb.EV_PRICE + 50)
    b = Alternative(vehicle=nearly, acquisition=Purchase(), label="b")
    result = compare([a, b], tl, usage=usage)
    assert not result.is_material()
    assert "too close to call" in result.verdict()


def test_materiality_threshold_is_configurable(tl: Timeline, usage: Usage) -> None:
    a = Alternative(vehicle=wb.ev_vehicle(), acquisition=Purchase(), label="a")
    nearly = replace(wb.ev_vehicle(), price=wb.EV_PRICE + 50)
    b = Alternative(vehicle=nearly, acquisition=Purchase(), label="b")
    assert compare([a, b], tl, usage=usage, materiality=0.0).is_material()


def test_ranking_uses_npv_not_absolute_value(tl: Timeline, usage: Usage) -> None:
    """A generously subsidised option can have a positive NPV; abs() would invert it."""
    cheap = Vehicle(
        name="heavily subsidised",
        price=1_000.0,
        energy=wb.ev_electricity(),
        residual=GeometricDecline(0.05),
    )
    from carbitrage import ThgQuote

    generous = Alternative(
        vehicle=cheap, incentives=(ThgQuote(annual_amount=5_000.0),), label="generous"
    )
    plain = Alternative(vehicle=wb.ev_vehicle(), label="plain")
    result = compare([generous, plain], tl, usage=usage)
    assert result["generous"].npv > 0
    assert result.best().name == "generous"


def test_pv_of_outflows_is_not_an_absolute_value(tl: Timeline, usage: Usage) -> None:
    from carbitrage import ThgQuote

    cheap = Vehicle(
        name="heavily subsidised",
        price=1_000.0,
        energy=wb.ev_electricity(),
        residual=GeometricDecline(0.05),
    )
    generous = Alternative(
        vehicle=cheap, incentives=(ThgQuote(annual_amount=5_000.0),), label="generous"
    )
    result = compare([generous], tl, usage=usage)
    assert result["generous"].pv_of_outflows < 0


def test_incremental_favours_the_better_alternative(result: ComparisonResult) -> None:
    inc = result.incremental(wb.A1, wb.A4)
    assert inc.pv > 0
    assert inc.favours == wb.A1


def test_incremental_is_antisymmetric(result: ComparisonResult) -> None:
    assert result.incremental(wb.A1, wb.A4).pv == pytest.approx(
        -result.incremental(wb.A4, wb.A1).pv
    )


def test_irr_is_withheld_when_the_stream_never_changes_sign(
    tl: Timeline, usage: Usage
) -> None:
    """One alternative dominates in every single period, so no rate equates them."""
    costlier = replace(wb.ev_vehicle(), insurance=wb.EV_INSURANCE + 200)
    a = Alternative(vehicle=wb.ev_vehicle(), label="a")
    b = Alternative(vehicle=costlier, label="b")
    inc = compare([a, b], tl, usage=usage).incremental("a", "b")
    assert inc.irr is None
    assert "never changes sign" in inc.irr_note


def test_irr_is_reported_when_exactly_one_sign_change_exists(
    tl: Timeline, usage: Usage
) -> None:
    """Pay more up front, save every month after: a classic single-crossing stream."""
    dearer = replace(wb.ev_vehicle(), price=wb.EV_PRICE + 5_000, insurance=wb.EV_INSURANCE + 200)
    a = Alternative(vehicle=wb.ev_vehicle(), label="a")
    b = Alternative(vehicle=dearer, label="b")
    inc = compare([a, b], tl, usage=usage).incremental("b", "a")
    assert inc.irr is not None
    assert "root unique" in inc.irr_note


def test_irr_is_withheld_when_several_roots_exist(result: ComparisonResult) -> None:
    inc = result.incremental(wb.A1, wb.A4)
    assert inc.irr is None
    assert "not unique" in inc.irr_note


def test_payback_is_reported_as_a_secondary_criterion(result: ComparisonResult) -> None:
    inc = result.incremental(wb.A1, wb.A4)
    assert inc.payback_years is not None
    assert inc.payback_years > 0


def test_breakdown_sums_for_every_alternative(result: ComparisonResult) -> None:
    for name in result.names:
        assert sum(result.breakdown(name).values()) == pytest.approx(
            result[name].npv, abs=1e-6
        )


def test_detail_sums_to_the_same_total(result: ComparisonResult) -> None:
    for name in result.names:
        assert sum(line.pv for line in result.detail(name)) == pytest.approx(
            result[name].npv, abs=1e-6
        )


def test_unknown_alternative_names_are_reported_helpfully(result: ComparisonResult) -> None:
    with pytest.raises(KeyError, match="available"):
        result["not an alternative"]


def test_to_rows_is_ranked_and_complete(result: ComparisonResult) -> None:
    rows = result.to_rows(baseline=wb.A4)
    assert [row["rank"] for row in rows] == [1, 2, 3, 4, 5]
    assert rows[0]["alternative"] == wb.A1
    baseline_row = next(row for row in rows if row["alternative"] == wb.A4)
    assert baseline_row["delta_vs_baseline"] == pytest.approx(0.0)


def test_to_frame_matches_to_rows(result: ComparisonResult) -> None:
    pd = pytest.importorskip("pandas")
    frame = result.to_frame(baseline=wb.A4)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame["alternative"]) == [row["alternative"] for row in result.to_rows(wb.A4)]
