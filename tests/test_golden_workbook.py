"""Regression against the reference workbook.

Every figure here comes from ``E-Auto_Kapitalwertmodell_2026_v3.xlsx``.  The
workbook is not opened at test time: its numbers live in
``tests/fixtures/workbook_base_case.py``.

The tolerance is EUR 1 on present values, as the brief requires.  In practice
the library reproduces the workbook to the cent, because it implements the same
conventions rather than approximating them.
"""

from __future__ import annotations

import pytest

from carbitrage import ComparisonResult, Component, compare
from tests.fixtures import workbook_base_case as wb

TOLERANCE = 1.0


@pytest.fixture(scope="module")
def result() -> ComparisonResult:
    return compare(
        wb.all_alternatives(),
        wb.timeline(),
        usage=wb.usage(),
        household=wb.household(),
        incumbent=wb.incumbent(),
    )


@pytest.mark.parametrize("name", list(wb.TARGETS))
def test_net_present_value_matches_workbook(result: ComparisonResult, name: str) -> None:
    assert result[name].npv == pytest.approx(wb.TARGETS[name].npv, abs=TOLERANCE)


@pytest.mark.parametrize("name", list(wb.TARGETS))
def test_pv_of_outflows_matches_workbook(result: ComparisonResult, name: str) -> None:
    assert result[name].pv_of_outflows == pytest.approx(
        wb.TARGETS[name].pv_of_outflows, abs=TOLERANCE
    )


@pytest.mark.parametrize("name", list(wb.TARGETS))
def test_equivalent_annual_cost_matches_workbook(result: ComparisonResult, name: str) -> None:
    assert result[name].eac() == pytest.approx(wb.TARGETS[name].eac, abs=TOLERANCE)


@pytest.mark.parametrize("name", list(wb.TARGETS))
def test_cost_per_km_matches_workbook(result: ComparisonResult, name: str) -> None:
    # A cent per kilometre over 12,000 km is EUR 120 a year, so this is tight.
    assert result[name].cost_per_km() == pytest.approx(
        wb.TARGETS[name].cost_per_km, abs=1e-4
    )


@pytest.mark.parametrize("name", list(wb.TARGETS))
def test_undiscounted_control_total_matches_workbook(
    result: ComparisonResult, name: str
) -> None:
    # Cashflow row 129: the nominal sum, which the workbook carries as a control.
    assert result[name].undiscounted == pytest.approx(
        wb.TARGETS[name].undiscounted, abs=TOLERANCE
    )


def test_capital_recovery_factor_matches_workbook() -> None:
    assert wb.timeline().capital_recovery_factor() == pytest.approx(
        wb.CAPITAL_RECOVERY_FACTOR, abs=1e-12
    )


def test_winner_matches_workbook(result: ComparisonResult) -> None:
    assert result.best().name == wb.WINNER


def test_margin_over_runner_up_matches_workbook(result: ComparisonResult) -> None:
    assert result.margin() == pytest.approx(wb.MARGIN_OVER_RUNNER_UP, abs=TOLERANCE)


def test_the_lead_is_material_at_the_workbook_threshold(result: ComparisonResult) -> None:
    # Ergebnis C15: 1,036 EUR on a 20,844 EUR present value clears 3 %.
    assert result.is_material()


def test_ranking_is_by_npv_not_absolute_value(result: ComparisonResult) -> None:
    order = [e.name for e in result.ranking()]
    assert order[0] == wb.WINNER
    npvs = [result[name].npv for name in order]
    assert npvs == sorted(npvs, reverse=True)


def test_a1_breakdown_sums_to_its_npv(result: ComparisonResult) -> None:
    breakdown = result.breakdown(wb.A1)
    assert sum(breakdown.values()) == pytest.approx(result[wb.A1].npv, abs=1e-6)


def test_a1_breakdown_components_match_workbook(result: ComparisonResult) -> None:
    """Ergebnis rows 25-31, mapped onto this library's finer labels."""
    b = result.breakdown(wb.A1)
    expected = wb.A1_BREAKDOWN
    assert b[Component.ACQUISITION] == pytest.approx(
        expected["acquisition_and_setup"], abs=TOLERANCE
    )
    assert b[Component.DISPOSAL] == pytest.approx(expected["incumbent_disposal"], abs=TOLERANCE)
    assert b[Component.ENERGY] == pytest.approx(expected["energy"], abs=TOLERANCE)
    assert b[Component.RESIDUAL] == pytest.approx(expected["residual"], abs=TOLERANCE)
    # The workbook lumps the grant and the annual credits into two lines.
    assert b[Component.SUBSIDY] == pytest.approx(
        expected["subsidy"] + expected["thg_credits"], abs=TOLERANCE
    )
    # ... and insurance, maintenance and circulation tax into one.
    running = (
        b[Component.INSURANCE] + b[Component.MAINTENANCE] + b.get(Component.TAX, 0.0)
    )
    assert running == pytest.approx(expected["insurance_maintenance_tax"], abs=TOLERANCE)


def test_subsidy_is_discounted_over_its_disbursement_lag(result: ComparisonResult) -> None:
    """Annahmen C17 with C19: 4,000 EUR paid in month 4, not at t=0."""
    grant = next(
        line for line in result.detail(wb.A1) if "Purchase premium" in line.description
    )
    assert grant.pv == pytest.approx(wb.A1_BREAKDOWN["subsidy"], abs=1e-6)
    assert grant.pv < 4_000.0


def test_the_chain_swaps_in_month_24(result: ComparisonResult) -> None:
    """Cashflow H30: the incumbent's sale and the successor's purchase coincide."""
    amounts = result[wb.A4].amounts
    swap = amounts[24]
    # Sale of the incumbent, less the escalated successor price and its wallbox,
    # plus that month's running costs.
    assert swap == pytest.approx(-25554.4029666667, abs=0.01)


def test_the_successor_is_valued_at_its_own_age(result: ComparisonResult) -> None:
    """Annahmen C82: four years old at the horizon, not six."""
    residual = next(
        line
        for line in result.detail(wb.A4)
        if line.label is Component.RESIDUAL and "Hyundai" in line.description
    )
    timeline = wb.timeline()
    expected_pv = wb.SUCCESSOR_RESIDUAL * timeline.discount_factor(72)
    assert residual.pv == pytest.approx(expected_pv, abs=0.01)


def test_the_deferred_premium_is_paid_in_month_28(result: ComparisonResult) -> None:
    """Cashflow H: the grant follows the successor's registration, not t=0."""
    timeline = wb.timeline()
    grant = next(
        line for line in result.detail(wb.A4) if "Purchase premium" in line.description
    )
    assert grant.pv == pytest.approx(4_000.0 * timeline.discount_factor(28), abs=1e-6)


def test_the_incumbent_leg_pays_only_the_repair_at_t0(result: ComparisonResult) -> None:
    """Cashflow H6: keeping the car costs the repair bill and nothing else at t=0.

    The forgone sale proceeds are not booked here; they appear as an inflow on
    every alternative that does dispose of the incumbent, which is the same
    differential expressed the other way round.
    """
    assert result[wb.A4].amounts[0] == pytest.approx(-wb.REPAIR_COST)


def test_disposing_alternatives_receive_the_incumbent_proceeds(
    result: ComparisonResult,
) -> None:
    for name in (wb.A1, wb.A3, wb.A5):
        assert result.breakdown(name)[Component.DISPOSAL] == pytest.approx(
            wb.INCUMBENT_VALUE_NOW
        )


def test_the_lease_collects_no_residual(result: ComparisonResult) -> None:
    """The lessor bears residual risk, so the lessee books no terminal value."""
    assert Component.RESIDUAL not in result.breakdown(wb.A2)


def test_the_used_ev_gets_no_purchase_premium(result: ComparisonResult) -> None:
    """Used vehicles are not eligible, however the incentive is configured."""
    detail = result.detail(wb.A5)
    assert not [line for line in detail if "Purchase premium" in line.description]
    # It does keep the quota credit and the tax exemption, which follow the car.
    assert result.breakdown(wb.A5)[Component.SUBSIDY] > 0


def test_markdown_table_is_readable(result: ComparisonResult) -> None:
    table = result.to_markdown(baseline=wb.A4)
    assert wb.A1 in table
    assert "PV of outflows" in table
    assert "clear difference" in table
    # The premium's minimum holding period is surfaced as a constraint.
    assert "minimum holding period" in table
