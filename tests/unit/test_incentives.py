"""Incentives: eligibility, timing, and the non-cash constraints they impose."""

from __future__ import annotations

from datetime import date

import pytest

from carbitrage import Timeline, Vehicle
from carbitrage.context import Context, Household, Usage
from carbitrage.energy import LPG, BivalentSource, Electricity, Hydrogen, Petrol
from carbitrage.errors import CarbitrageError
from carbitrage.incentives import BAFA2026, BAFA_2026_PHEV, ThgQuote, VehicleTaxExemption
from carbitrage.residual import GeometricDecline
from carbitrage.vehicle import Propulsion, VehicleCategory


def ev(**kwargs: object) -> Vehicle:
    defaults: dict[str, object] = {
        "name": "test EV",
        "price": 24_000.0,
        "energy": Electricity(consumption=15.0, home_price=0.30),
        "residual": GeometricDecline(0.15),
        "first_registration": date(2026, 1, 1),
    }
    defaults.update(kwargs)
    return Vehicle(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def tl() -> Timeline:
    return Timeline(horizon_years=6, periods_per_year=12, rate=0.03)


@pytest.fixture
def ctx(tl: Timeline) -> Context:
    return Context(timeline=tl, usage=Usage(annual_km=12_000.0), start=0, end=72)


# ------------------------------------------------------------------ the matrix


@pytest.mark.parametrize(
    ("income", "children", "expected"),
    [
        (30_000, 0, 5_000),
        (30_000, 1, 5_500),
        (30_000, 3, 6_000),
        (55_000, 0, 4_000),
        (55_000, 1, 4_500),
        (55_000, 2, 5_000),
        (70_000, 0, 3_000),
        (70_000, 2, 4_000),
        (82_000, 0, 0),
        (82_000, 1, 3_500),
        (87_000, 1, 0),
        (87_000, 2, 4_000),
        (95_000, 3, 0),
    ],
)
def test_the_income_by_children_matrix_matches_the_workbook(
    income: float, children: int, expected: float
) -> None:
    """Foerdermatrix rows 8-13, read through the MATCH/INDEX lookup."""
    assert BAFA2026().amount(Household(income, children)) == expected


def test_band_boundaries_fall_on_the_higher_grant() -> None:
    """At exactly 45,000 EUR the household is still in the lowest band."""
    assert BAFA2026().amount(Household(45_000.0, 0)) == 5_000
    assert BAFA2026().amount(Household(45_000.01, 0)) == 4_000


def test_more_than_two_children_does_not_add_more() -> None:
    assert BAFA2026().amount(Household(30_000.0, 5)) == BAFA2026().amount(Household(30_000.0, 2))


def test_the_plug_in_hybrid_schedule_is_separate() -> None:
    phev = BAFA2026(matrix=BAFA_2026_PHEV)
    assert phev.amount(Household(55_000.0, 0)) == 2_500


# ------------------------------------------------------------- eligibility


def test_a_used_vehicle_is_not_eligible(ctx: Context) -> None:
    household = Household(55_000.0, 0)
    assert not BAFA2026().is_eligible(ev(age_at_acquisition=3.0), household)


def test_light_quadricycles_are_not_eligible(ctx: Context) -> None:
    household = Household(55_000.0, 0)
    for category in (VehicleCategory.L6E, VehicleCategory.L7E, VehicleCategory.N1):
        assert not BAFA2026().is_eligible(ev(category=category), household)


def test_a_petrol_car_is_not_eligible() -> None:
    petrol = ev(energy=Petrol(consumption=6.0, price=2.10), propulsion=Propulsion.ICE)
    assert not BAFA2026().is_eligible(petrol, Household(55_000.0, 0))


def test_a_fuel_cell_vehicle_is_eligible() -> None:
    fcev = ev(energy=Hydrogen(consumption=1.0, price=13.85))
    assert fcev.drivetrain is Propulsion.FCEV
    assert BAFA2026().is_eligible(fcev, Household(55_000.0, 0))


def test_too_much_income_removes_eligibility() -> None:
    assert not BAFA2026().is_eligible(ev(), Household(200_000.0, 0))


def test_an_exhausted_budget_removes_eligibility() -> None:
    assert not BAFA2026(available=False).is_eligible(ev(), Household(55_000.0, 0))


def test_an_ineligible_incentive_produces_no_flows(ctx: Context) -> None:
    used = ev(age_at_acquisition=3.0)
    assert BAFA2026().flows(used, ctx).flows == ()


# ----------------------------------------------------------------- timing


def test_the_grant_lands_after_the_disbursement_lag(ctx: Context, tl: Timeline) -> None:
    series = BAFA2026(disbursement_lag_months=4).flows(
        ev(), Context(timeline=tl, household=Household(55_000.0, 0), start=0, end=72)
    )
    amounts = series.amounts(tl)
    assert amounts[4] == 4_000
    assert amounts[0] == 0


def test_the_lag_runs_from_the_leg_start_not_from_t0(tl: Timeline) -> None:
    """A chain's successor registers in month 24, so its grant arrives in month 28."""
    later = Context(timeline=tl, household=Household(55_000.0, 0), start=24, end=72)
    amounts = BAFA2026(disbursement_lag_months=4).flows(ev(), later).amounts(tl)
    assert amounts[28] == 4_000
    assert amounts[4] == 0


def test_a_grant_arriving_past_the_horizon_is_dropped(tl: Timeline) -> None:
    late = Context(timeline=tl, household=Household(55_000.0, 0), start=70, end=72)
    assert BAFA2026(disbursement_lag_months=6).flows(ev(), late).flows == ()


def test_a_negative_lag_is_rejected() -> None:
    with pytest.raises(CarbitrageError, match="disbursement_lag_months"):
        BAFA2026(disbursement_lag_months=-1)


# ------------------------------------------------------------- constraints


def test_the_minimum_holding_period_is_a_constraint_not_a_cash_flow(ctx: Context) -> None:
    incentive = BAFA2026()
    with_household = Context(
        timeline=ctx.timeline, household=Household(55_000.0, 0), start=0, end=72
    )
    constraints = incentive.constraints(ev(), with_household)
    assert len(constraints) == 1
    assert "36 months" in constraints[0]
    # It does not appear anywhere in the money.
    assert len(incentive.flows(ev(), with_household).flows) == 1


def test_an_ineligible_incentive_imposes_no_constraint(ctx: Context) -> None:
    assert BAFA2026().constraints(ev(age_at_acquisition=3.0), ctx) == ()


# -------------------------------------------------------------- thg quota


def test_the_quota_credit_falls_at_every_year_end(tl: Timeline, ctx: Context) -> None:
    amounts = ThgQuote(annual_amount=300.0).flows(ev(), ctx).amounts(tl)
    assert amounts.nonzero()[0].tolist() == [12, 24, 36, 48, 60, 72]
    assert amounts[12] == 300.0


def test_the_quota_credit_is_for_battery_electric_vehicles_only() -> None:
    petrol = ev(energy=Petrol(consumption=6.0, price=2.10))
    assert not ThgQuote(annual_amount=300.0).is_eligible(petrol, Household())
    hybrid = ev(
        energy=BivalentSource(
            Electricity(consumption=15.0, home_price=0.3), LPG(consumption=8.0, price=1.0), 0.5
        )
    )
    assert hybrid.drivetrain is Propulsion.PHEV
    assert not ThgQuote(annual_amount=300.0).is_eligible(hybrid, Household())


def test_a_used_electric_vehicle_still_earns_the_quota_credit(ctx: Context) -> None:
    """The credit follows the car, not the buyer, so a used BEV keeps it."""
    used = ev(age_at_acquisition=3.0)
    assert ThgQuote(annual_amount=300.0).is_eligible(used, Household())


def test_the_quota_credit_can_grow(tl: Timeline, ctx: Context) -> None:
    amounts = ThgQuote(annual_amount=300.0, growth=0.05).flows(ev(), ctx).amounts(tl)
    assert amounts[12] == pytest.approx(300.0)
    assert amounts[24] == pytest.approx(300.0 * 1.05)


def test_a_zero_credit_produces_nothing(ctx: Context) -> None:
    assert ThgQuote(annual_amount=0.0).flows(ev(), ctx).flows == ()


# ------------------------------------------------------- tax exemption


def test_the_exemption_credits_back_the_circulation_tax(tl: Timeline, ctx: Context) -> None:
    taxed = ev(annual_tax=180.0)
    amounts = VehicleTaxExemption(max_years=10.0).flows(taxed, ctx).amounts(tl)
    assert amounts[1] == pytest.approx(15.0)  # 180 / 12
    assert amounts.sum() == pytest.approx(180.0 * 6)


def test_the_exemption_leaves_the_underlying_cost_visible(ctx: Context) -> None:
    """Crediting back is not the same as zeroing: both lines stay in the breakdown."""
    taxed = ev(annual_tax=180.0)
    series = VehicleTaxExemption().flows(taxed, ctx)
    assert len(series.flows) == 1
    assert "exemption" in series.flows[0].description


def test_a_used_vehicle_inherits_only_the_unexpired_remainder(ctx: Context) -> None:
    """The exemption is keyed on first registration, not on the purchase date."""
    exemption = VehicleTaxExemption(max_years=10.0, expires=None)
    three_year_old = ev(annual_tax=180.0, age_at_acquisition=3.0)
    assert exemption.remaining_years(three_year_old, ctx) == pytest.approx(7.0)


def test_an_expired_exemption_credits_nothing(ctx: Context) -> None:
    exemption = VehicleTaxExemption(max_years=10.0, expires=None)
    old = ev(annual_tax=180.0, age_at_acquisition=12.0)
    assert exemption.remaining_years(old, ctx) == 0.0
    assert exemption.flows(old, ctx).flows == ()


def test_the_statutory_end_date_caps_the_exemption(tl: Timeline, ctx: Context) -> None:
    """Even a fresh registration cannot be exempt past the scheme's end."""
    exemption = VehicleTaxExemption(max_years=10.0, expires=date(2030, 1, 1))
    car = ev(annual_tax=180.0, first_registration=date(2026, 1, 1))
    assert exemption.remaining_years(car, ctx) == pytest.approx(4.0, abs=0.02)


def test_the_exemption_never_outlasts_the_holding_period(tl: Timeline) -> None:
    short = Context(timeline=tl, start=0, end=24)
    exemption = VehicleTaxExemption(max_years=10.0, expires=None)
    amounts = exemption.flows(ev(annual_tax=180.0), short).amounts(tl)
    assert amounts.nonzero()[0].max() == 24


def test_a_vehicle_with_no_tax_needs_no_exemption(ctx: Context) -> None:
    assert VehicleTaxExemption().flows(ev(annual_tax=0.0), ctx).flows == ()


def test_a_petrol_car_gets_no_exemption(ctx: Context) -> None:
    petrol = ev(annual_tax=180.0, energy=Petrol(consumption=6.0, price=2.10))
    assert not VehicleTaxExemption().is_eligible(petrol, Household())
