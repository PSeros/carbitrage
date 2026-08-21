"""Property-based tests.

These assert the invariants that must hold for *any* inputs, not just the
workbook's.  Where the golden tests pin one known answer, these pin the shape of
the arithmetic.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from carbitrage import (
    Alternative,
    GeometricDecline,
    ReplacementChain,
    Timeline,
    Usage,
    Vehicle,
    compare,
)
from carbitrage.core.cashflow import CashFlowSeries, Component, Frequency, OneOff, Recurring
from carbitrage.domain.energy import Electricity, Petrol

SLOW = settings(max_examples=25, deadline=None)

rates = st.floats(min_value=0.0, max_value=0.25, allow_nan=False, allow_infinity=False)
positive_rates = st.floats(min_value=0.001, max_value=0.25)
horizons = st.integers(min_value=1, max_value=12)
money = st.floats(min_value=0.0, max_value=50_000, allow_nan=False, allow_infinity=False)
scales = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)


def _vehicle(price: float = 20_000.0, **kwargs: object) -> Vehicle:
    defaults: dict[str, object] = {
        "name": "test asset",
        "price": price,
        "energy": Electricity(consumption=15.0, home_price=0.30),
        "residual": GeometricDecline(0.15),
        "insurance": 600.0,
        "maintenance": 400.0,
    }
    defaults.update(kwargs)
    return Vehicle(**defaults)  # type: ignore[arg-type]


# ------------------------------------------------------------------- timeline


@given(horizon=horizons, per_year=st.sampled_from([1, 2, 4, 12]), rate=rates)
def test_conformal_rate_always_compounds_back(horizon: int, per_year: int, rate: float) -> None:
    tl = Timeline(horizon_years=horizon, periods_per_year=per_year, rate=rate)
    assert (1 + tl.periodic_rate) ** per_year == pytest.approx(1 + rate, rel=1e-12)


@given(horizon=horizons, rate=positive_rates)
def test_crf_and_annuity_factor_are_always_reciprocal(horizon: int, rate: float) -> None:
    tl = Timeline(horizon_years=horizon, rate=rate)
    assert tl.capital_recovery_factor() * tl.annuity_factor() == pytest.approx(1.0, rel=1e-12)


@given(horizon=horizons, rate=positive_rates, amount=money)
def test_eac_times_the_annuity_factor_round_trips_to_pv(
    horizon: int, rate: float, amount: float
) -> None:
    """EAC is an affine transform of PV, so the round trip must be exact."""
    tl = Timeline(horizon_years=horizon, rate=rate)
    pv = amount
    eac = pv * tl.capital_recovery_factor()
    assert eac * tl.annuity_factor() == pytest.approx(pv, rel=1e-9, abs=1e-9)


# ------------------------------------------------------------------ cashflow


@given(one_off=money, monthly=money, horizon=horizons)
def test_a_zero_rate_makes_pv_the_undiscounted_sum(
    one_off: float, monthly: float, horizon: int
) -> None:
    tl = Timeline(horizon_years=horizon, periods_per_year=12, rate=0.0)
    series = CashFlowSeries.of(
        OneOff(-one_off, at=0),
        Recurring(-monthly, Frequency.MONTHLY),
        Recurring(monthly / 2, Frequency.ANNUAL),
    )
    assert series.pv(tl) == pytest.approx(series.undiscounted(tl), rel=1e-9, abs=1e-9)


@given(rate=rates, horizon=horizons, monthly=st.floats(min_value=1.0, max_value=2_000))
def test_the_breakdown_always_sums_to_the_total(
    rate: float, horizon: int, monthly: float
) -> None:
    tl = Timeline(horizon_years=horizon, periods_per_year=12, rate=rate)
    series = CashFlowSeries.of(
        OneOff(-10_000.0, at=0, label=Component.ACQUISITION),
        Recurring(-monthly, Frequency.MONTHLY, label=Component.ENERGY),
        Recurring(300.0, Frequency.ANNUAL, label=Component.SUBSIDY),
    )
    assert sum(series.breakdown(tl).values()) == pytest.approx(series.pv(tl), rel=1e-9, abs=1e-6)


@given(low=positive_rates, bump=st.floats(min_value=0.001, max_value=0.2), horizon=horizons)
def test_raising_the_discount_rate_never_raises_the_pv_of_a_cost_stream(
    low: float, bump: float, horizon: int
) -> None:
    """A pure cost stream is all negative, so discounting harder brings it toward zero."""
    series = CashFlowSeries.of(Recurring(-500.0, Frequency.MONTHLY))
    cheap = Timeline(horizon_years=horizon, periods_per_year=12, rate=low)
    dear = Timeline(horizon_years=horizon, periods_per_year=12, rate=low + bump)
    assert series.pv(dear) >= series.pv(cheap) - 1e-9


# --------------------------------------------------------------- comparison


@given(scale=scales)
@SLOW
def test_scaling_every_monetary_input_scales_every_pv(scale: float) -> None:
    """The model is homogeneous of degree one in money."""
    tl = Timeline(horizon_years=4, periods_per_year=12, rate=0.03)
    usage = Usage(annual_km=12_000.0)

    def build(k: float) -> Alternative:
        return Alternative(
            vehicle=_vehicle(
                price=20_000.0 * k,
                energy=Electricity(consumption=15.0, home_price=0.30 * k),
                insurance=600.0 * k,
                maintenance=400.0 * k,
            ),
            label="scaled",
        )

    base = compare([build(1.0)], tl, usage=usage)["scaled"].npv
    scaled = compare([build(scale)], tl, usage=usage)["scaled"].npv
    assert scaled == pytest.approx(base * scale, rel=1e-9)


@given(horizon=horizons, rate=positive_rates)
@SLOW
def test_identical_alternatives_have_identical_eac(horizon: int, rate: float) -> None:
    tl = Timeline(horizon_years=horizon, periods_per_year=12, rate=rate)
    usage = Usage(annual_km=10_000.0)
    a = Alternative(vehicle=_vehicle(), label="a")
    b = Alternative(vehicle=_vehicle(), label="b")
    result = compare([a, b], tl, usage=usage)
    assert result["a"].eac() == pytest.approx(result["b"].eac(), rel=1e-12)
    assert result.margin() == pytest.approx(0.0, abs=1e-9)
    assert not result.is_material()


@given(rate=positive_rates)
@SLOW
def test_a_self_chain_has_the_same_eac_as_the_single_alternative(rate: float) -> None:
    """Two identical three-year holdings back to back annualise to one of them.

    This only holds when the repetitions really are identical, which is why the
    chain must be built without price escalation and with a residual curve that
    depends on age alone.
    """
    single_tl = Timeline(horizon_years=3, periods_per_year=12, rate=rate)
    double_tl = Timeline(horizon_years=6, periods_per_year=12, rate=rate)
    usage = Usage(annual_km=10_000.0)

    single = Alternative(vehicle=_vehicle(), label="one holding")
    first = Alternative(vehicle=_vehicle(), life_years=3, label="first")
    second = Alternative(vehicle=_vehicle(), label="second")
    chain = ReplacementChain(first, second, label="two holdings", escalate_successor_price=False)

    once = compare([single], single_tl, usage=usage)["one holding"].eac()
    twice = compare([chain], double_tl, usage=usage)["two holdings"].eac()
    assert twice == pytest.approx(once, rel=1e-9)


@given(km=st.floats(min_value=1_000, max_value=40_000))
@SLOW
def test_more_mileage_never_reduces_the_cost_of_a_cost_only_alternative(km: float) -> None:
    tl = Timeline(horizon_years=5, periods_per_year=12, rate=0.03)
    alt = Alternative(vehicle=_vehicle(), label="a")
    base = compare([alt], tl, usage=Usage(annual_km=1_000.0))["a"].npv
    more = compare([alt], tl, usage=Usage(annual_km=km))["a"].npv
    assert more <= base + 1e-9


@given(price=st.floats(min_value=5_000, max_value=60_000))
@SLOW
def test_cost_per_km_is_eac_divided_by_mileage(price: float) -> None:
    tl = Timeline(horizon_years=5, periods_per_year=12, rate=0.03)
    usage = Usage(annual_km=15_000.0)
    result = compare([Alternative(vehicle=_vehicle(price=price), label="a")], tl, usage=usage)
    assert result["a"].cost_per_km() == pytest.approx(result["a"].eac() / 15_000.0, rel=1e-12)


@given(rate=positive_rates, horizon=horizons)
@SLOW
def test_a_dominated_alternative_never_wins(rate: float, horizon: int) -> None:
    """Strictly worse on every input must mean strictly worse in the ranking."""
    tl = Timeline(horizon_years=horizon, periods_per_year=12, rate=rate)
    usage = Usage(annual_km=12_000.0)
    good = Alternative(vehicle=_vehicle(price=20_000.0), label="good")
    worse = Alternative(
        vehicle=_vehicle(price=25_000.0, insurance=900.0, maintenance=700.0), label="worse"
    )
    result = compare([good, worse], tl, usage=usage)
    assert result.best().name == "good"


# ------------------------------------------------------------------ residual


@given(
    price=st.floats(min_value=1_000, max_value=80_000),
    rate=st.floats(min_value=0.01, max_value=0.4),
    age=st.floats(min_value=0.0, max_value=15.0),
)
def test_a_residual_value_is_never_negative_and_never_exceeds_the_price(
    price: float, rate: float, age: float
) -> None:
    value = GeometricDecline(rate).value(price, age)
    assert 0.0 <= value <= price + 1e-9


@given(
    price=st.floats(min_value=1_000, max_value=80_000),
    rate=st.floats(min_value=0.01, max_value=0.4),
    younger=st.floats(min_value=0.0, max_value=10.0),
    older_by=st.floats(min_value=0.01, max_value=5.0),
)
def test_older_assets_are_never_worth_more(
    price: float, rate: float, younger: float, older_by: float
) -> None:
    curve = GeometricDecline(rate)
    assert curve.value(price, younger + older_by) <= curve.value(price, younger) + 1e-9


@given(
    rate=st.floats(min_value=0.01, max_value=0.4),
    age_at_acquisition=st.floats(min_value=0.0, max_value=8.0),
    held=st.floats(min_value=0.0, max_value=8.0),
)
def test_value_after_is_consistent_with_the_underlying_curve(
    rate: float, age_at_acquisition: float, held: float
) -> None:
    """Buying at the curve's own price must leave the curve unchanged."""
    curve = GeometricDecline(rate)
    as_new = 30_000.0
    market_price = curve.value(as_new, age_at_acquisition)
    assume(market_price > 1.0)
    rescaled = curve.value_after(
        market_price, years_held=held, age_at_acquisition=age_at_acquisition
    )
    assert rescaled == pytest.approx(curve.value(as_new, age_at_acquisition + held), rel=1e-9)


# --------------------------------------------------------------------- energy


@given(
    share=st.floats(min_value=0.0, max_value=1.0),
    gas=st.floats(min_value=0.3, max_value=3.0),
    petrol=st.floats(min_value=0.5, max_value=4.0),
)
def test_a_bivalent_cost_lies_between_its_two_legs(
    share: float, gas: float, petrol: float
) -> None:
    from carbitrage.domain.energy import LPG, BivalentSource

    tl = Timeline(horizon_years=2, periods_per_year=12, rate=0.03)
    primary = LPG(consumption=8.0, price=gas)
    secondary = Petrol(consumption=8.0, price=petrol)
    mixed = BivalentSource(primary, secondary, primary_share=share)
    lo = min(primary.cost_per_100km(1, tl), secondary.cost_per_100km(1, tl))
    hi = max(primary.cost_per_100km(1, tl), secondary.cost_per_100km(1, tl))
    assert lo - 1e-9 <= mixed.cost_per_100km(1, tl) <= hi + 1e-9


# ----------------------------------------------------------------- workbook
