"""The mistakes this library exists to prevent.

Each of these is a way of producing a plausible-looking but wrong answer.  They
are gathered here, separately from the tests of the features they belong to, so
that the contract is readable in one place.
"""

from __future__ import annotations

import pytest

from carbitrage import Alternative, Timeline, Vehicle, compare
from carbitrage.energy import Electricity
from carbitrage.errors import InconsistentRateBasisError
from carbitrage.rates import RateBasis, fisher_real
from carbitrage.residual import GeometricDecline

# 1 --------------------------------------------- unequal lives must not be compared


# 2 ------------------------------------------------ subsidy double counting must warn


# 3 ------------------------------- real and nominal must not be mixed, caught early


def test_a_nominal_rate_with_inflation_but_unescalated_flows_raises() -> None:
    """Flows in today's money discounted at a nominal rate understates every PV."""
    with pytest.raises(InconsistentRateBasisError, match="today's money"):
        Timeline(horizon_years=6, rate=0.05, inflation=0.02)


def test_a_real_basis_without_a_stated_inflation_raises() -> None:
    with pytest.raises(InconsistentRateBasisError, match="needs an explicit"):
        Timeline(horizon_years=6, rate=0.01, basis=RateBasis.REAL)


def test_the_error_names_the_fisher_remedy() -> None:
    with pytest.raises(InconsistentRateBasisError) as excinfo:
        Timeline(horizon_years=6, rate=0.05, inflation=0.02)
    assert "Timeline.real" in str(excinfo.value)


def test_both_consistent_framings_give_the_same_answer() -> None:
    """Real flows at a real rate, or nominal flows at a nominal rate: same PV.

    This is the invariant the guard above protects.  Discounting a stream that
    escalates at 2 % with a 5.06 % nominal rate must equal discounting the
    unescalated stream at the 3 % real rate the Fisher relation implies, since
    1.03 x 1.02 = 1.0506 exactly.

    The asset carries no terminal value here, deliberately: a residual is quoted
    in money of the day and so belongs to neither framing automatically.  See
    :func:`test_a_residual_value_is_nominal_by_construction`.
    """
    inflation = 0.02
    nominal_rate = 0.0506
    real_rate = fisher_real(nominal_rate, inflation)
    assert real_rate == pytest.approx(0.03, abs=1e-12)

    def build(timeline: Timeline) -> float:
        alternative = Alternative(
            vehicle=Vehicle(
                name="asset",
                price=0.0,
                energy=Electricity(consumption=0.0, home_price=0.0),
                residual=GeometricDecline(0.0),
                insurance=1_000.0,
            ),
            label="a",
        )
        return compare([alternative], timeline)["a"].npv

    nominal = build(
        Timeline(horizon_years=5, periods_per_year=12, rate=nominal_rate, cost_escalation=inflation)
    )
    real = build(Timeline.real(5, rate=real_rate, inflation=inflation, periods_per_year=12))
    # The escalation index steps once a year rather than compounding monthly, so
    # the two agree to within that within-year timing difference.
    assert nominal == pytest.approx(real, rel=0.02)


def test_a_residual_value_is_nominal_by_construction() -> None:
    """A depreciation rate read off used-car prices is a *nominal* rate of decline.

    Applying it to today's nominal price therefore yields a future nominal
    price, which is correctly discounted at a nominal rate — this is what the
    reference workbook does.  Working in real terms means restating the decline
    rate in real terms too; the library cannot detect which one was supplied, so
    it is documented rather than guessed at.
    """
    price = 20_000.0
    nominal_decline = 0.15
    inflation = 0.02
    curve = GeometricDecline(nominal_decline)

    nominal_value_in_5y = curve.value(price, 5)
    assert nominal_value_in_5y == pytest.approx(price * 0.85**5)

    # The same asset, stated in today's money, declines faster in real terms.
    real_decline = 1.0 - (1.0 - nominal_decline) / (1.0 + inflation)
    real_value_in_5y = GeometricDecline(real_decline).value(price, 5)
    assert real_value_in_5y < nominal_value_in_5y
    assert real_value_in_5y == pytest.approx(nominal_value_in_5y / (1 + inflation) ** 5)


def test_converting_a_real_timeline_uses_exact_fisher() -> None:
    real = Timeline.real(6, rate=0.01, inflation=0.02, energy_escalation=0.0)
    assert real.to_nominal().rate == pytest.approx(1.01 * 1.02 - 1)


# 4 ------------------------------------------- ranking must never use absolute value


# 5 --------------------------------------------- a breakdown that does not sum is a bug


# 6 ------------------------------------- an unreliable IRR is withheld, not guessed


# 7 -------------------------------- a missing switch point is reported, not invented
