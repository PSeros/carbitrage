"""The timeline is the module everything else resolves against, so it is tested hardest.

Constants come from the reference workbook's `Annahmen` and `Ergebnis` sheets.
"""

import math

import pytest

from carbitrage.errors import InconsistentRateBasisError, TimelineError
from carbitrage.timeline import (
    Escalation,
    Periodisation,
    RateBasis,
    Timeline,
    fisher_inflation,
    fisher_nominal,
    fisher_real,
)

WORKBOOK_MONTHLY_RATE = 0.00246626977230369
WORKBOOK_CRF = 0.184597500450177
WORKBOOK_ANNUITY_FACTOR = 5.41719144387819


@pytest.fixture
def base() -> Timeline:
    return Timeline(horizon_years=6, periods_per_year=12, rate=0.03, energy_escalation=0.02)


def test_conformal_monthly_rate_matches_workbook(base: Timeline) -> None:
    assert base.periodic_rate == pytest.approx(WORKBOOK_MONTHLY_RATE, abs=1e-15)


def test_conformal_rate_compounds_back_to_the_annual_rate(base: Timeline) -> None:
    assert (1 + base.periodic_rate) ** 12 == pytest.approx(1.03, abs=1e-13)


def test_proportional_periodisation_must_be_opted_into() -> None:
    tl = Timeline(horizon_years=6, rate=0.03, periodisation=Periodisation.PROPORTIONAL)
    assert tl.periodic_rate == pytest.approx(0.03 / 12)
    assert (1 + tl.periodic_rate) ** 12 > 1.03  # overstates, which is why it is not the default


def test_capital_recovery_factor_matches_workbook(base: Timeline) -> None:
    assert base.capital_recovery_factor() == pytest.approx(WORKBOOK_CRF, abs=1e-12)


def test_annuity_factor_matches_workbook(base: Timeline) -> None:
    assert base.annuity_factor() == pytest.approx(WORKBOOK_ANNUITY_FACTOR, abs=1e-11)


def test_crf_and_annuity_factor_are_reciprocal(base: Timeline) -> None:
    assert base.capital_recovery_factor() * base.annuity_factor() == pytest.approx(1.0)


def test_zero_rate_degenerates_gracefully() -> None:
    tl = Timeline(horizon_years=5, rate=0.0)
    assert tl.annuity_factor() == 5.0
    assert tl.capital_recovery_factor() == pytest.approx(0.2)
    assert tl.discount_factor(tl.n_periods) == 1.0


def test_discount_factor_at_horizon_matches_annual_compounding(base: Timeline) -> None:
    # 72 conformal monthly periods must equal 6 annual periods exactly.
    assert base.discount_factor(72) == pytest.approx(1 / 1.03**6, abs=1e-13)


def test_escalation_holds_the_whole_first_year_then_steps(base: Timeline) -> None:
    assert base.escalation(0, 0.02) == 1.0
    assert base.escalation(1, 0.02) == 1.0
    assert base.escalation(12, 0.02) == 1.0  # month 12 is still a first-year payment
    assert base.escalation(13, 0.02) == pytest.approx(1.02)
    assert base.escalation(24, 0.02) == pytest.approx(1.02)
    assert base.escalation(25, 0.02) == pytest.approx(1.02**2)


def test_continuous_escalation_compounds_every_period() -> None:
    tl = Timeline(horizon_years=2, escalation_mode=Escalation.CONTINUOUS)
    assert tl.escalation(6, 0.02) == pytest.approx(1.02**0.5)


def test_escalation_index_is_vectorised_escalation(base: Timeline) -> None:
    index = base.escalation_index(0.02)
    assert [index[t] for t in (0, 1, 12, 13, 72)] == pytest.approx(
        [base.escalation(t, 0.02) for t in (0, 1, 12, 13, 72)]
    )


def test_index_of_converts_years_to_periods(base: Timeline) -> None:
    assert base.index_of(0) == 0
    assert base.index_of(2.0) == 24
    assert base.index_of(6) == 72


def test_index_of_rejects_non_boundaries(base: Timeline) -> None:
    with pytest.raises(TimelineError, match="not a period boundary"):
        base.index_of(0.5 / 12)


def test_index_of_rejects_out_of_horizon(base: Timeline) -> None:
    with pytest.raises(TimelineError, match="outside the horizon"):
        base.index_of(7)


def test_period_bounds_are_checked(base: Timeline) -> None:
    with pytest.raises(TimelineError, match=r"outside 0\.\.72"):
        base.discount_factor(73)


@pytest.mark.parametrize("horizon", [0, -1])
def test_non_positive_horizon_is_rejected(horizon: float) -> None:
    with pytest.raises(TimelineError, match="horizon_years must be positive"):
        Timeline(horizon_years=horizon)


def test_nominal_rate_with_inflation_but_no_escalation_is_rejected() -> None:
    # The classic real/nominal mismatch: flows in today's money, rate not.
    with pytest.raises(InconsistentRateBasisError, match="today's money"):
        Timeline(horizon_years=6, rate=0.05, inflation=0.02)


def test_real_basis_requires_stated_inflation() -> None:
    with pytest.raises(InconsistentRateBasisError, match="needs an explicit"):
        Timeline(horizon_years=6, rate=0.01, basis=RateBasis.REAL)


def test_real_timeline_constructor_is_accepted() -> None:
    tl = Timeline.real(6, rate=0.0098, inflation=0.02)
    assert tl.basis is RateBasis.REAL


def test_fisher_round_trips() -> None:
    nominal = fisher_nominal(0.01, 0.02)
    assert nominal == pytest.approx(0.0302)
    assert fisher_real(nominal, 0.02) == pytest.approx(0.01)
    assert fisher_inflation(nominal, 0.01) == pytest.approx(0.02)


def test_to_nominal_uses_exact_fisher_not_addition() -> None:
    real = Timeline.real(6, rate=0.01, inflation=0.02, energy_escalation=0.005)
    nominal = real.to_nominal()
    assert nominal.basis is RateBasis.NOMINAL
    assert nominal.rate == pytest.approx(0.0302)  # not 0.03
    assert nominal.energy_escalation == pytest.approx(1.005 * 1.02 - 1)


def test_to_nominal_is_identity_for_a_nominal_timeline(base: Timeline) -> None:
    assert base.to_nominal() is base


def test_pv_rejects_a_mismatched_vector(base: Timeline) -> None:
    with pytest.raises(TimelineError, match="does not match a timeline"):
        base.pv(Timeline(horizon_years=2).zeros())


def test_pv_of_a_unit_at_t_equals_its_discount_factor(base: Timeline) -> None:
    amounts = base.zeros()
    amounts[24] = 1.0
    assert base.pv(amounts) == pytest.approx(base.discount_factor(24))


def test_year_index_groups_payments_into_policy_years(base: Timeline) -> None:
    assert [base.year_index(t) for t in (0, 1, 12, 13, 24, 25)] == [0, 0, 0, 1, 1, 2]


def test_years_at_inverts_index_of(base: Timeline) -> None:
    assert math.isclose(base.years_at(base.index_of(3.0)), 3.0)
