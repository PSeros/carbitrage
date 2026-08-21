"""Residual value models.  They take an age, which is what makes chains consistent."""

import pytest

from carbitrage.errors import CarbitrageError
from carbitrage.residual import (
    FirstYearDropThenGeometric,
    GeometricDecline,
    TabulatedResiduals,
)


def test_geometric_decline_matches_workbook_ev_residual() -> None:
    # Annahmen C38: 23,900 EUR at 15 % p.a. over 6 years.
    assert GeometricDecline(0.15).value(23_900, 6) == pytest.approx(9013.8734234375)


def test_geometric_decline_matches_workbook_ice_residual() -> None:
    # Annahmen C60: 22,000 EUR at 13 % p.a. over 6 years.
    assert GeometricDecline(0.13).value(22_000, 6) == pytest.approx(9539.776422198)


def test_geometric_decline_at_age_zero_is_the_full_price() -> None:
    assert GeometricDecline(0.15).value(23_900, 0) == pytest.approx(23_900)


def test_successor_in_a_chain_is_valued_at_its_own_age() -> None:
    # Annahmen C82: the replacement bought after 2 years is 4 years old at the
    # 6-year horizon, and is worth more than the vehicle bought at t=0.
    price_then = 23_900 * 1.015**2
    model = GeometricDecline(0.15)
    assert model.value(price_then, 4) == pytest.approx(12853.0349448594)
    assert model.value(price_then, 4) > model.value(23_900, 6)


def test_first_year_drop_is_steeper_than_a_constant_rate() -> None:
    steep = FirstYearDropThenGeometric(drop=0.20, rate=0.13)
    flat = GeometricDecline(0.13)
    assert steep.value(23_900, 1) < flat.value(23_900, 1)


def test_first_year_drop_prorates_within_the_first_year() -> None:
    model = FirstYearDropThenGeometric(drop=0.20, rate=0.13)
    assert model.value(1000, 0) == pytest.approx(1000)
    assert model.value(1000, 0.5) == pytest.approx(900)
    assert model.value(1000, 1) == pytest.approx(800)
    assert model.value(1000, 2) == pytest.approx(800 * 0.87)


def test_retained_is_the_price_free_fraction() -> None:
    assert GeometricDecline(0.15).retained(6) == pytest.approx(0.85**6)


def test_tabulated_interpolates_between_points() -> None:
    curve = TabulatedResiduals({0: 1.0, 1: 0.75, 3: 0.55})
    assert curve.value(20_000, 1) == pytest.approx(15_000)
    assert curve.value(20_000, 2) == pytest.approx(20_000 * 0.65)


def test_tabulated_extrapolates_along_the_final_segment() -> None:
    curve = TabulatedResiduals({0: 1.0, 2: 0.8})
    # Continues at the final slope rather than freezing at 0.8.
    assert curve.value(10_000, 3) == pytest.approx(7_000)


def test_tabulated_never_goes_negative() -> None:
    curve = TabulatedResiduals({0: 1.0, 1: 0.5})
    assert curve.value(10_000, 10) == 0.0


def test_from_values_builds_a_curve_from_a_dealer_quote() -> None:
    # The incumbent: worth 1,500 EUR today, 800 EUR in two years.
    curve = TabulatedResiduals.from_values(1_500, {2: 800})
    assert curve.value(1_500, 0) == pytest.approx(1_500)
    assert curve.value(1_500, 2) == pytest.approx(800)
    assert curve.value(1_500, 1) == pytest.approx(1_150)


@pytest.mark.parametrize("rate", [-0.1, 1.0, 1.5])
def test_geometric_rate_is_validated(rate: float) -> None:
    with pytest.raises(CarbitrageError, match="rate must lie"):
        GeometricDecline(rate)


def test_negative_age_is_rejected() -> None:
    with pytest.raises(CarbitrageError, match="age_years must not be negative"):
        GeometricDecline(0.15).value(1_000, -1)


def test_a_table_needs_at_least_two_points() -> None:
    with pytest.raises(CarbitrageError, match="at least two points"):
        TabulatedResiduals({0: 1.0})


def test_value_after_rescales_the_curve_to_a_used_purchase_price() -> None:
    # Annahmen C91: a 3-year-old EV bought for 18,000 EUR, 11 % p.a. from purchase.
    curve = GeometricDecline(0.11)
    assert curve.value_after(18_000, years_held=6, age_at_acquisition=3) == pytest.approx(
        8945.663237298
    )


def test_value_after_on_a_new_asset_is_plain_value() -> None:
    curve = GeometricDecline(0.15)
    assert curve.value_after(23_900, years_held=6) == pytest.approx(curve.value(23_900, 6))


def test_value_after_places_a_used_price_on_the_curve_at_its_own_age() -> None:
    curve = FirstYearDropThenGeometric(drop=0.20, rate=0.13)
    # Bought at age 1 for exactly what the curve says a 1-year-old is worth,
    # the rescaled curve must coincide with the original.
    price_at_one = curve.value(30_000, 1)
    assert curve.value_after(price_at_one, years_held=2, age_at_acquisition=1) == pytest.approx(
        curve.value(30_000, 3)
    )


def test_value_after_refuses_a_worthless_acquisition_age() -> None:
    curve = TabulatedResiduals({0: 1.0, 1: 0.0})
    with pytest.raises(CarbitrageError, match="retains nothing at age"):
        curve.value_after(5_000, years_held=1, age_at_acquisition=2)
