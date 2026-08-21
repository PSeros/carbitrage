"""Tax treatments: the private default, and a business case proving the abstraction."""

from __future__ import annotations

import pytest

from carbitrage import (
    Alternative,
    BusinessAssets,
    Component,
    Context,
    PrivateHousehold,
    Purchase,
    Timeline,
    Usage,
    Vehicle,
    compare,
)
from carbitrage.domain.energy import Electricity
from carbitrage.domain.residual import GeometricDecline
from carbitrage.domain.tax import DEGRESSIVE_2026, STRAIGHT_LINE_6Y
from carbitrage.errors import CarbitrageError


@pytest.fixture
def tl() -> Timeline:
    return Timeline(horizon_years=6, periods_per_year=12, rate=0.03)


@pytest.fixture
def car() -> Vehicle:
    return Vehicle(
        name="fleet EV",
        price=35_700.0,
        energy=Electricity(consumption=16.0, home_price=0.30),
        residual=GeometricDecline(0.15),
        insurance=800.0,
        maintenance=500.0,
    )


def test_a_private_household_changes_nothing(tl: Timeline, car: Vehicle) -> None:
    """The gross flows already tell the whole story for a private buyer."""
    ctx = Context(timeline=tl, usage=Usage(annual_km=15_000.0), start=0, end=72)
    series = Alternative(vehicle=car).flows(ctx)
    assert PrivateHousehold().adjust(series, car, ctx) is series


def test_a_business_recovers_input_vat(tl: Timeline, car: Vehicle) -> None:
    usage = Usage(annual_km=15_000.0)
    alt = Alternative(vehicle=car, acquisition=Purchase(), label="fleet")
    private = compare([alt], tl, usage=usage)["fleet"].npv
    business = compare(
        [alt], tl, usage=usage, tax=BusinessAssets(vat_rate=0.19, marginal_tax_rate=0.0)
    )["fleet"].npv
    assert business > private  # recovering VAT is worth money


def test_vat_is_recovered_on_the_purchase_price(tl: Timeline, car: Vehicle) -> None:
    usage = Usage(annual_km=0.0)
    alt = Alternative(vehicle=car, acquisition=Purchase(), label="fleet")
    result = compare(
        [alt], tl, usage=usage, tax=BusinessAssets(vat_rate=0.19, marginal_tax_rate=0.0)
    )
    recovered = next(
        line for line in result.detail("fleet") if line.description == "Input VAT"
    )
    assert recovered.pv == pytest.approx(35_700.0 * 0.19 / 1.19, abs=1e-6)


def test_output_vat_is_remitted_on_the_residual(tl: Timeline, car: Vehicle) -> None:
    alt = Alternative(vehicle=car, acquisition=Purchase(), label="fleet")
    result = compare(
        [alt],
        tl,
        usage=Usage(annual_km=0.0),
        tax=BusinessAssets(vat_rate=0.19, marginal_tax_rate=0.0),
    )
    remitted = [
        line
        for line in result.detail("fleet")
        if line.description == "Output VAT on proceeds"
    ]
    assert remitted
    assert all(line.pv < 0 for line in remitted)


def test_insurance_carries_no_vat(tl: Timeline, car: Vehicle) -> None:
    """Insurance premiums are VAT-exempt in Germany, so nothing is recoverable."""
    only_insurance = Vehicle(
        name="insurance only",
        price=0.0,
        energy=Electricity(consumption=0.0, home_price=0.0),
        residual=GeometricDecline(0.0),
        insurance=1_200.0,
    )
    alt = Alternative(vehicle=only_insurance, label="ins")
    result = compare(
        [alt],
        tl,
        usage=Usage(annual_km=0.0),
        tax=BusinessAssets(vat_rate=0.19, marginal_tax_rate=0.0),
    )
    assert not [line for line in result.detail("ins") if "VAT" in line.description]


def test_depreciation_shelters_taxable_profit(tl: Timeline, car: Vehicle) -> None:
    alt = Alternative(vehicle=car, acquisition=Purchase(), label="fleet")
    result = compare(
        [alt],
        tl,
        usage=Usage(annual_km=0.0),
        tax=BusinessAssets(vat_rate=0.0, marginal_tax_rate=0.30, depreciation=STRAIGHT_LINE_6Y),
    )
    shields = [line for line in result.detail("fleet") if "Depreciation shield" in line.description]
    assert len(shields) == 6
    assert all(line.pv > 0 for line in shields)


def test_the_accelerated_schedule_front_loads_the_shield(tl: Timeline, car: Vehicle) -> None:
    """75 % in year one is worth more than 1/6 a year, because it arrives sooner."""
    alt = Alternative(vehicle=car, acquisition=Purchase(), label="fleet")
    usage = Usage(annual_km=0.0)
    straight = compare(
        [alt], tl, usage=usage,
        tax=BusinessAssets(vat_rate=0.0, marginal_tax_rate=0.30, depreciation=STRAIGHT_LINE_6Y),
    )["fleet"].npv
    accelerated = compare(
        [alt], tl, usage=usage,
        tax=BusinessAssets(vat_rate=0.0, marginal_tax_rate=0.30, depreciation=DEGRESSIVE_2026),
    )["fleet"].npv
    assert accelerated > straight


def test_a_shield_beyond_the_horizon_is_not_claimed(car: Vehicle) -> None:
    short = Timeline(horizon_years=2, periods_per_year=12, rate=0.03)
    alt = Alternative(vehicle=car, acquisition=Purchase(), label="fleet")
    result = compare(
        [alt], short, usage=Usage(annual_km=0.0),
        tax=BusinessAssets(vat_rate=0.0, marginal_tax_rate=0.30),
    )
    shields = [line for line in result.detail("fleet") if "Depreciation shield" in line.description]
    assert len(shields) == 2


def test_private_use_is_neither_deductible_nor_recoverable(tl: Timeline, car: Vehicle) -> None:
    alt = Alternative(vehicle=car, acquisition=Purchase(), label="fleet")
    usage = Usage(annual_km=15_000.0)
    full = compare([alt], tl, usage=usage, tax=BusinessAssets(business_share=1.0))["fleet"].npv
    half = compare([alt], tl, usage=usage, tax=BusinessAssets(business_share=0.5))["fleet"].npv
    private = compare([alt], tl, usage=usage)["fleet"].npv
    assert private < half < full


def test_the_breakdown_still_sums_under_a_business_treatment(
    tl: Timeline, car: Vehicle
) -> None:
    alt = Alternative(vehicle=car, acquisition=Purchase(), label="fleet")
    result = compare([alt], tl, usage=Usage(annual_km=15_000.0), tax=BusinessAssets())
    assert sum(result.breakdown("fleet").values()) == pytest.approx(
        result["fleet"].npv, abs=1e-6
    )


def test_tax_adjustments_are_labelled_as_tax(tl: Timeline, car: Vehicle) -> None:
    alt = Alternative(vehicle=car, acquisition=Purchase(), label="fleet")
    result = compare([alt], tl, usage=Usage(annual_km=15_000.0), tax=BusinessAssets())
    assert Component.TAX in result.breakdown("fleet")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"vat_rate": -0.1}, "vat_rate must not be negative"),
        ({"business_share": 1.5}, "business_share must lie"),
        ({"marginal_tax_rate": 1.0}, "marginal_tax_rate must lie"),
        ({"depreciation": (0.8, 0.8)}, "exceeds 1"),
    ],
)
def test_business_treatment_validates_its_inputs(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(CarbitrageError, match=match):
        BusinessAssets(**kwargs)  # type: ignore[arg-type]
