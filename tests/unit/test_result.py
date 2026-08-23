"""The comparison result's readers.

The frame adapters are covered here because they are where a traceable number
meets a renderer, and a renderer that silently drops or misaligns a component
would break the one promise :mod:`carbitrage.engine.result` makes.
"""

from __future__ import annotations

import pytest

from carbitrage import Alternative, Timeline, Vehicle, compare
from carbitrage.cashflow import Component
from carbitrage.context import Usage
from carbitrage.energy import Electricity, Petrol
from carbitrage.residual import GeometricDecline


@pytest.fixture
def result():
    """Two alternatives that differ in which components they use at all."""
    timeline = Timeline(horizon_years=4, periods_per_year=12, rate=0.03)
    electric = Vehicle(
        name="electric",
        price=30_000.0,
        energy=Electricity(consumption=15.0, home_price=0.30),
        residual=GeometricDecline(0.15),
        insurance=600.0,
        maintenance=400.0,
    )
    petrol = Vehicle(
        name="petrol",
        price=22_000.0,
        energy=Petrol(consumption=7.0, price=1.80),
        residual=GeometricDecline(0.15),
        insurance=500.0,
        maintenance=700.0,
        annual_tax=150.0,
    )
    return compare(
        [Alternative(vehicle=electric, label="ev"), Alternative(vehicle=petrol, label="ice")],
        timeline,
        usage=Usage(annual_km=12_000.0),
    )


def test_the_breakdown_frame_has_one_column_per_alternative_in_order(result, pandas) -> None:
    frame = result.breakdown_frame()
    assert list(frame.columns) == list(result.names) == ["ev", "ice"]


def test_the_breakdown_frame_rows_follow_the_component_order(result, pandas) -> None:
    frame = result.breakdown_frame()
    order = [c.name for c in Component]
    assert list(frame.index) == [name for name in order if name in set(frame.index)]
    assert frame.index.name == "component"


def test_every_column_sums_to_that_alternative_s_npv(result, pandas) -> None:
    frame = result.breakdown_frame()
    for name, npv in result.npv().items():
        assert frame[name].sum() == pytest.approx(npv, rel=1e-9, abs=1e-6)


def test_a_component_one_alternative_never_uses_reads_as_zero(result, pandas) -> None:
    frame = result.breakdown_frame()
    assert Component.TAX.name in frame.index
    assert frame.loc[Component.TAX.name, "ev"] == 0.0
    assert frame.loc[Component.TAX.name, "ice"] < 0.0
    assert not frame.isna().to_numpy().any()


def test_rounding_to_whole_units_gives_integers(result, pandas) -> None:
    frame = result.breakdown_frame(decimals=0)
    assert all(pandas.api.types.is_integer_dtype(dtype) for dtype in frame.dtypes)
    assert frame.loc["ACQUISITION", "ev"] == round(result.breakdown("ev")[Component.ACQUISITION])


def test_rounding_to_places_keeps_floats(result, pandas) -> None:
    frame = result.breakdown_frame(decimals=2)
    exact = result.breakdown("ev")[Component.ENERGY]
    assert frame.loc["ENERGY", "ev"] == pytest.approx(round(exact, 2), abs=1e-12)
