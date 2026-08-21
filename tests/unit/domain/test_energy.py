"""Energy sources.  Reference figures come from the workbook's `Annahmen` sheet."""

import pytest

from carbitrage.core.timeline import Timeline
from carbitrage.domain.energy import (
    LPG,
    BivalentSource,
    Diesel,
    Electricity,
    Hydrogen,
    Petrol,
)
from carbitrage.errors import CarbitrageError


@pytest.fixture
def tl() -> Timeline:
    return Timeline(horizon_years=6, periods_per_year=12, rate=0.03, energy_escalation=0.02)


@pytest.fixture
def ev_electricity() -> Electricity:
    # Annahmen C34/C35/C24/C26/C25
    return Electricity(
        consumption=15.1, real_world_factor=1.18, home_price=0.30, public_price=0.55, home_share=0.8
    )


def test_blended_price_matches_workbook(ev_electricity: Electricity) -> None:
    assert ev_electricity.blended_price == pytest.approx(0.35)  # C27


def test_effective_consumption_matches_workbook(ev_electricity: Electricity) -> None:
    assert ev_electricity.effective_consumption == pytest.approx(17.818)  # C36


def test_cost_per_100km_matches_workbook(ev_electricity: Electricity, tl: Timeline) -> None:
    assert ev_electricity.cost_per_100km(1, tl) == pytest.approx(6.2363)  # Modelle J11


def test_charging_loss_applies_on_top_of_consumption() -> None:
    lossy = Electricity(consumption=15.0, home_price=0.30, charging_loss=0.10)
    assert lossy.effective_consumption == pytest.approx(15.0 / 0.9)


def test_home_share_of_zero_uses_the_public_price() -> None:
    public_only = Electricity(consumption=15.0, home_price=0.30, public_price=0.55, home_share=0.0)
    assert public_only.blended_price == pytest.approx(0.55)


def test_unit_cost_escalates_on_the_timeline_index(
    ev_electricity: Electricity, tl: Timeline
) -> None:
    assert ev_electricity.unit_cost(12, tl) == pytest.approx(0.35)
    assert ev_electricity.unit_cost(13, tl) == pytest.approx(0.35 * 1.02)


def test_lpg_volumetric_penalty_matches_workbook() -> None:
    # C68 = 7.5 l/100 km on petrol, C69 = 1.2 penalty, so C70 = 9.0 on gas.
    gas = LPG(consumption=7.5, price=0.99, volumetric_penalty=1.2)
    assert gas.effective_consumption == pytest.approx(9.0)


def test_bivalent_source_matches_workbook_blended_fuel_cost(tl: Timeline) -> None:
    # C72: 0.9 * 9.0 l * 0.99 EUR + 0.1 * 7.5 l * 2.10 EUR = 9.594 EUR/100 km
    incumbent = BivalentSource(
        primary=LPG(consumption=7.5, price=0.99, volumetric_penalty=1.2),
        secondary=Petrol(consumption=7.5, price=2.10),
        primary_share=0.90,
    )
    assert incumbent.cost_per_100km(1, tl) == pytest.approx(9.594)


def test_bivalent_source_escalates_both_legs(tl: Timeline) -> None:
    incumbent = BivalentSource(
        primary=LPG(consumption=7.5, price=0.99, volumetric_penalty=1.2),
        secondary=Petrol(consumption=7.5, price=2.10),
        primary_share=0.90,
    )
    assert incumbent.cost_per_100km(13, tl) == pytest.approx(9.594 * 1.02)


def test_bivalent_share_of_one_is_the_primary_source(tl: Timeline) -> None:
    gas = LPG(consumption=9.0, price=0.99)
    both = BivalentSource(gas, Petrol(consumption=7.5, price=2.10), primary_share=1.0)
    assert both.cost_per_100km(1, tl) == pytest.approx(gas.cost_per_100km(1, tl))


def test_diesel_and_hydrogen_are_priced_per_their_own_unit(tl: Timeline) -> None:
    assert Diesel(consumption=5.0, price=1.80).cost_per_100km(1, tl) == pytest.approx(9.0)
    assert Hydrogen(consumption=1.0, price=13.85).cost_per_100km(1, tl) == pytest.approx(13.85)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"consumption": -1.0}, "consumption must not be negative"),
        ({"consumption": 15.0, "real_world_factor": 0.0}, "real_world_factor must be positive"),
        ({"consumption": 15.0, "home_share": 1.5}, "home_share must lie"),
        ({"consumption": 15.0, "charging_loss": 1.0}, "charging_loss must lie"),
    ],
)
def test_electricity_validates_its_inputs(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(CarbitrageError, match=match):
        Electricity(**kwargs)


def test_bivalent_validates_its_share() -> None:
    with pytest.raises(CarbitrageError, match="primary_share must lie"):
        BivalentSource(Petrol(1.0), Petrol(1.0), primary_share=1.5)
