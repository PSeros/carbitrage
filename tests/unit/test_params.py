"""Naming a parameter: marks, aliases and dotted paths.

The three ways to name one all end at :func:`~carbitrage.params.resolve`, so
that is where most of this file aims.  What it holds in place is that a mark
behaves as an ordinary number everywhere the engine touches it, and as a name
everywhere a study does.
"""

from __future__ import annotations

import pytest

from carbitrage import Alternative, Case, Timeline, Vehicle
from carbitrage.acquisition import Purchase
from carbitrage.comparison import ReplacementChain
from carbitrage.context import Usage
from carbitrage.energy import LPG, Electricity
from carbitrage.errors import CarbitrageError
from carbitrage.params import (
    ALIASES,
    Uncertain,
    describe_parameters,
    find,
    get_param,
    name_of,
    resolve,
    set_param,
    uncertainties,
)
from carbitrage.residual import GeometricDecline

REPAIR = Uncertain(2_500.0, "repair_bill")
LIFE = Uncertain(2.0, "incumbent_life")


@pytest.fixture
def case() -> Case:
    """A chain whose first leg carries both marks, so paths are nested."""
    ev = Vehicle(
        "EV",
        price=23_900.0,
        energy=Electricity(consumption=15.0, home_price=0.30),
        residual=GeometricDecline(0.15),
    )
    old = Vehicle(
        "incumbent",
        price=1_500.0,
        energy=LPG(consumption=7.5, price=0.99),
        residual=GeometricDecline(0.25),
    )
    defer = ReplacementChain(
        Alternative(
            old,
            Purchase(upfront_extra=REPAIR, already_owned=True),
            life_years=LIFE,
            label="keep",
        ),
        Alternative(ev, Purchase(), label="later"),
        label="defer",
    )
    return Case(
        alternatives=(Alternative(ev, Purchase(), label="now"), defer),
        timeline=Timeline(horizon_years=6, periods_per_year=12, rate=0.03),
        usage=Usage(annual_km=12_000.0),
    )


# ----------------------------------------------------------------- the mark


def test_a_mark_is_an_ordinary_number() -> None:
    """The base case has to evaluate, so the mark cannot be a hole."""
    assert REPAIR == 2_500.0
    assert REPAIR + 1 == 2_501.0
    assert f"{REPAIR:,.2f}" == "2,500.00"
    assert isinstance(REPAIR, float)


def test_a_mark_needs_a_label_to_be_addressed_by() -> None:
    with pytest.raises(CarbitrageError, match="non-empty label"):
        Uncertain(1.0, "  ")


def test_a_mark_says_what_it_is() -> None:
    assert repr(REPAIR) == "Uncertain(2500.0, 'repair_bill')"


def test_a_name_is_a_string_or_a_mark() -> None:
    assert name_of(REPAIR) == "repair_bill"
    assert name_of("annual_km") == "annual_km"


# ------------------------------------------------------------- resolution


def test_a_mark_resolves_to_where_it_was_written(case: Case) -> None:
    assert resolve(case, REPAIR) == ("alternatives[defer].legs[keep].acquisition.upfront_extra",)


def test_the_label_resolves_the_same_as_the_mark(case: Case) -> None:
    assert resolve(case, "repair_bill") == resolve(case, REPAIR)


def test_aliases_and_paths_still_resolve(case: Case) -> None:
    assert resolve(case, "annual_km") == ("usage.annual_km",)
    assert resolve(case, "timeline.rate") == ("timeline.rate",)


def test_one_label_on_several_fields_addresses_them_together() -> None:
    """How you say "these two move together"."""
    shared = Uncertain(0.30, "electricity")
    ev = Vehicle(
        "EV",
        price=20_000.0,
        energy=Electricity(consumption=15.0, home_price=shared, public_price=shared),
        residual=GeometricDecline(0.15),
    )
    case = Case(
        alternatives=(Alternative(ev, Purchase(), label="a"),),
        timeline=Timeline(horizon_years=3, periods_per_year=12, rate=0.03),
    )
    assert len(resolve(case, "electricity")) == 2
    moved = set_param(case, shared, 0.50)
    assert get_param(moved, "electricity") == 0.50


def test_a_mark_shadows_an_alias_of_the_same_name() -> None:
    """A mark was written for this case; an alias was written for every case."""
    marked = Uncertain(11_000.0, "annual_km")
    ev = Vehicle(
        "EV",
        price=20_000.0,
        energy=Electricity(consumption=15.0, home_price=marked),
        residual=GeometricDecline(0.15),
    )
    case = Case(
        alternatives=(Alternative(ev, Purchase(), label="a"),),
        timeline=Timeline(horizon_years=3, periods_per_year=12, rate=0.03),
        usage=Usage(annual_km=12_000.0),
    )
    assert "annual_km" in ALIASES
    assert resolve(case, "annual_km") == ("alternatives[a].vehicle.energy.home_price",)


# ---------------------------------------------------------------- overrides


def test_an_override_keeps_the_mark_so_a_swept_case_can_be_swept_again(case: Case) -> None:
    once = set_param(case, REPAIR, 6_000.0)
    assert get_param(once, "repair_bill") == 6_000.0
    assert get_param(set_param(once, "repair_bill", 800.0), REPAIR) == 800.0


def test_a_mark_does_not_survive_into_a_field_declared_int() -> None:
    """Keeping the declared type wins: a count of periods is not a float."""
    case = Case(
        alternatives=(
            Alternative(
                Vehicle(
                    "EV",
                    price=20_000.0,
                    energy=Electricity(consumption=15.0, home_price=0.30),
                    residual=GeometricDecline(0.15),
                ),
                Purchase(),
                label="a",
            ),
        ),
        timeline=Timeline(horizon_years=3, periods_per_year=12, rate=0.03),
    )
    changed = set_param(case, "timeline.periods_per_year", 4)
    assert changed.timeline.periods_per_year == 4
    assert type(changed.timeline.periods_per_year) is int


# ---------------------------------------------------------------- discovery


def test_the_case_lists_its_marks(case: Case) -> None:
    assert uncertainties(case) == {
        "repair_bill": ("alternatives[defer].legs[keep].acquisition.upfront_extra",),
        "incumbent_life": ("alternatives[defer].legs[keep].life_years",),
    }


def test_find_locates_a_parameter_nobody_marked(case: Case) -> None:
    assert find(case, "LIFE") == ["alternatives[defer].legs[keep].life_years"]
    assert set(find(case, "rate")) <= set(describe_parameters(case))


def test_an_unknown_name_says_what_would_have_worked(case: Case) -> None:
    with pytest.raises(CarbitrageError) as raised:
        resolve(case, "repair")
    message = str(raised.value)
    assert "Uncertain(value, 'repair')" in message
    assert "repair_bill" in message  # the mark it might have meant
    assert "annual_km" in message  # the aliases it might have meant


def test_an_unknown_name_points_at_the_paths_that_contain_it(case: Case) -> None:
    with pytest.raises(CarbitrageError, match=r"acquisition\.upfront_extra"):
        resolve(case, "upfront")


def test_a_wrong_path_still_gets_the_walker_s_complaint(case: Case) -> None:
    """A name with a separator was meant as a path, so answer it as one."""
    with pytest.raises(CarbitrageError, match="has no field 'rat'"):
        resolve(case, "timeline.rat")
