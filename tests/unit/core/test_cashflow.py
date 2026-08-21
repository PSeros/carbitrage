"""Cash-flow construction, escalation, shifting and the labelled breakdown."""

import numpy as np
import pytest

from carbitrage.core.cashflow import (
    CashFlowSeries,
    Component,
    Frequency,
    OneOff,
    Recurring,
    Terminal,
)
from carbitrage.core.timeline import Timeline
from carbitrage.domain.residual import GeometricDecline
from carbitrage.errors import TimelineError


@pytest.fixture
def tl() -> Timeline:
    return Timeline(horizon_years=6, periods_per_year=12, rate=0.03, energy_escalation=0.02)


def test_oneoff_lands_at_its_period(tl: Timeline) -> None:
    amounts = OneOff(-1000.0, at=0, label=Component.ACQUISITION).amounts(tl)
    assert amounts[0] == -1000.0
    assert np.count_nonzero(amounts) == 1


def test_oneoff_pv_at_t0_is_undiscounted(tl: Timeline) -> None:
    assert CashFlowSeries.of(OneOff(-1000.0, at=0)).pv(tl) == pytest.approx(-1000.0)


def test_recurring_monthly_pays_in_arrears(tl: Timeline) -> None:
    amounts = Recurring(-100.0, Frequency.MONTHLY).amounts(tl)
    assert amounts[0] == 0.0  # nothing at t=0: payments are in arrears
    assert amounts[1] == -100.0
    assert np.count_nonzero(amounts) == 72


def test_recurring_annual_pays_at_year_ends(tl: Timeline) -> None:
    amounts = Recurring(300.0, Frequency.ANNUAL).amounts(tl)
    assert np.flatnonzero(amounts).tolist() == [12, 24, 36, 48, 60, 72]


def test_recurring_escalates_on_the_timeline_index(tl: Timeline) -> None:
    amounts = Recurring(-100.0, Frequency.MONTHLY, growth=0.02).amounts(tl)
    assert amounts[12] == pytest.approx(-100.0)  # still year one
    assert amounts[13] == pytest.approx(-102.0)
    assert amounts[25] == pytest.approx(-100.0 * 1.02**2)


def test_recurring_respects_start_and_end(tl: Timeline) -> None:
    amounts = Recurring(-100.0, Frequency.MONTHLY, start=24, end=36).amounts(tl)
    assert np.flatnonzero(amounts).tolist() == list(range(25, 37))


def test_recurring_starting_at_the_horizon_pays_nothing(tl: Timeline) -> None:
    assert np.count_nonzero(Recurring(-100.0, start=72).amounts(tl)) == 0


def test_annual_frequency_rejected_on_an_incompatible_grid() -> None:
    quarterly_grid = Timeline(horizon_years=3, periods_per_year=4)
    with pytest.raises(TimelineError, match="not representable"):
        Recurring(-10.0, Frequency.MONTHLY).amounts(quarterly_grid)


def test_zero_rate_pv_equals_the_undiscounted_sum() -> None:
    flat = Timeline(horizon_years=3, periods_per_year=12, rate=0.0)
    series = CashFlowSeries.of(OneOff(-500.0), Recurring(-10.0, Frequency.MONTHLY))
    assert series.pv(flat) == pytest.approx(series.undiscounted(flat))


def test_terminal_from_residual_records_its_basis(tl: Timeline) -> None:
    terminal = Terminal.from_residual(
        GeometricDecline(0.15), price=23_900, years_held=6, at=72, description="Residual"
    )
    assert terminal.amount == pytest.approx(23_900 * 0.85**6)
    assert "GeometricDecline" in terminal.basis
    assert "held to age 6 y" in terminal.basis


def test_shifting_moves_every_kind_of_flow(tl: Timeline) -> None:
    series = CashFlowSeries.of(
        OneOff(-100.0, at=0),
        Recurring(-10.0, Frequency.MONTHLY, start=0, end=12),
        Terminal(50.0, at=12),
    )
    shifted = series.shifted(24)
    amounts = shifted.amounts(tl)
    assert amounts[24] == pytest.approx(-100.0)
    assert np.flatnonzero(Recurring(-10.0, start=24, end=36).amounts(tl)).tolist() == list(
        range(25, 37)
    )
    assert amounts[36] == pytest.approx(50.0 - 10.0)  # terminal and last rent coincide


def test_breakdown_sums_to_the_total(tl: Timeline) -> None:
    series = CashFlowSeries.of(
        OneOff(-25_400.0, at=0, label=Component.ACQUISITION),
        OneOff(1_500.0, at=0, label=Component.DISPOSAL),
        Recurring(-60.0, Frequency.MONTHLY, growth=0.02, label=Component.ENERGY),
        Recurring(300.0, Frequency.ANNUAL, label=Component.SUBSIDY),
        Terminal(9_013.87, at=72, label=Component.RESIDUAL),
    )
    assert sum(series.breakdown(tl).values()) == pytest.approx(series.pv(tl))


def test_detail_keeps_one_line_per_flow(tl: Timeline) -> None:
    series = CashFlowSeries.of(
        OneOff(-100.0, label=Component.ACQUISITION, description="Purchase price"),
        OneOff(-50.0, label=Component.ACQUISITION, description="Wallbox"),
    )
    detail = series.detail(tl)
    assert [d.description for d in detail] == ["Purchase price", "Wallbox"]
    # Same label, two lines: the breakdown collapses them, the detail does not.
    assert series.breakdown(tl)[Component.ACQUISITION] == pytest.approx(-150.0)


def test_labelled_selects_a_subseries(tl: Timeline) -> None:
    series = CashFlowSeries.of(
        OneOff(-100.0, label=Component.ACQUISITION),
        OneOff(300.0, label=Component.SUBSIDY),
    )
    assert series.labelled(Component.SUBSIDY).pv(tl) == pytest.approx(300.0)


def test_series_addition_concatenates(tl: Timeline) -> None:
    a = CashFlowSeries.of(OneOff(-100.0))
    b = CashFlowSeries.of(OneOff(-50.0))
    assert (a + b).pv(tl) == pytest.approx(-150.0)
    assert len(a + b) == 2
