"""Scenario overlays, expected value and minimax regret."""

from __future__ import annotations

import pytest

from carbitrage.errors import CarbitrageError
from carbitrage.scenario import Scenario, ScenarioSet

DEFERRED_SUBSIDY = (
    "alternatives[A4 Incumbent then EV].legs[Hyundai Inster (deferred)].incentives[0].available"
)
REPAIR = "alternatives[A4 Incumbent then EV].legs[LPG incumbent].acquisition.upfront_extra"


@pytest.fixture(scope="module")
def overlays() -> ScenarioSet:
    return ScenarioSet(
        [
            Scenario("fuel collapse", {"lpg_price": 0.65, "petrol_price": 1.40}, probability=0.2),
            Scenario("subsidy expires", {DEFERRED_SUBSIDY: 0}, probability=0.3),
            Scenario("incumbent dies early", {REPAIR: 6_000.0}, probability=0.2),
        ]
    )


def test_weights_are_normalised(overlays: ScenarioSet) -> None:
    assert sum(overlays.weights()) == pytest.approx(1.0)


def test_the_base_case_absorbs_the_unclaimed_probability(overlays: ScenarioSet) -> None:
    # The three scenarios claim 0.7, so the base case gets 0.3.
    assert overlays.weights()[0] == pytest.approx(0.3)


def test_probabilities_may_not_exceed_one() -> None:
    with pytest.raises(CarbitrageError, match="exceeds 1"):
        ScenarioSet([Scenario("a", {}, probability=0.7), Scenario("b", {}, probability=0.6)])


def test_a_probability_must_be_a_probability() -> None:
    with pytest.raises(CarbitrageError, match=r"probability must lie in \[0, 1\]"):
        Scenario("a", {}, probability=1.5)


def test_scenario_names_must_be_unique() -> None:
    with pytest.raises(CarbitrageError, match="names must be unique"):
        ScenarioSet([Scenario("a", {}), Scenario("a", {})])


def test_a_set_needs_a_scenario() -> None:
    with pytest.raises(CarbitrageError, match="at least one scenario"):
        ScenarioSet([])
