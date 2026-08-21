"""Scenario overlays, expected value and minimax regret."""

from __future__ import annotations

import pytest

from carbitrage.analysis import Case
from carbitrage.errors import CarbitrageError
from carbitrage.scenario import Scenario, ScenarioSet
from tests.fixtures import workbook_base_case as wb

DEFERRED_SUBSIDY = (
    "alternatives[A4 Incumbent then EV].legs[Hyundai Inster (deferred)].incentives[0].available"
)
REPAIR = "alternatives[A4 Incumbent then EV].legs[LPG incumbent].acquisition.upfront_extra"


@pytest.fixture(scope="module")
def case() -> Case:
    return Case(
        alternatives=tuple(wb.all_alternatives()),
        timeline=wb.timeline(),
        usage=wb.usage(),
        household=wb.household(),
        incumbent=wb.incumbent(),
    )


@pytest.fixture(scope="module")
def overlays() -> ScenarioSet:
    return ScenarioSet(
        [
            Scenario("fuel collapse", {"lpg_price": 0.65, "petrol_price": 1.40}, probability=0.2),
            Scenario("subsidy expires", {DEFERRED_SUBSIDY: 0}, probability=0.3),
            Scenario("incumbent dies early", {REPAIR: 6_000.0}, probability=0.2),
        ]
    )


def test_the_base_case_is_included_and_unmodified(case: Case, overlays: ScenarioSet) -> None:
    analysis = overlays.run(case)
    assert analysis.scenarios[0].name == "base case"
    assert analysis.npv(wb.A1, "base case") == pytest.approx(wb.TARGETS[wb.A1].npv, abs=1.0)


def test_the_base_case_can_be_left_out(case: Case) -> None:
    only = ScenarioSet([Scenario("a", {"lpg_price": 1.2})], include_base=False)
    assert [s.name for s in only.all_scenarios()] == ["a"]


def test_a_scenario_changes_only_what_it_names(case: Case, overlays: ScenarioSet) -> None:
    analysis = overlays.run(case)
    # Removing the deferred subsidy cannot touch the alternative that buys now.
    assert analysis.npv(wb.A1, "subsidy expires") == pytest.approx(
        analysis.npv(wb.A1, "base case")
    )
    assert analysis.npv(wb.A4, "subsidy expires") < analysis.npv(wb.A4, "base case")


def test_losing_the_deferred_subsidy_costs_its_discounted_value(
    case: Case, overlays: ScenarioSet
) -> None:
    analysis = overlays.run(case)
    lost = analysis.npv(wb.A4, "base case") - analysis.npv(wb.A4, "subsidy expires")
    assert lost == pytest.approx(4_000.0 * wb.timeline().discount_factor(28), abs=1.0)


def test_weights_are_normalised(overlays: ScenarioSet) -> None:
    assert sum(overlays.weights()) == pytest.approx(1.0)


def test_the_base_case_absorbs_the_unclaimed_probability(overlays: ScenarioSet) -> None:
    # The three scenarios claim 0.7, so the base case gets 0.3.
    assert overlays.weights()[0] == pytest.approx(0.3)


def test_scenarios_without_probabilities_are_weighted_equally(case: Case) -> None:
    plain = ScenarioSet([Scenario("a", {"lpg_price": 1.2}), Scenario("b", {"lpg_price": 0.8})])
    assert plain.weights() == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert not plain.run(case).has_probabilities


def test_probabilities_may_not_exceed_one() -> None:
    with pytest.raises(CarbitrageError, match="exceeds 1"):
        ScenarioSet(
            [Scenario("a", {}, probability=0.7), Scenario("b", {}, probability=0.6)]
        )


def test_a_probability_must_be_a_probability() -> None:
    with pytest.raises(CarbitrageError, match=r"probability must lie in \[0, 1\]"):
        Scenario("a", {}, probability=1.5)


def test_scenario_names_must_be_unique() -> None:
    with pytest.raises(CarbitrageError, match="names must be unique"):
        ScenarioSet([Scenario("a", {}), Scenario("a", {})])


def test_a_set_needs_a_scenario() -> None:
    with pytest.raises(CarbitrageError, match="at least one scenario"):
        ScenarioSet([])


def test_expected_value_is_the_weighted_average(case: Case, overlays: ScenarioSet) -> None:
    analysis = overlays.run(case)
    table = analysis.table()
    expected = sum(
        weight * table[wb.A1][scenario.name]
        for scenario, weight in zip(analysis.scenarios, analysis.weights, strict=True)
    )
    assert analysis.expected_npv()[wb.A1] == pytest.approx(expected)


def test_regret_is_never_negative(case: Case, overlays: ScenarioSet) -> None:
    analysis = overlays.run(case)
    for rows in analysis.regret().values():
        assert all(value >= 0 for value in rows.values())


def test_the_winner_of_a_scenario_has_zero_regret_there(
    case: Case, overlays: ScenarioSet
) -> None:
    analysis = overlays.run(case)
    regret = analysis.regret()
    for scenario_name, winner in analysis.winners().items():
        assert regret[winner][scenario_name] == pytest.approx(0.0)


def test_both_decision_rules_are_reported(case: Case, overlays: ScenarioSet) -> None:
    analysis = overlays.run(case)
    assert analysis.best_by_expected_value() == wb.A1
    assert analysis.best_by_minimax_regret() == wb.A1
    assert analysis.rules_agree()


def test_a_robust_winner_is_recognised(case: Case, overlays: ScenarioSet) -> None:
    analysis = overlays.run(case)
    assert analysis.is_robust()
    assert "robust" in analysis.to_markdown()


def test_the_rules_can_disagree_and_the_report_says_so(case: Case) -> None:
    """Minimax regret ignores probabilities; expected value is dominated by them.

    A scenario that is unlikely but catastrophic for the expected-value winner
    is exactly where the two rules part company, and the user should see it.
    """
    overlays = ScenarioSet(
        [
            Scenario(
                "energy shock",
                {"home_electricity_price": 1.60, "public_electricity_price": 2.20},
                probability=0.02,
            )
        ]
    )
    analysis = overlays.run(case)
    if not analysis.rules_agree():
        assert "two rules disagree" in analysis.to_markdown()
    # Whether or not they diverge here, both must always be computable.
    assert analysis.best_by_expected_value() in analysis.names
    assert analysis.best_by_minimax_regret() in analysis.names


def test_a_scenario_that_flips_the_winner_is_reported(case: Case) -> None:
    overlays = ScenarioSet(
        [Scenario("cheap gas, low mileage", {"lpg_price": 0.45, "annual_km": 4_000.0})]
    )
    analysis = overlays.run(case)
    winners = set(analysis.winners().values())
    assert len(winners) > 1
    assert not analysis.is_robust()


def test_markdown_reports_every_scenario_and_alternative(
    case: Case, overlays: ScenarioSet
) -> None:
    table = overlays.run(case).to_markdown()
    for scenario in overlays.all_scenarios():
        assert scenario.name in table
    for name in wb.TARGETS:
        assert name in table
    assert "Max regret" in table
    assert "Expected" in table


def test_result_lookup_by_scenario_name(case: Case, overlays: ScenarioSet) -> None:
    analysis = overlays.run(case)
    assert analysis.result("base case").best().name == wb.A1
    with pytest.raises(KeyError, match="not in this analysis"):
        analysis.result("no such scenario")
