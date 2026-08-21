"""Named parameter overlays, and two ways of choosing between them.

A scenario is a bundle of parameter overrides with a name: "fuel price
collapse", "subsidy expires", "incumbent dies early".  Running the comparison
under each gives a table of how the ranking holds up.

Two decision rules are offered, and the library deliberately reports both:

* **Expected value** weights each scenario by its probability.  It is the right
  rule when the probabilities are trustworthy.
* **Minimax regret** picks the alternative whose worst-case shortfall against
  the best available choice is smallest.  It needs no probabilities at all.

Here the probabilities are soft — nobody knows the odds that a subsidy budget is
exhausted — and when they are soft the two rules can disagree.  The user
deserves to see that disagreement rather than have one rule chosen for them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..errors import CarbitrageError
from .params import set_params

if TYPE_CHECKING:  # pragma: no cover
    from ..engine.comparison import Case
    from ..engine.result import ComparisonResult

__all__ = ["Scenario", "ScenarioAnalysis", "ScenarioSet"]


@dataclass(frozen=True)
class Scenario:
    """A named set of parameter overrides.

    Args:
        name: Display name.
        overrides: Parameter name to value, using the same aliases and dotted
            paths as :mod:`carbitrage.study.params`.
        probability: Subjective probability.  Optional: leave it unset and only
            the regret analysis is meaningful, which is the honest position when
            the odds are unknown.
        note: Why this scenario is worth considering.
    """

    name: str
    overrides: Mapping[str, float] = field(default_factory=dict)
    probability: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.probability is not None and not 0.0 <= self.probability <= 1.0:
            raise CarbitrageError(
                f"probability must lie in [0, 1], got {self.probability!r}"
            )

    def apply(self, case: Case) -> Case:
        """``case`` with this scenario's overrides applied."""
        return set_params(case, dict(self.overrides))


@dataclass(frozen=True)
class ScenarioAnalysis:
    """Every alternative under every scenario, with both decision rules."""

    names: tuple[str, ...]
    scenarios: tuple[Scenario, ...]
    results: tuple[ComparisonResult, ...]
    weights: tuple[float, ...]

    # -------------------------------------------------------------- readings

    def result(self, scenario: str) -> ComparisonResult:
        """The full comparison under one named scenario."""
        for candidate, result in zip(self.scenarios, self.results, strict=True):
            if candidate.name == scenario:
                return result
        raise KeyError(
            f"{scenario!r} is not in this analysis: {[s.name for s in self.scenarios]}"
        )

    def npv(self, alternative: str, scenario: str) -> float:
        """Net present value of one alternative under one scenario."""
        return self.result(scenario)[alternative].npv

    def table(self) -> dict[str, dict[str, float]]:
        """Net present value keyed by alternative, then scenario."""
        return {
            name: {
                scenario.name: result[name].npv
                for scenario, result in zip(self.scenarios, self.results, strict=True)
            }
            for name in self.names
        }

    def winners(self) -> dict[str, str]:
        """The best alternative in each scenario."""
        return {
            scenario.name: result.best().name
            for scenario, result in zip(self.scenarios, self.results, strict=True)
        }

    def is_robust(self) -> bool:
        """Whether the same alternative wins in every scenario."""
        return len(set(self.winners().values())) == 1

    # ------------------------------------------------------- expected value

    @property
    def has_probabilities(self) -> bool:
        """Whether any scenario carried an explicit probability."""
        return any(s.probability is not None for s in self.scenarios)

    def expected_npv(self) -> dict[str, float]:
        """Probability-weighted net present value per alternative."""
        table = self.table()
        return {
            name: sum(
                weight * table[name][scenario.name]
                for scenario, weight in zip(self.scenarios, self.weights, strict=True)
            )
            for name in self.names
        }

    def best_by_expected_value(self) -> str:
        """The alternative with the highest expected net present value."""
        expected = self.expected_npv()
        return max(expected, key=lambda name: expected[name])

    # -------------------------------------------------------------- regret

    def regret(self) -> dict[str, dict[str, float]]:
        """Shortfall against the best available choice, per alternative and scenario.

        Regret is never negative: it is what you lose by having committed to this
        alternative once you learn which scenario came true.
        """
        table = self.table()
        out: dict[str, dict[str, float]] = {name: {} for name in self.names}
        for scenario, result in zip(self.scenarios, self.results, strict=True):
            best = max(result[name].npv for name in self.names)
            for name in self.names:
                out[name][scenario.name] = best - table[name][scenario.name]
        return out

    def max_regret(self) -> dict[str, float]:
        """The worst regret each alternative suffers across all scenarios."""
        return {name: max(rows.values()) for name, rows in self.regret().items()}

    def best_by_minimax_regret(self) -> str:
        """The alternative whose worst case is least bad.  Needs no probabilities."""
        worst = self.max_regret()
        return min(worst, key=lambda name: worst[name])

    def rules_agree(self) -> bool:
        """Whether expected value and minimax regret pick the same alternative."""
        return self.best_by_expected_value() == self.best_by_minimax_regret()

    # ------------------------------------------------------------ reporting

    def to_markdown(self, *, decimals: int = 0) -> str:
        """A table of present values by scenario, with both rules underneath."""
        header = "| Alternative | " + " | ".join(s.name for s in self.scenarios)
        if self.has_probabilities:
            header += " | Expected"
        header += " | Max regret |"
        rule = "|---|" + "---:|" * (len(self.scenarios) + (2 if self.has_probabilities else 1))

        table = self.table()
        expected = self.expected_npv()
        worst = self.max_regret()
        lines = [header, rule]
        for name in self.names:
            cells = [f"{table[name][s.name]:,.{decimals}f}" for s in self.scenarios]
            if self.has_probabilities:
                cells.append(f"{expected[name]:,.{decimals}f}")
            cells.append(f"{worst[name]:,.{decimals}f}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")

        lines.append("")
        lines.append("| Scenario | Winner |")
        lines.append("|---|---|")
        for scenario_name, winner in self.winners().items():
            lines.append(f"| {scenario_name} | {winner} |")

        lines.append("")
        ev = self.best_by_expected_value()
        mr = self.best_by_minimax_regret()
        if self.has_probabilities:
            lines.append(f"Highest expected value: **{ev}**.")
        lines.append(f"Lowest worst-case regret: **{mr}** ({worst[mr]:,.{decimals}f}).")
        if self.is_robust():
            lines.append("The same alternative wins in every scenario, so the choice is robust.")
        elif self.rules_agree():
            lines.append(
                "The winner changes between scenarios, but both decision rules still agree."
            )
        else:
            lines.append(
                f"The two rules disagree: expected value favours {ev}, minimax regret favours "
                f"{mr}.  With probabilities this soft, that disagreement is the finding."
            )
        return "\n".join(lines)


@dataclass(frozen=True, init=False)
class ScenarioSet:
    """A collection of scenarios to run one case against.

    Args:
        scenarios: The overlays.  Probabilities, where given, are normalised to
            sum to one; scenarios without one share whatever is left over, and if
            none are given every scenario is weighted equally.
        include_base: Prepend an unmodified "base case" scenario.
    """

    scenarios: tuple[Scenario, ...]
    include_base: bool = True

    def __init__(
        self, scenarios: Sequence[Scenario], *, include_base: bool = True
    ) -> None:
        if not scenarios:
            raise CarbitrageError("a scenario set needs at least one scenario")
        names = [s.name for s in scenarios]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise CarbitrageError(f"scenario names must be unique; repeated: {sorted(duplicates)}")
        stated = sum(s.probability for s in scenarios if s.probability is not None)
        if stated > 1.0 + 1e-9:
            raise CarbitrageError(
                f"stated scenario probabilities sum to {stated:.4f}, which exceeds 1"
            )
        object.__setattr__(self, "scenarios", tuple(scenarios))
        object.__setattr__(self, "include_base", include_base)

    def __replace__(self, **changes: object) -> ScenarioSet:
        scenarios = changes.pop("scenarios", self.scenarios)
        include_base = changes.pop("include_base", self.include_base)
        if changes:
            raise TypeError(f"unexpected fields for ScenarioSet: {sorted(changes)}")
        assert isinstance(scenarios, tuple)
        return ScenarioSet(scenarios, include_base=bool(include_base))

    def all_scenarios(self) -> tuple[Scenario, ...]:
        """The scenarios that will be run, base case included if requested."""
        if not self.include_base:
            return self.scenarios
        stated = sum(s.probability for s in self.scenarios if s.probability is not None)
        base_probability = (
            max(1.0 - stated, 0.0)
            if all(s.probability is not None for s in self.scenarios)
            else None
        )
        base = Scenario(
            name="base case",
            overrides={},
            probability=base_probability,
            note="the assumptions as supplied",
        )
        return (base, *self.scenarios)

    def weights(self) -> tuple[float, ...]:
        """Normalised probabilities, one per scenario.

        Scenarios without a stated probability share whatever probability the
        stated ones leave unclaimed; if none states one, the weights are equal.
        """
        scenarios = self.all_scenarios()
        stated = [s.probability for s in scenarios]
        known = [p for p in stated if p is not None]
        if not known:
            return tuple(1.0 / len(scenarios) for _ in scenarios)
        remaining = max(1.0 - sum(known), 0.0)
        unstated = sum(1 for p in stated if p is None)
        share = remaining / unstated if unstated else 0.0
        raw = [share if p is None else p for p in stated]
        total = sum(raw)
        if total <= 0:
            raise CarbitrageError("scenario probabilities sum to zero, so nothing can be weighted")
        return tuple(value / total for value in raw)

    def run(self, case: Case) -> ScenarioAnalysis:
        """Evaluate ``case`` under every scenario."""
        scenarios = self.all_scenarios()
        results = tuple(scenario.apply(case).run() for scenario in scenarios)
        names = tuple(alt.name for alt in case.alternatives)
        return ScenarioAnalysis(
            names=names,
            scenarios=scenarios,
            results=results,
            weights=self.weights(),
        )
