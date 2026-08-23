"""The result grid: expected value where probabilities are known, minimax
regret where they are not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..comparison import ComparisonResult
    from .overlay import Scenario  # pragma: no cover

__all__ = [
    "ScenarioAnalysis",
]


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
        raise KeyError(f"{scenario!r} is not in this analysis: {[s.name for s in self.scenarios]}")

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
