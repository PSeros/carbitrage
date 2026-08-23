"""A named bundle of overrides, and a collection of them to run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..errors import CarbitrageError
from ..params import set_params
from .analysis import ScenarioAnalysis

if TYPE_CHECKING:  # pragma: no cover
    from ..comparison import Case

__all__ = [
    "Scenario",
    "ScenarioSet",
]


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
            raise CarbitrageError(f"probability must lie in [0, 1], got {self.probability!r}")

    def apply(self, case: Case) -> Case:
        """``case`` with this scenario's overrides applied."""
        return set_params(case, dict(self.overrides))


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

    def __init__(self, scenarios: Sequence[Scenario], *, include_base: bool = True) -> None:
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
