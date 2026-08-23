"""Named bundles of overrides, and choosing between them.

A scenario is a set of parameter overrides under a name, optionally with a
probability.  Running a case against a `ScenarioSet` gives expected NPV where
probabilities are known and minimax regret where they are not.
"""

from __future__ import annotations

from .analysis import ScenarioAnalysis
from .overlay import Scenario, ScenarioSet

__all__ = [
    "Scenario",
    "ScenarioAnalysis",
    "ScenarioSet",
]
