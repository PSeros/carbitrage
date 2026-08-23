"""What every energy carrier has to answer: unit cost, and cost per 100 km."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..errors import CarbitrageError
from ..rates import Timeline

__all__ = [
    "EnergySource",
]


class EnergySource(ABC):
    """What one unit of energy costs, and what 100 km of driving costs.

    Attributes:
        consumption: Consumption per 100 km in the source's own unit, before
            any real-world uplift.
    """

    consumption: float
    real_world_factor: float

    @abstractmethod
    def unit_cost(self, t: int, timeline: Timeline) -> float:
        """Price of one unit of energy at period ``t``, escalated."""

    @property
    def effective_consumption(self) -> float:
        """Consumption per 100 km after real-world and delivery-loss factors."""
        return self.consumption * self.real_world_factor

    def cost_per_100km(self, t: int, timeline: Timeline) -> float:
        """Cost of 100 km at period ``t``, escalated."""
        return self.effective_consumption * self.unit_cost(t, timeline)

    def _escalated(self, price: float, t: int, timeline: Timeline) -> float:
        return price * timeline.escalation(t, timeline.energy_escalation)

    @staticmethod
    def _check(consumption: float, real_world_factor: float) -> None:
        if consumption < 0:
            raise CarbitrageError(f"consumption must not be negative, got {consumption!r}")
        if real_world_factor <= 0:
            raise CarbitrageError(f"real_world_factor must be positive, got {real_world_factor!r}")
