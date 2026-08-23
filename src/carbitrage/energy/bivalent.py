"""Two carriers sharing one vehicle's mileage."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import CarbitrageError
from ..rates import Timeline
from .base import EnergySource

__all__ = [
    "BivalentSource",
]


@dataclass(frozen=True)
class BivalentSource(EnergySource):
    """Two sources sharing one vehicle's mileage.

    A bivalent LPG conversion runs mostly on gas and falls back to petrol on
    cold starts and an empty gas tank.  Composing two sources expresses that
    without a bespoke class, and the same composite covers a plug-in hybrid's
    electric/petrol split.

    Args:
        primary: The source carrying ``primary_share`` of the mileage.
        secondary: The source carrying the rest.
        primary_share: Fraction of mileage on the primary source.
    """

    primary: EnergySource
    secondary: EnergySource
    primary_share: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.primary_share <= 1.0:
            raise CarbitrageError(f"primary_share must lie in [0, 1], got {self.primary_share!r}")

    @property
    def consumption(self) -> float:  # type: ignore[override]
        """Mileage-weighted consumption.  Only meaningful when the units agree."""
        return self.primary.consumption * self.primary_share + self.secondary.consumption * (
            1.0 - self.primary_share
        )

    @property
    def real_world_factor(self) -> float:  # type: ignore[override]
        """Not defined for a composite; each leg carries its own."""
        return 1.0

    @property
    def effective_consumption(self) -> float:
        return (
            self.primary.effective_consumption * self.primary_share
            + self.secondary.effective_consumption * (1.0 - self.primary_share)
        )

    def unit_cost(self, t: int, timeline: Timeline) -> float:
        """The mileage-weighted blended unit price.

        Units differ between the legs, so this figure is only interpretable
        alongside :meth:`cost_per_100km`, which is the number that matters.
        """
        return self.primary.unit_cost(t, timeline) * self.primary_share + self.secondary.unit_cost(
            t, timeline
        ) * (1.0 - self.primary_share)

    def cost_per_100km(self, t: int, timeline: Timeline) -> float:
        return self.primary.cost_per_100km(
            t, timeline
        ) * self.primary_share + self.secondary.cost_per_100km(t, timeline) * (
            1.0 - self.primary_share
        )
