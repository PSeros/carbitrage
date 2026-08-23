"""The carriers sold by volume or mass.

Petrol and diesel are the plain cases; LPG carries a volumetric penalty because
a litre of it holds less energy, and hydrogen is priced per kilogram.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import CarbitrageError
from ..rates import Timeline
from .base import EnergySource

__all__ = [
    "LPG",
    "Diesel",
    "Hydrogen",
    "Petrol",
]


@dataclass(frozen=True)
class _LiquidFuel(EnergySource):
    """Shared behaviour of the volumetric fuels."""

    consumption: float
    price: float = 0.0
    real_world_factor: float = 1.0

    def __post_init__(self) -> None:
        self._check(self.consumption, self.real_world_factor)

    def unit_cost(self, t: int, timeline: Timeline) -> float:
        return self._escalated(self.price, t, timeline)


@dataclass(frozen=True)
class Petrol(_LiquidFuel):
    """Petrol, litres per 100 km."""


@dataclass(frozen=True)
class Diesel(_LiquidFuel):
    """Diesel, litres per 100 km."""


@dataclass(frozen=True)
class LPG(_LiquidFuel):
    """Autogas, litres per 100 km.

    LPG carries its own volumetric penalty: its energy density is lower than
    petrol's, so the same journey burns more litres.  State ``consumption`` as
    the petrol figure and set ``volumetric_penalty`` (typically 1.15 to 1.25),
    or state the LPG figure directly and leave the penalty at 1.
    """

    volumetric_penalty: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.volumetric_penalty <= 0:
            raise CarbitrageError(
                f"volumetric_penalty must be positive, got {self.volumetric_penalty!r}"
            )

    @property
    def effective_consumption(self) -> float:
        return self.consumption * self.real_world_factor * self.volumetric_penalty


@dataclass(frozen=True)
class Hydrogen(_LiquidFuel):
    """Hydrogen, kilograms per 100 km."""
