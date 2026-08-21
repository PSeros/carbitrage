"""Energy sources and what a hundred kilometres cost on each.

Consumption is stated in the source's own unit per 100 km — kWh for
electricity, litres for liquid fuels, kilograms for hydrogen — and the model
turns that into money for a given period, escalating the unit price on the
timeline's energy index.

Two adjustments belong here rather than in the vehicle, because both are
properties of how the energy is delivered and consumed rather than of the asset:

* ``real_world_factor`` — the uplift of realistic consumption over the WLTP or
  NEDC type-approval figure.
* the charging-loss factor on :class:`Electricity` — energy billed at the meter
  exceeds energy stored in the battery, and the type-approval figure is measured
  at the battery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..core.timeline import Timeline
from ..errors import CarbitrageError

__all__ = [
    "LPG",
    "BivalentSource",
    "Diesel",
    "Electricity",
    "EnergySource",
    "Hydrogen",
    "Petrol",
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
            raise CarbitrageError(
                f"real_world_factor must be positive, got {real_world_factor!r}"
            )


@dataclass(frozen=True)
class Electricity(EnergySource):
    """Grid electricity with a home/public price split and charging losses.

    Args:
        consumption: kWh per 100 km at the battery, typically the WLTP figure.
        home_price: Price per kWh charging at home or a wallbox.
        public_price: Price per kWh charging publicly.
        home_share: Fraction of energy drawn at the home price.  Without a
            private charging point this is 0, which frequently flips the result.
        real_world_factor: Uplift of real consumption over type approval.
        charging_loss: Fraction of energy lost between meter and battery.  It is
            applied *on top of* nominal consumption, because the type-approval
            figure is measured at the battery but the bill is at the meter.
    """

    consumption: float
    home_price: float = 0.0
    public_price: float = 0.0
    home_share: float = 1.0
    real_world_factor: float = 1.0
    charging_loss: float = 0.0

    def __post_init__(self) -> None:
        self._check(self.consumption, self.real_world_factor)
        if not 0.0 <= self.home_share <= 1.0:
            raise CarbitrageError(f"home_share must lie in [0, 1], got {self.home_share!r}")
        if not 0.0 <= self.charging_loss < 1.0:
            raise CarbitrageError(f"charging_loss must lie in [0, 1), got {self.charging_loss!r}")

    @property
    def blended_price(self) -> float:
        """The home/public mix price per kWh, before escalation."""
        return self.home_price * self.home_share + self.public_price * (1.0 - self.home_share)

    @property
    def effective_consumption(self) -> float:
        return self.consumption * self.real_world_factor / (1.0 - self.charging_loss)

    def unit_cost(self, t: int, timeline: Timeline) -> float:
        return self._escalated(self.blended_price, t, timeline)


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
            raise CarbitrageError(
                f"primary_share must lie in [0, 1], got {self.primary_share!r}"
            )

    @property
    def consumption(self) -> float:  # type: ignore[override]
        """Mileage-weighted consumption.  Only meaningful when the units agree."""
        return (
            self.primary.consumption * self.primary_share
            + self.secondary.consumption * (1.0 - self.primary_share)
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
        return (
            self.primary.unit_cost(t, timeline) * self.primary_share
            + self.secondary.unit_cost(t, timeline) * (1.0 - self.primary_share)
        )

    def cost_per_100km(self, t: int, timeline: Timeline) -> float:
        return (
            self.primary.cost_per_100km(t, timeline) * self.primary_share
            + self.secondary.cost_per_100km(t, timeline) * (1.0 - self.primary_share)
        )
