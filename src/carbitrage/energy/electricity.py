"""Grid electricity, with charging losses and a home/public price split."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import CarbitrageError
from ..rates import Timeline
from .base import EnergySource

__all__ = [
    "Electricity",
]


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
