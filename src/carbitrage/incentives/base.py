"""What every incentive has to answer: am I eligible, and what do I pay out."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Final

from ..cashflow import CashFlowSeries
from ..errors import EligibilityError
from ..vehicle import Propulsion, Vehicle

if TYPE_CHECKING:  # pragma: no cover
    from ..context import Context, Household

__all__ = [
    "Incentive",
]

_ZERO_EMISSION: Final[tuple[Propulsion, ...]] = (Propulsion.BEV, Propulsion.FCEV)


class Incentive(ABC):
    """A public benefit attaching to an asset and a buyer."""

    @abstractmethod
    def is_eligible(self, vehicle: Vehicle, household: Household) -> bool:
        """Whether this incentive applies at all."""

    @abstractmethod
    def flows(self, vehicle: Vehicle, ctx: Context) -> CashFlowSeries:
        """The cash consequences over ``ctx``'s window.  Empty when ineligible."""

    def constraints(self, vehicle: Vehicle, ctx: Context) -> tuple[str, ...]:
        """Non-cash obligations the incentive imposes, for the result to surface."""
        return ()

    def _require_eligible(self, vehicle: Vehicle, household: Household) -> None:
        if not self.is_eligible(vehicle, household):
            raise EligibilityError(
                f"{type(self).__name__} does not apply to {vehicle.name!r} for this household"
            )
