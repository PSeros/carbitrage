"""What every acquisition mode has to answer, and the double-counting guard."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..cashflow import (
    CashFlow,
    CashFlowSeries,
    Terminal,
)
from ..errors import DoubleCountingWarning

if TYPE_CHECKING:  # pragma: no cover
    from ..context import Context
    from ..vehicle import Vehicle

__all__ = [
    "Acquisition",
]


class Acquisition(ABC):
    """Turns an asset and a window into the cash flows of owning or renting it."""

    @abstractmethod
    def flows(self, vehicle: Vehicle, ctx: Context) -> CashFlowSeries:
        """Acquisition, financing and terminal-value flows over ``ctx``'s window."""

    @property
    def bears_residual_risk(self) -> bool:
        """Whether the holder collects the asset's terminal value."""
        return True

    @property
    def subsidy_capitalised(self) -> bool:
        """Whether a public subsidy is already priced into this arrangement.

        Advertised German lease rates almost always have the purchase premium
        baked in as a capitalised initial payment.  Adding an explicit
        :class:`~carbitrage.domain.incentive.Incentive` on top then counts it twice.
        """
        return False

    def _terminal(self, vehicle: Vehicle, ctx: Context) -> tuple[CashFlow, ...]:
        """The residual value, derived from the vehicle's curve at *its* age."""
        if ctx.years_held <= 0 or vehicle.price <= 0:
            return ()
        return (
            Terminal.from_residual(
                vehicle.residual,
                price=vehicle.price,
                years_held=ctx.years_held,
                age_at_acquisition=vehicle.age_at_acquisition,
                at=ctx.last,
                description=f"Residual value of {vehicle.name}",
            ),
        )


def warn_on_double_counted_subsidy(acquisition: Acquisition, has_incentive: bool) -> None:
    """Warn when a capitalised subsidy meets an explicit incentive.

    Advertised German lease factors of 0.15 to 0.35 % of list price already
    contain the purchase premium as a capitalised initial payment.  Supplying an
    :class:`~carbitrage.domain.incentive.Incentive` as well counts the same money twice
    and can invert the ranking, so this is loud rather than quiet.
    """
    if has_incentive and acquisition.subsidy_capitalised:
        warnings.warn(
            "This lease rate is marked as already including the purchase premium, and an "
            "explicit incentive was supplied as well.  The premium is being counted twice.  "
            "Either drop the incentive or quote a rate without it capitalised.",
            DoubleCountingWarning,
            stacklevel=3,
        )
