"""The annual greenhouse-gas quota credit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..cashflow import CashFlowSeries, Component, Frequency, Recurring
from ..vehicle import Propulsion, Vehicle
from .base import Incentive

if TYPE_CHECKING:  # pragma: no cover
    from ..context import Context, Household

__all__ = [
    "ThgQuote",
]


@dataclass(frozen=True)
class ThgQuote(Incentive):
    """The annual greenhouse-gas quota credit.  Battery-electric vehicles only.

    Paid yearly on application, and tax-free for a private person.  It is a
    genuine inflow and is netted inside the stream rather than deducted from a
    cost, so it stays visible in the breakdown.

    Args:
        annual_amount: Credit per year.
        growth: Annual growth of the credit.  The market for these certificates
            is volatile; leaving this at zero holds it flat in nominal terms.
    """

    annual_amount: float = 0.0
    growth: float = 0.0

    def is_eligible(self, vehicle: Vehicle, household: Household) -> bool:
        return vehicle.drivetrain is Propulsion.BEV and self.annual_amount != 0.0

    def flows(self, vehicle: Vehicle, ctx: Context) -> CashFlowSeries:
        if not self.is_eligible(vehicle, ctx.household):
            return CashFlowSeries()
        return CashFlowSeries.of(
            Recurring(
                amount=self.annual_amount,
                frequency=Frequency.ANNUAL,
                growth=self.growth,
                start=ctx.start,
                end=ctx.last,
                label=Component.SUBSIDY,
                description="Greenhouse-gas quota credit",
            )
        )

    def constraints(self, vehicle: Vehicle, ctx: Context) -> tuple[str, ...]:
        return ()
