"""Exemption from circulation tax, for a capped number of years."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from ..cashflow import CashFlowSeries, Component, Frequency, Recurring
from ..vehicle import Propulsion, Vehicle
from .base import Incentive

if TYPE_CHECKING:  # pragma: no cover
    from ..context import Context, Household

__all__ = [
    "VehicleTaxExemption",
]


@dataclass(frozen=True)
class VehicleTaxExemption(Incentive):
    """Suppresses circulation tax for a capped number of years.

    Modelled as a *credit* against the vehicle's own ``annual_tax`` rather than
    by zeroing that field.  Two things follow, and both matter:

    * the exemption is keyed on **first registration**, not on purchase, so a
      used vehicle inherits only the unexpired remainder;
    * the tax that would otherwise be paid stays visible in the breakdown next
      to the credit that offsets it, instead of silently vanishing.

    Args:
        max_years: Length of the exemption from first registration.
        expires: Statutory end date of the scheme, capping the exemption
            regardless of how much of ``max_years`` is left.
        eligible_propulsion: Which drivetrains qualify.
    """

    max_years: float = 10.0
    expires: date | None = date(2035, 12, 31)
    eligible_propulsion: tuple[Propulsion, ...] = (Propulsion.BEV,)

    def is_eligible(self, vehicle: Vehicle, household: Household) -> bool:
        return vehicle.drivetrain in self.eligible_propulsion and vehicle.annual_tax != 0.0

    def remaining_years(self, vehicle: Vehicle, ctx: Context) -> float:
        """Unexpired exemption in years at the moment this leg acquires the vehicle.

        A vehicle first registered three years ago has three years of a ten-year
        exemption behind it, whoever was driving it.
        """
        elapsed = vehicle.age_at_acquisition
        remaining = max(self.max_years - elapsed, 0.0)
        if self.expires is not None and vehicle.first_registration is not None:
            acquired_on = _add_years(vehicle.first_registration, elapsed)
            statutory = (self.expires - acquired_on).days / 365.25
            remaining = min(remaining, max(statutory, 0.0))
        return remaining

    def flows(self, vehicle: Vehicle, ctx: Context) -> CashFlowSeries:
        if not self.is_eligible(vehicle, ctx.household):
            return CashFlowSeries()
        years = min(self.remaining_years(vehicle, ctx), ctx.years_held)
        if years <= 0.0:
            return CashFlowSeries()
        end = ctx.start + round(years * ctx.timeline.periods_per_year)
        return CashFlowSeries.of(
            Recurring(
                amount=vehicle.annual_tax / ctx.timeline.periods_per_year,
                frequency=Frequency.PER_PERIOD,
                start=ctx.start,
                end=min(end, ctx.last),
                label=Component.TAX,
                description=f"Circulation tax exemption, {years:.2f} y remaining",
            )
        )

    def constraints(self, vehicle: Vehicle, ctx: Context) -> tuple[str, ...]:
        return ()


def _add_years(start: date, years: float) -> date:
    """``start`` advanced by ``years``, to day resolution."""
    return start.fromordinal(start.toordinal() + round(years * 365.25))
