"""Public incentives: grants, recurring credits and tax exemptions.

One protocol covers all three, which is the point.  A tax exemption modelled as
an incentive rather than as a hard-coded zero is what lets a used vehicle inherit
the *remainder* of an exemption that began at someone else's first registration.

Eligibility and cash flow are deliberately separate concerns.  Some consequences
of an incentive are not cash flows at all: the 2026 purchase premium carries a
minimum holding period, which constrains the decision without ever appearing in
a stream.  Those surface as :attr:`Incentive.constraints`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final

from ..core.cashflow import CashFlowSeries, Component, Frequency, OneOff, Recurring
from ..errors import CarbitrageError, EligibilityError
from .vehicle import Propulsion, Vehicle, VehicleCategory

if TYPE_CHECKING:  # pragma: no cover
    from .context import Context, Household

__all__ = [
    "BAFA2026",
    "BAFA_2026_BEV",
    "BAFA_2026_PHEV",
    "Incentive",
    "ThgQuote",
    "VehicleTaxExemption",
]

_ZERO_EMISSION: Final[tuple[Propulsion, ...]] = (Propulsion.BEV, Propulsion.FCEV)
"""Drivetrains the battery-electric schedule covers."""

#: Income bands of the 2026 purchase premium, as (lower bound, no child, 1 child, 2+).
#: Source: workbook sheet `Foerdermatrix`, after the ADAC/BMUV schedule.
#:
#: The bands are published as "up to 45,000" and "45,001 to 60,000", so a
#: household on exactly 45,000 sits in the *lower* band.  The bounds carry the
#: extra cent that makes that true; rounding them to whole thousands would
#: misgrant 1,000 EUR to every household sitting precisely on a boundary.
BAFA_2026_BEV: Final[tuple[tuple[float, float, float, float], ...]] = (
    (0.0, 5000.0, 5500.0, 6000.0),
    (45_000.01, 4000.0, 4500.0, 5000.0),
    (60_000.01, 3000.0, 3500.0, 4000.0),
    (80_000.01, 0.0, 3500.0, 4000.0),
    (85_000.01, 0.0, 0.0, 4000.0),
    (90_000.01, 0.0, 0.0, 0.0),
)

#: The same schedule for plug-in hybrids and range extenders.
BAFA_2026_PHEV: Final[tuple[tuple[float, float, float, float], ...]] = (
    (0.0, 3500.0, 4000.0, 4500.0),
    (45_000.01, 2500.0, 3000.0, 3500.0),
    (60_000.01, 1500.0, 2000.0, 2500.0),
    (80_000.01, 0.0, 2000.0, 2500.0),
    (85_000.01, 0.0, 0.0, 2500.0),
    (90_000.01, 0.0, 0.0, 0.0),
)


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


@dataclass(frozen=True)
class BAFA2026(Incentive):
    """The 2026 German purchase premium: means-tested, new vehicles only.

    The grant is looked up on an income-by-children matrix, applied for after
    registration and paid out after processing, so it lands with a lag rather
    than at t=0 — which matters once it is discounted.

    Used vehicles are not eligible, and neither are light quadricycles of class
    L6e or L7e.  A minimum holding period applies; breaking it triggers
    repayment.  That is a constraint on the decision, not a cash flow, so it is
    reported through :meth:`constraints` rather than modelled as one.

    Args:
        disbursement_lag_months: Months from acquisition to payment.
        minimum_holding_months: Holding period the grant is conditional on.
        available: Set to ``False`` to model the programme's budget being
            exhausted — the relevant case for an alternative that defers its
            purchase into a later year.
        matrix: The income-by-children schedule to apply.
    """

    disbursement_lag_months: int = 4
    minimum_holding_months: int = 36
    available: bool = True
    matrix: tuple[tuple[float, float, float, float], ...] = BAFA_2026_BEV

    def __post_init__(self) -> None:
        if self.disbursement_lag_months < 0:
            raise CarbitrageError(
                f"disbursement_lag_months must not be negative, got "
                f"{self.disbursement_lag_months!r}"
            )

    def amount(self, household: Household) -> float:
        """The grant this household qualifies for, before eligibility checks."""
        column = min(household.children, 2) + 1
        band = self.matrix[0]
        for row in self.matrix:
            if household.taxable_income >= row[0]:
                band = row
            else:
                break
        return band[column]

    def is_eligible(self, vehicle: Vehicle, household: Household) -> bool:
        if not self.available:
            return False
        if vehicle.is_used:
            return False
        if vehicle.category is not VehicleCategory.M1:
            return False
        if self.matrix is BAFA_2026_BEV and vehicle.drivetrain not in _ZERO_EMISSION:
            return False
        return self.amount(household) > 0.0

    def flows(self, vehicle: Vehicle, ctx: Context) -> CashFlowSeries:
        if not self.is_eligible(vehicle, ctx.household):
            return CashFlowSeries()
        at = ctx.start + self.disbursement_lag_months
        if at > ctx.timeline.n_periods:
            return CashFlowSeries()
        return CashFlowSeries.of(
            OneOff(
                amount=self.amount(ctx.household),
                at=at,
                label=Component.SUBSIDY,
                description=f"Purchase premium, paid month {at}",
            )
        )

    def constraints(self, vehicle: Vehicle, ctx: Context) -> tuple[str, ...]:
        if not self.is_eligible(vehicle, ctx.household):
            return ()
        return (
            f"the purchase premium on the {vehicle.name} requires a minimum holding period of "
            f"{self.minimum_holding_months} months; disposing earlier triggers repayment.",
        )


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
