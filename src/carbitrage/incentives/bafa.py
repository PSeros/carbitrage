"""The 2026 German purchase premium, and the rate tables behind it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from ..cashflow import CashFlowSeries, Component, OneOff
from ..errors import CarbitrageError
from ..vehicle import Vehicle, VehicleCategory
from .base import _ZERO_EMISSION, Incentive

if TYPE_CHECKING:  # pragma: no cover
    from ..context import Context, Household

__all__ = [
    "BAFA2026",
    "BAFA_2026_BEV",
    "BAFA_2026_PHEV",
]

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
