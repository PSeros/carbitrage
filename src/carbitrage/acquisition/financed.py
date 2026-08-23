"""Purchase on an annuity loan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..cashflow import (
    CashFlow,
    CashFlowSeries,
    Component,
    Frequency,
    OneOff,
    Recurring,
)
from ..errors import CarbitrageError
from .base import Acquisition

if TYPE_CHECKING:  # pragma: no cover
    from ..context import Context
    from ..vehicle import Vehicle

__all__ = [
    "Financed",
]


@dataclass(frozen=True)
class Financed(Acquisition):
    """Purchase funded by an annuity loan.

    The buyer owns the asset and collects its residual; only the timing of the
    money changes.  That timing is not neutral: discounting a level annuity at
    ``i`` when it was priced at ``loan_rate`` leaves a gain when the loan is
    cheap and a loss when it is dear.  With ``loan_rate == rate`` and no fees,
    the present value collapses back to that of :class:`Purchase`.

    Args:
        loan_rate: Nominal annual rate of the loan.
        term_months: Repayment term.  The balloon, if any, falls due at its end.
        down_payment: Paid at acquisition; the balance is financed.
        balloon: Final instalment falling due with the last payment.
        fees: One-off arrangement fees paid at acquisition.
    """

    loan_rate: float = 0.0
    term_months: int = 48
    down_payment: float = 0.0
    balloon: float = 0.0
    fees: float = 0.0

    def __post_init__(self) -> None:
        if self.term_months <= 0:
            raise CarbitrageError(f"term_months must be positive, got {self.term_months!r}")
        if self.loan_rate <= -1.0:
            raise CarbitrageError(f"loan_rate must exceed -100 %, got {self.loan_rate!r}")
        if self.down_payment < 0 or self.balloon < 0 or self.fees < 0:
            raise CarbitrageError("down_payment, balloon and fees must not be negative")

    def instalment(self, vehicle: Vehicle, periods_per_year: int) -> float:
        """The level payment that amortises the financed balance over the term."""
        principal = vehicle.price + vehicle.setup_cost - self.down_payment
        if principal <= 0:
            return 0.0
        n = self.term_months
        i = (1.0 + self.loan_rate) ** (1.0 / periods_per_year) - 1.0
        balloon_pv = self.balloon / (1.0 + i) ** n if self.balloon else 0.0
        amortised = principal - balloon_pv
        if i == 0.0:
            return amortised / n
        annuity_factor = float(1.0 - (1.0 + i) ** -n) / i
        return float(amortised / annuity_factor)

    def flows(self, vehicle: Vehicle, ctx: Context) -> CashFlowSeries:
        per_year = ctx.timeline.periods_per_year
        payment = self.instalment(vehicle, per_year)
        upfront = self.down_payment + self.fees
        out: list[CashFlow] = []
        if upfront:
            out.append(
                OneOff(
                    amount=-upfront,
                    at=ctx.start,
                    label=Component.ACQUISITION,
                    description=f"Down payment and fees on {vehicle.name}",
                )
            )
        if payment:
            out.append(
                Recurring(
                    amount=-payment,
                    frequency=Frequency.PER_PERIOD,
                    start=ctx.start,
                    end=min(ctx.start + self.term_months, ctx.last),
                    label=Component.FINANCING,
                    description=f"Loan instalment at {self.loan_rate:.2%}",
                )
            )
        if self.balloon:
            at = ctx.start + self.term_months
            if at <= ctx.last:
                out.append(
                    OneOff(
                        amount=-self.balloon,
                        at=at,
                        label=Component.FINANCING,
                        description="Balloon payment",
                    )
                )
        return CashFlowSeries(tuple(out) + self._terminal(vehicle, ctx))
