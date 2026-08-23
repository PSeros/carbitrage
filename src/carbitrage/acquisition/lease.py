"""Leasing, chained across terms, with an excess-kilometre settlement."""

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
    "Lease",
]


@dataclass(frozen=True)
class Lease(Acquisition):
    """A lease, chained across as many terms as the horizon requires.

    The lessee never owns the asset, so no residual is collected: that is the
    economic substance of :attr:`bears_residual_risk` being ``False``, and it is
    the single biggest difference from :class:`Financed`.

    Excess mileage is settled at the end of every term, on the mileage actually
    driven over that term.  When the horizon outruns one term, a further
    contract begins with the rate escalated by ``renewal_escalation`` and a
    further initial payment falling due.

    Args:
        monthly_rate: Contractual monthly rent.
        term_months: Length of one contract.
        initial_payment: Capitalised payment at the start of each contract.
        included_km: Mileage included per year.
        excess_km_rate: Charge per kilometre above the allowance.
        renewal_escalation: Rate increase applied to each successive contract.
        residual_risk_borne_by: Who carries the residual.  Only ``"lessor"`` is
            modelled; an open-calculation lease that hands residual risk back to
            the lessee is a different instrument and is rejected rather than
            silently approximated.
        rate_includes_subsidy: Set when the quoted rate already has a purchase
            premium capitalised into it, which is the norm for advertised German
            offers.  Combining that with an explicit incentive double counts,
            and the library warns.
    """

    monthly_rate: float = 0.0
    term_months: int = 36
    initial_payment: float = 0.0
    included_km: float = 0.0
    excess_km_rate: float = 0.0
    renewal_escalation: float = 0.0
    residual_risk_borne_by: str = "lessor"
    rate_includes_subsidy: bool = False

    def __post_init__(self) -> None:
        if self.term_months <= 0:
            raise CarbitrageError(f"term_months must be positive, got {self.term_months!r}")
        if self.residual_risk_borne_by != "lessor":
            raise CarbitrageError(
                "only a closed-calculation lease with residual risk on the lessor is modelled; "
                f"got residual_risk_borne_by={self.residual_risk_borne_by!r}"
            )

    @property
    def bears_residual_risk(self) -> bool:
        return False

    @property
    def subsidy_capitalised(self) -> bool:
        return self.rate_includes_subsidy

    def rate_for_contract(self, index: int) -> float:
        """Monthly rent of contract ``index``, counting the first as zero."""
        return self.monthly_rate * (1.0 + self.renewal_escalation) ** index

    def excess_km_settlement(self, ctx: Context) -> float:
        """Charge falling due at the end of one full term."""
        if self.excess_km_rate <= 0:
            return 0.0
        excess_per_year = max(ctx.usage.annual_km - self.included_km, 0.0)
        return excess_per_year * self.excess_km_rate * (self.term_months / 12.0)

    def flows(self, vehicle: Vehicle, ctx: Context) -> CashFlowSeries:
        out: list[CashFlow] = []
        if vehicle.setup_cost:
            out.append(
                OneOff(
                    amount=-vehicle.setup_cost,
                    at=ctx.start,
                    label=Component.ACQUISITION,
                    description=f"Setup cost for {vehicle.name}",
                )
            )
        settlement = self.excess_km_settlement(ctx)
        contract = 0
        start = ctx.start
        while start < ctx.last:
            end = min(start + self.term_months, ctx.last)
            rate = self.rate_for_contract(contract)
            if self.initial_payment:
                out.append(
                    OneOff(
                        amount=-self.initial_payment * (1.0 + self.renewal_escalation) ** contract,
                        at=start,
                        label=Component.LEASE,
                        description=f"Initial payment, contract {contract + 1}",
                    )
                )
            out.append(
                Recurring(
                    amount=-rate,
                    frequency=Frequency.PER_PERIOD,
                    start=start,
                    end=end,
                    label=Component.LEASE,
                    description=f"Lease rent, contract {contract + 1}",
                )
            )
            if settlement and end == start + self.term_months:
                out.append(
                    OneOff(
                        amount=-settlement,
                        at=end,
                        label=Component.LEASE,
                        description=f"Excess mileage, contract {contract + 1}",
                    )
                )
            start = end
            contract += 1
        return CashFlowSeries(tuple(out))
