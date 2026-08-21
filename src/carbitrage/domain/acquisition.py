"""How an asset is paid for.

The acquisition mode owns the acquisition outflow *and* the terminal value,
because the two belong together: whoever pays for the asset is the one who owns
its residual risk.  A lessee pays neither the price nor collects the residual,
which is exactly why a lease is not a purchase with a payment plan.

An annuity loan is likewise not the same as a cash purchase once the loan rate
differs from the discount rate — borrowing below the discount rate is worth
something, borrowing above it costs something, and the library should be able to
show that rather than assert it.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.cashflow import (
    CashFlow,
    CashFlowSeries,
    Component,
    Frequency,
    OneOff,
    Recurring,
    Terminal,
)
from ..errors import CarbitrageError, DoubleCountingWarning

if TYPE_CHECKING:  # pragma: no cover
    from .context import Context
    from .vehicle import Vehicle

__all__ = ["Acquisition", "Financed", "Lease", "Purchase"]


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


@dataclass(frozen=True)
class Purchase(Acquisition):
    """Outright purchase, paid in full at acquisition.

    Args:
        upfront_extra: A further payment made at acquisition that is not part of
            the price — a repair bill that triggers the decision, delivery or
            registration charges.
        already_owned: The asset is in hand, so no price is paid.  Its value is
            still at risk and its residual is still collected.  The cash
            consequence of keeping it is the *forgone* disposal proceeds, which
            are recognised as an inflow on every alternative that disposes of
            it — see :class:`~carbitrage.domain.context.Incumbent`.
    """

    upfront_extra: float = 0.0
    already_owned: bool = False

    def flows(self, vehicle: Vehicle, ctx: Context) -> CashFlowSeries:
        out: list[CashFlow] = []
        if not self.already_owned:
            out.append(
                OneOff(
                    amount=-vehicle.price,
                    at=ctx.start,
                    label=Component.ACQUISITION,
                    description=f"Purchase price of {vehicle.name}",
                )
            )
        if vehicle.setup_cost:
            out.append(
                OneOff(
                    amount=-vehicle.setup_cost,
                    at=ctx.start,
                    label=Component.ACQUISITION,
                    description=f"Setup cost for {vehicle.name}",
                )
            )
        if self.upfront_extra:
            out.append(
                OneOff(
                    amount=-self.upfront_extra,
                    at=ctx.start,
                    label=Component.ACQUISITION,
                    description="Upfront extra",
                )
            )
        return CashFlowSeries(tuple(out) + self._terminal(vehicle, ctx))


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
                        amount=-self.initial_payment
                        * (1.0 + self.renewal_escalation) ** contract,
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
