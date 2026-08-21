"""One course of action, evaluated as a stream of labelled cash flows.

An :class:`Alternative` binds an asset to the way it is paid for, the way it is
used, the incentives it attracts and what happens to the asset already owned.
It is the smallest unit that can be compared against another.

Operating costs live here rather than on the acquisition mode, because they are
incurred whether the asset is bought, financed or leased.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from ..core.cashflow import CashFlow, CashFlowSeries, Component, Frequency, OneOff, Recurring
from ..domain.acquisition import Acquisition, Lease, Purchase, warn_on_double_counted_subsidy
from ..domain.context import Context
from ..domain.incentive import Incentive
from ..domain.vehicle import Vehicle
from ..errors import CarbitrageError

__all__ = ["Alternative", "Evaluable"]


@runtime_checkable
class Evaluable(Protocol):
    """Anything :func:`~carbitrage.engine.comparison.compare` can rank.

    Both :class:`Alternative` and :class:`~carbitrage.engine.chain.ReplacementChain`
    satisfy it, which is what lets a chain stand beside a plain alternative in
    the same comparison.
    """

    @property
    def name(self) -> str:
        """Display name, unique within one comparison."""
        ...

    @property
    def life_years(self) -> float | None:
        """Useful life, or ``None`` when the alternative runs to the horizon."""
        ...

    def flows(self, ctx: Context) -> CashFlowSeries:
        """The complete labelled stream for this course of action."""
        ...

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        """Non-cash obligations a decision-maker has to honour."""
        ...


@dataclass(frozen=True)
class Alternative:
    """An asset, how it is paid for, and what it attracts.

    Args:
        vehicle: The asset.
        acquisition: How it is paid for.  Defaults to an outright purchase.
        incentives: Public benefits claimed.
        label: Display name.  Defaults to the vehicle's name.
        life_years: Useful life when it is shorter than the horizon, which
            obliges the caller to wrap it in a
            :class:`~carbitrage.engine.chain.ReplacementChain`.
        disposes_incumbent: Whether taking this course sells the asset already
            owned.  True for every alternative that replaces it; false for the
            one that keeps it.  The proceeds are an inflow at t=0, and their
            absence is the opportunity cost of keeping the old asset.
    """

    vehicle: Vehicle
    acquisition: Acquisition = field(default_factory=Purchase)
    incentives: tuple[Incentive, ...] = ()
    label: str | None = None
    life_years: float | None = None
    disposes_incumbent: bool = True

    def __post_init__(self) -> None:
        if self.life_years is not None and self.life_years <= 0:
            raise CarbitrageError(f"life_years must be positive, got {self.life_years!r}")
        warn_on_double_counted_subsidy(self.acquisition, bool(self.incentives))

    @property
    def name(self) -> str:
        return self.label or self.vehicle.name

    def leased(self, lease: Lease, *, label: str | None = None) -> Alternative:
        """The same vehicle and usage, leased instead of bought."""
        return replace(
            self,
            acquisition=lease,
            label=label or f"{self.name} (leased)",
        )

    # ------------------------------------------------------------------ flows

    def flows(self, ctx: Context) -> CashFlowSeries:
        """Acquisition, operating costs, incentives and incumbent disposal."""
        series = self.acquisition.flows(self.vehicle, ctx)
        series += self._operating_flows(ctx)
        series += self._incumbent_disposal(ctx)
        for incentive in self.incentives:
            series += incentive.flows(self.vehicle, ctx)
        return ctx.tax.adjust(series, self.vehicle, ctx)

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        out: tuple[str, ...] = ()
        for incentive in self.incentives:
            out += incentive.constraints(self.vehicle, ctx)
        return out

    def _operating_flows(self, ctx: Context) -> CashFlowSeries:
        """Energy and the fixed annual costs, paid in arrears over the window."""
        v = self.vehicle
        per_year = ctx.timeline.periods_per_year
        out: list[CashFlow] = []

        km_per_period = ctx.usage.km_per_period(ctx.timeline)
        if km_per_period and v.energy.effective_consumption:
            base_price = v.energy.cost_per_100km(0, ctx.timeline)
            out.append(
                Recurring(
                    amount=-km_per_period / 100.0 * base_price,
                    frequency=Frequency.PER_PERIOD,
                    growth=ctx.timeline.energy_escalation,
                    start=ctx.start,
                    end=ctx.last,
                    label=Component.ENERGY,
                    description=f"Energy for {v.name}",
                )
            )

        for amount, label, description in (
            (v.insurance, Component.INSURANCE, "Insurance"),
            (v.maintenance, Component.MAINTENANCE, "Maintenance and wear"),
            (v.annual_tax, Component.TAX, "Circulation tax"),
            (v.other_annual_cost, Component.MAINTENANCE, "Other recurring cost"),
        ):
            if amount:
                out.append(
                    Recurring(
                        amount=-amount / per_year,
                        frequency=Frequency.PER_PERIOD,
                        growth=ctx.timeline.cost_escalation,
                        start=ctx.start,
                        end=ctx.last,
                        label=label,
                        description=f"{description} for {v.name}",
                    )
                )
        return CashFlowSeries(tuple(out))

    def _incumbent_disposal(self, ctx: Context) -> CashFlowSeries:
        """Proceeds of selling the asset already owned, at t=0.

        Only the leg that starts the comparison disposes of it; a chain's later
        legs inherit an incumbent that is already gone.
        """
        incumbent = ctx.incumbent
        if incumbent is None or not self.disposes_incumbent or ctx.start != 0:
            return CashFlowSeries()
        if incumbent.market_value == 0:
            return CashFlowSeries()
        return CashFlowSeries.of(
            OneOff(
                amount=incumbent.market_value,
                at=0,
                label=Component.DISPOSAL,
                description=f"Disposal of {incumbent.vehicle.name}",
            )
        )
