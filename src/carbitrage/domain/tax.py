"""Tax treatments.

The library's default is a **private household**: no depreciation, no input VAT
recovery, and subsidies received tax-free.  That default is not baked into the
engine.  It sits behind an ABC so that a business case — degressive depreciation,
input VAT, a company-car benefit in kind — can be added without the cash-flow
engine changing at all.

A treatment sees the finished, labelled series for one leg and returns an
adjusted one.  It may only *add* labelled flows; it never rewrites the flows it
was given, so the original components remain traceable in the breakdown.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from ..core.cashflow import CashFlow, CashFlowSeries, Component, OneOff
from ..errors import CarbitrageError

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from .context import Context
    from .vehicle import Vehicle

__all__ = [
    "DEGRESSIVE_2026",
    "STRAIGHT_LINE_6Y",
    "BusinessAssets",
    "PrivateHousehold",
    "TaxTreatment",
]

DEGRESSIVE_2026: Final[tuple[float, ...]] = (0.75, 0.10, 0.05, 0.05, 0.03, 0.02)
"""The German accelerated schedule for business EVs acquired to 31.12.2027:
75 % in the first year, then 10/5/5/3/2 %."""

STRAIGHT_LINE_6Y: Final[tuple[float, ...]] = (1 / 6,) * 6
"""Straight line over the six-year useful life the tax tables assume for cars."""

_VAT_BEARING: Final[frozenset[Component]] = frozenset(
    {
        Component.ACQUISITION,
        Component.ENERGY,
        Component.MAINTENANCE,
        Component.LEASE,
        Component.RESIDUAL,
        Component.DISPOSAL,
    }
)
"""Components that carry VAT in Germany.  Insurance premiums and circulation tax
do not, interest does not, and grants are outside the scope of VAT."""


class TaxTreatment(ABC):
    """Adjusts a leg's cash flows for the buyer's tax position."""

    @abstractmethod
    def adjust(self, series: CashFlowSeries, vehicle: Vehicle, ctx: Context) -> CashFlowSeries:
        """Return ``series`` with any tax consequences appended."""


@dataclass(frozen=True)
class PrivateHousehold(TaxTreatment):
    """A private buyer: nothing is deductible and nothing is recoverable.

    No depreciation, because the asset is not business property.  No input VAT
    recovery, because the buyer is not registered.  Subsidies and THG credits are
    tax-free in the hands of a private person, so they need no adjustment
    either.  The result is that the gross cash flows already tell the whole
    story, which is why this returns the series untouched.
    """

    def adjust(self, series: CashFlowSeries, vehicle: Vehicle, ctx: Context) -> CashFlowSeries:
        return series


@dataclass(frozen=True)
class BusinessAssets(TaxTreatment):
    """The asset is business property held by a VAT-registered buyer.

    Two consequences are modelled, both as additional labelled flows:

    * **Input VAT** on VAT-bearing components is recovered in the period the
      expense falls, and output VAT is remitted on the disposal proceeds.
    * **Depreciation** shelters taxable profit, worth ``marginal_tax_rate`` per
      unit written off.  The shield lands at each year end.

    Args:
        vat_rate: Statutory VAT rate.
        business_share: Fraction of use that is business use.  Private use is
            neither deductible nor recoverable.
        marginal_tax_rate: The rate at which a deduction is worth cash.
        depreciation: Fraction of the depreciable base written off in each year,
            starting with the year of acquisition.  Must not exceed 1 in total.
    """

    vat_rate: float = 0.19
    business_share: float = 1.0
    marginal_tax_rate: float = 0.30
    depreciation: tuple[float, ...] = STRAIGHT_LINE_6Y

    def __post_init__(self) -> None:
        if self.vat_rate < 0:
            raise CarbitrageError(f"vat_rate must not be negative, got {self.vat_rate!r}")
        if not 0.0 <= self.business_share <= 1.0:
            raise CarbitrageError(f"business_share must lie in [0, 1], got {self.business_share!r}")
        if not 0.0 <= self.marginal_tax_rate < 1.0:
            raise CarbitrageError(
                f"marginal_tax_rate must lie in [0, 1), got {self.marginal_tax_rate!r}"
            )
        if sum(self.depreciation) > 1.0 + 1e-9:
            raise CarbitrageError(
                f"depreciation writes off {sum(self.depreciation):.4f} of the base, which exceeds 1"
            )

    def adjust(self, series: CashFlowSeries, vehicle: Vehicle, ctx: Context) -> CashFlowSeries:
        added: list[CashFlow] = list(self._vat_flows(series, ctx))
        added.extend(self._depreciation_flows(vehicle, ctx))
        return series.with_flows(*added)

    def _vat_flows(self, series: CashFlowSeries, ctx: Context) -> list[CashFlow]:
        """Input VAT recovered, and output VAT remitted on proceeds."""
        if self.vat_rate == 0.0:
            return []
        fraction = self.vat_rate / (1.0 + self.vat_rate) * self.business_share
        amounts = series.labelled(*_VAT_BEARING).amounts(ctx.timeline)
        flows: list[CashFlow] = []
        for t in amounts.nonzero()[0]:
            # Reverses the sign: VAT on an outflow is recovered, VAT on proceeds is remitted.
            flows.append(
                OneOff(
                    amount=-float(amounts[t]) * fraction,
                    at=int(t),
                    label=Component.TAX,
                    description="Input VAT" if amounts[t] < 0 else "Output VAT on proceeds",
                )
            )
        return flows

    def _depreciation_flows(self, vehicle: Vehicle, ctx: Context) -> list[CashFlow]:
        """The tax shield from writing the asset down, realised at year ends."""
        if self.marginal_tax_rate == 0.0 or vehicle.price <= 0:
            return []
        net_of_vat = vehicle.price / (1.0 + self.vat_rate) if self.vat_rate else vehicle.price
        base = net_of_vat * self.business_share
        per_year = ctx.timeline.periods_per_year
        flows: list[CashFlow] = []
        for year, fraction in enumerate(self.depreciation, start=1):
            at = ctx.start + year * per_year
            if at > ctx.last or fraction == 0.0:
                break
            flows.append(
                OneOff(
                    amount=base * fraction * self.marginal_tax_rate,
                    at=at,
                    label=Component.TAX,
                    description=f"Depreciation shield, year {year}",
                )
            )
        return flows
