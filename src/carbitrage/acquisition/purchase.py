"""Outright purchase: pay at acquisition, own the residual."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..cashflow import (
    CashFlow,
    CashFlowSeries,
    Component,
    OneOff,
)
from .base import Acquisition

if TYPE_CHECKING:  # pragma: no cover
    from ..context import Context
    from ..vehicle import Vehicle

__all__ = [
    "Purchase",
]


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
