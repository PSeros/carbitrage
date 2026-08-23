"""The three shapes every cost in a comparison takes.

A payment happens once, repeats, or lands when the asset leaves the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..rates import Timeline
from .base import Component, Frequency, ResidualValueModelLike

__all__ = [
    "OneOff",
    "Recurring",
    "Terminal",
]


@dataclass(frozen=True)
class OneOff:
    """A single payment at one period.

    Args:
        amount: Signed amount.  Negative for an outflow.
        at: Absolute period index.
        label: Which component this belongs to.
        description: Human-readable provenance, shown in breakdowns.
    """

    amount: float
    at: int = 0
    label: Component = Component.OTHER
    description: str = ""

    def amounts(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        timeline.check_period(self.at)
        out = timeline.zeros()
        out[self.at] = self.amount
        return out


@dataclass(frozen=True)
class Recurring:
    """A repeating payment, in arrears, optionally escalating and time-limited.

    Payments land at ``start + k * stride`` for ``k = 1, 2, ...`` while the period
    is at most ``end``.  Paying in arrears is why an annual flow starting at
    period 0 first pays at period 12 rather than period 0.

    Args:
        amount: Signed amount per payment, in period-0 money.
        frequency: How often the payment recurs.
        growth: Annual growth rate applied through the timeline's escalation
            index, so escalation is consistent with every other flow.
        start: Absolute period the entitlement begins.  The first payment is one
            stride later.
        end: Last absolute period a payment may land on.  ``None`` means the
            timeline horizon.
        label: Which component this belongs to.
        description: Human-readable provenance.
    """

    amount: float
    frequency: Frequency = Frequency.MONTHLY
    growth: float = 0.0
    start: int = 0
    end: int | None = None
    label: Component = Component.OTHER
    description: str = ""

    def amounts(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        stride = self.frequency.stride(timeline)
        last = timeline.n_periods if self.end is None else min(self.end, timeline.n_periods)
        timeline.check_period(self.start)
        out = timeline.zeros()
        if last <= self.start or self.amount == 0.0:
            return out
        periods = np.arange(self.start + stride, last + 1, stride, dtype=np.int64)
        if periods.size == 0:
            return out
        index = timeline.escalation_index(self.growth)
        out[periods] = self.amount * index[periods]
        return out


@dataclass(frozen=True)
class Terminal:
    """A terminal value realised when an asset leaves the model.

    Terminal values are *derived*, not entered free-hand: build them with
    :meth:`from_residual` so that assets of different ages — the successor in a
    replacement chain, for instance — are valued on the same curve.  The
    ``basis`` string records what the number came from and is carried into the
    breakdown.
    """

    amount: float
    at: int
    label: Component = Component.RESIDUAL
    description: str = ""
    basis: str = ""

    @classmethod
    def from_residual(
        cls,
        model: ResidualValueModelLike,
        *,
        price: float,
        years_held: float,
        at: int,
        age_at_acquisition: float = 0.0,
        description: str = "",
        label: Component = Component.RESIDUAL,
    ) -> Terminal:
        """Value ``price`` after ``years_held`` on ``model``'s curve, realised at ``at``."""
        value = model.value_after(
            price, years_held=years_held, age_at_acquisition=age_at_acquisition
        )
        age = age_at_acquisition + years_held
        return cls(
            amount=value,
            at=at,
            label=label,
            description=description,
            basis=f"{type(model).__name__} on {price:,.2f} paid at age "
            f"{age_at_acquisition:g} y, held to age {age:g} y",
        )

    def amounts(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        timeline.check_period(self.at)
        out = timeline.zeros()
        out[self.at] = self.amount
        return out
