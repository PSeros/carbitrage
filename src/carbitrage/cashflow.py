"""Cash flows and the labelled series that aggregates them.

Sign convention, applied without exception below the presentation layer:
**outflows are negative, inflows positive**.  Grants, recurring credits, disposal
proceeds and terminal residual values are inflows and are netted *inside* the
stream; they are never dropped.  What is absent from these streams is operating
revenue attributable to the asset, because the asset earns nothing.

A :class:`CashFlowSeries` keeps its components labelled.  That breakdown is what
makes a result explainable, so it is never collapsed early: a user must be able
to ask "where does this EUR 20,844 come from" and get an answer that sums.

Flows carry *absolute* period indices and are resolved against a timeline only
when a present value is asked for.  That is what lets a replacement chain place
its second leg at month 24 without rewriting the flows it composes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from .errors import TimelineError
from .timeline import Timeline

__all__ = [
    "CashFlow",
    "CashFlowSeries",
    "Component",
    "ComponentPV",
    "Frequency",
    "OneOff",
    "Recurring",
    "Terminal",
]


class Component(StrEnum):
    """The labelled parts a present value decomposes into."""

    ACQUISITION = "acquisition"
    FINANCING = "financing"
    LEASE = "lease"
    ENERGY = "energy"
    INSURANCE = "insurance"
    MAINTENANCE = "maintenance"
    TAX = "tax"
    SUBSIDY = "subsidy"
    RESIDUAL = "residual"
    DISPOSAL = "disposal"
    OTHER = "other"


class Frequency(StrEnum):
    """How often a recurring flow is paid."""

    PER_PERIOD = "per_period"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMIANNUAL = "semiannual"
    ANNUAL = "annual"

    def stride(self, timeline: Timeline) -> int:
        """Number of timeline periods between two consecutive payments.

        Raises:
            TimelineError: if the frequency is not representable on the grid,
                e.g. monthly payments on an annual timeline.
        """
        if self is Frequency.PER_PERIOD:
            return 1
        per_year = {
            Frequency.MONTHLY: 12,
            Frequency.QUARTERLY: 4,
            Frequency.SEMIANNUAL: 2,
            Frequency.ANNUAL: 1,
        }[self]
        stride, remainder = divmod(timeline.periods_per_year, per_year)
        if remainder or stride < 1:
            raise TimelineError(
                f"{self.value} payments are not representable on a timeline with "
                f"{timeline.periods_per_year} periods per year"
            )
        return stride


@runtime_checkable
class CashFlow(Protocol):
    """A stream of amounts that can be resolved against a timeline.

    Declared with read-only members so that frozen dataclasses satisfy it.
    """

    @property
    def label(self) -> Component:
        """Which component of the breakdown this flow belongs to."""
        ...

    @property
    def description(self) -> str:
        """Human-readable provenance, shown in breakdowns."""
        ...

    def amounts(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """Signed amounts per period, indexed ``0 .. timeline.n_periods``."""
        ...


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


class ResidualValueModelLike(Protocol):
    """The part of a residual value model :class:`Terminal` depends on."""

    def value_after(
        self, price_paid: float, *, years_held: float, age_at_acquisition: float = 0.0
    ) -> float: ...


@dataclass(frozen=True)
class ComponentPV:
    """One labelled line of a present value breakdown."""

    label: Component
    description: str
    pv: float


@dataclass(frozen=True)
class CashFlowSeries:
    """An aggregate of labelled cash flows.

    The series is timeline-independent until :meth:`pv` or :meth:`amounts` is
    called, which is what makes it composable across the legs of a chain.
    """

    flows: tuple[CashFlow, ...] = ()

    def __add__(self, other: CashFlowSeries) -> CashFlowSeries:
        return CashFlowSeries(self.flows + other.flows)

    def __len__(self) -> int:
        return len(self.flows)

    @classmethod
    def of(cls, *flows: CashFlow) -> CashFlowSeries:
        """Build a series from loose flows, dropping ``None``-like empties."""
        return cls(tuple(flows))

    @classmethod
    def concat(cls, series: Iterable[CashFlowSeries]) -> CashFlowSeries:
        """Concatenate several series into one."""
        out: tuple[CashFlow, ...] = ()
        for s in series:
            out += s.flows
        return cls(out)

    def with_flows(self, *flows: CashFlow) -> CashFlowSeries:
        """A copy with additional flows appended."""
        return CashFlowSeries(self.flows + flows)

    def labelled(self, *labels: Component) -> CashFlowSeries:
        """The sub-series carrying any of ``labels``."""
        wanted = set(labels)
        return CashFlowSeries(tuple(f for f in self.flows if f.label in wanted))

    def shifted(self, periods: int) -> CashFlowSeries:
        """Every flow moved ``periods`` later.  Used when composing chain legs."""
        moved: list[CashFlow] = []
        for flow in self.flows:
            if isinstance(flow, OneOff | Terminal):
                moved.append(replace(flow, at=flow.at + periods))
            elif isinstance(flow, Recurring):
                moved.append(
                    replace(
                        flow,
                        start=flow.start + periods,
                        end=None if flow.end is None else flow.end + periods,
                    )
                )
            else:  # pragma: no cover - defensive, third-party CashFlow
                raise TypeError(f"cannot shift a {type(flow).__name__}")
        return CashFlowSeries(tuple(moved))

    def amounts(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """The netted amount vector over every period."""
        total = timeline.zeros()
        for flow in self.flows:
            total += flow.amounts(timeline)
        return total

    def pv(self, timeline: Timeline) -> float:
        """Net present value of the whole series.  Negative for a net cost."""
        return timeline.pv(self.amounts(timeline))

    def undiscounted(self, timeline: Timeline) -> float:
        """The nominal sum of every amount.  A control total, not a decision figure."""
        return float(np.sum(self.amounts(timeline)))

    def detail(self, timeline: Timeline) -> tuple[ComponentPV, ...]:
        """Per-flow present values, in declaration order."""
        return tuple(
            ComponentPV(
                label=flow.label,
                description=flow.description or flow.label.value,
                pv=timeline.pv(flow.amounts(timeline)),
            )
            for flow in self.flows
        )

    def breakdown(self, timeline: Timeline) -> dict[Component, float]:
        """Present value per component label.  Sums to :meth:`pv` by construction."""
        out: dict[Component, float] = {}
        for flow in self.flows:
            out[flow.label] = out.get(flow.label, 0.0) + timeline.pv(flow.amounts(timeline))
        return out
