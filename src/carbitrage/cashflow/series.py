"""A labelled bundle of flows, and the breakdown it discounts to.

Keeping the label on every flow is what makes a present value traceable
afterwards rather than one opaque number.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from ..rates import Timeline
from .base import CashFlow, Component
from .flows import OneOff, Recurring, Terminal

__all__ = [
    "CashFlowSeries",
    "ComponentPV",
]


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
