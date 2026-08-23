"""One alternative resolved against one context, and the difference between two."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..cashflow import CashFlowSeries, Component, ComponentPV
from ..context import Context
from ..errors import CarbitrageError

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = [
    "Evaluation",
    "Incremental",
]

_DEFAULT_MATERIALITY = 0.03


@dataclass(frozen=True)
class Evaluation:
    """One alternative, resolved against one context."""

    name: str
    series: CashFlowSeries
    ctx: Context
    constraints: tuple[str, ...] = ()

    @property
    def amounts(self) -> npt.NDArray[np.float64]:
        """The netted signed amount vector."""
        return self.series.amounts(self.ctx.timeline)

    @property
    def npv(self) -> float:
        """Net present value of the stream.  Negative for a net cost."""
        return self.series.pv(self.ctx.timeline)

    @property
    def pv_of_outflows(self) -> float:
        """The NPV with its sign flipped, for readers who prefer costs positive.

        This is a presentation convenience only.  It is *not* an ``abs()``: an
        alternative whose inflows exceed its outflows keeps a negative figure
        here, which is the honest answer.
        """
        return -self.npv

    @property
    def undiscounted(self) -> float:
        """Nominal sum of every amount.  A control total, not a decision figure."""
        return self.series.undiscounted(self.ctx.timeline)

    def eac(self) -> float:
        """Equivalent annual cost: ``PV x CRF``, positive for a net cost."""
        return self.pv_of_outflows * self.ctx.timeline.capital_recovery_factor()

    def cost_per_km(self) -> float:
        """Equivalent annual cost per kilometre driven.

        Raises:
            CarbitrageError: when no mileage was supplied, since the figure
                would otherwise divide by zero and read as infinite cost.
        """
        annual_km = self.ctx.usage.annual_km
        if annual_km <= 0:
            raise CarbitrageError(
                "cost_per_km needs a positive Usage(annual_km=...); none was supplied"
            )
        return self.eac() / annual_km

    def breakdown(self) -> dict[Component, float]:
        """Present value per labelled component.  Sums to :attr:`npv`."""
        out = self.series.breakdown(self.ctx.timeline)
        total = sum(out.values())
        if not math.isclose(total, self.npv, rel_tol=1e-9, abs_tol=1e-6):
            raise AssertionError(  # pragma: no cover - guards a corrupted series
                f"breakdown of {self.name!r} sums to {total!r}, not {self.npv!r}"
            )
        return out

    def detail(self) -> tuple[ComponentPV, ...]:
        """One line per constituent flow, in the order they were built."""
        return self.series.detail(self.ctx.timeline)


@dataclass(frozen=True)
class Incremental:
    """The differential cash flow between two alternatives.

    The stream is ``a - b``, so a positive present value means ``a`` is the
    better course of action.
    """

    a: str
    b: str
    amounts: npt.NDArray[np.float64]
    pv: float
    irr: float | None
    irr_note: str
    payback_years: float | None

    @property
    def favours(self) -> str:
        """Which alternative the differential favours."""
        return self.a if self.pv > 0 else self.b


def _irr(amounts: npt.NDArray[np.float64], periods_per_year: int) -> tuple[float | None, str]:
    """The annualised internal rate of a differential stream, where one exists.

    Returns ``(None, reason)`` unless the stream has exactly one sign change and
    exactly one real root above -100 %.  Descartes' rule bounds the number of
    positive roots by the number of sign changes, so a single change guarantees
    the root is unique and the answer is meaningful.
    """
    nonzero = amounts[np.abs(amounts) > 1e-12]
    if nonzero.size == 0:
        return None, "the differential stream is empty"
    signs = np.sign(nonzero)
    changes = int(np.count_nonzero(np.diff(signs) != 0))
    if changes == 0:
        return None, "the differential stream never changes sign, so no IRR exists"
    if changes > 1:
        return None, (
            f"the differential stream changes sign {changes} times, so the IRR is not unique"
        )
    roots = np.roots(amounts[::-1])
    real = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
    if not real:
        return None, "no real root above -100 % exists"
    per_period = 1.0 / max(real) - 1.0
    return float((1.0 + per_period) ** periods_per_year - 1.0), "single sign change; root unique"


def _payback_years(amounts: npt.NDArray[np.float64], periods_per_year: int) -> float | None:
    """First period at which the cumulative differential turns non-negative.

    Undiscounted and therefore a **secondary** criterion only: it ignores both
    the time value of money and everything that happens after the payback point.
    """
    cumulative = np.cumsum(amounts)
    if cumulative[0] >= 0:
        return 0.0
    crossings = np.flatnonzero(cumulative >= 0)
    if crossings.size == 0:
        return None
    return float(crossings[0]) / periods_per_year
