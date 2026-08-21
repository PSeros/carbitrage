"""The comparison result: rankings, breakdowns, incremental analysis, reporting.

Two rules govern everything here.

**Rank by net present value, never by absolute value.**  A cheap asset with a
large grant, recurring credits and a strong residual can produce a *positive*
NPV, and an ``abs()``-based ranking silently inverts the order when it does.
Equivalent annual cost and cost per kilometre are affine transforms of the NPV
and inherit the same caveat.

**Every number is traceable.**  If a user cannot ask where a figure comes from
and get a breakdown that sums to it, the library has failed at its actual job,
which is not arithmetic but justification.  :meth:`ComparisonResult.breakdown`
therefore asserts that its components sum to the total.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ..core.cashflow import CashFlowSeries, Component, ComponentPV
from ..domain.context import Context
from ..errors import CarbitrageError

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

    import numpy.typing as _npt

    from ..study.sensitivity import (
        Distribution,
        Metric,
        MonteCarlo,
        NoSwitchPoint,
        OneWayGrid,
        Range,
        SwitchPoint,
        Tornado,
        TwoWayGrid,
    )
    from .comparison import Case

__all__ = ["ComparisonResult", "Evaluation", "Incremental"]

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


@dataclass(frozen=True)
class ComparisonResult:
    """Every alternative evaluated on one timeline, with the tools to read it."""

    evaluations: tuple[Evaluation, ...]
    materiality: float = _DEFAULT_MATERIALITY
    case: Case | None = None
    """The inputs this result came from, when it was built by
    :func:`~carbitrage.engine.comparison.compare`.  Sensitivity analysis needs them in
    order to perturb one parameter and re-run."""

    _by_name: dict[str, Evaluation] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        names = [e.name for e in self.evaluations]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise CarbitrageError(
                f"alternative names must be unique within a comparison; repeated: "
                f"{sorted(duplicates)}"
            )
        if not 0.0 <= self.materiality < 1.0:
            raise CarbitrageError(
                f"materiality must lie in [0, 1), got {self.materiality!r}"
            )
        object.__setattr__(self, "_by_name", {e.name: e for e in self.evaluations})

    # ----------------------------------------------------------------- access

    @property
    def names(self) -> tuple[str, ...]:
        """Alternative names, in the order they were supplied."""
        return tuple(e.name for e in self.evaluations)

    def __getitem__(self, name: str) -> Evaluation:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not in this comparison; available: {list(self._by_name)}"
            ) from None

    def npv(self) -> dict[str, float]:
        """Net present value per alternative.  Negative for a net cost."""
        return {e.name: e.npv for e in self.evaluations}

    def pv_of_outflows(self) -> dict[str, float]:
        """Present value of outflows per alternative, sign-flipped for reading."""
        return {e.name: e.pv_of_outflows for e in self.evaluations}

    def eac(self) -> dict[str, float]:
        """Equivalent annual cost per alternative."""
        return {e.name: e.eac() for e in self.evaluations}

    def cost_per_km(self) -> dict[str, float]:
        """Equivalent annual cost per kilometre, per alternative."""
        return {e.name: e.cost_per_km() for e in self.evaluations}

    def constraints(self) -> dict[str, tuple[str, ...]]:
        """Non-cash obligations per alternative."""
        return {e.name: e.constraints for e in self.evaluations if e.constraints}

    # ---------------------------------------------------------------- ranking

    def ranking(self) -> tuple[Evaluation, ...]:
        """Alternatives ordered best first, by net present value."""
        return tuple(sorted(self.evaluations, key=lambda e: e.npv, reverse=True))

    def best(self) -> Evaluation:
        """The alternative with the highest NPV — equivalently the lowest cost."""
        return self.ranking()[0]

    def margin(self) -> float:
        """NPV lead of the winner over the runner-up.  Zero with one alternative."""
        order = self.ranking()
        if len(order) < 2:
            return 0.0
        return order[0].npv - order[1].npv

    def is_material(self) -> bool:
        """Whether the winner's lead exceeds :attr:`materiality` of its own PV.

        A EUR 200 lead over a six-year horizon is noise, not a decision.  Below
        the threshold the alternatives are indistinguishable on cost, and the
        choice belongs to criteria this library deliberately does not monetise.
        """
        order = self.ranking()
        if len(order) < 2:
            return True
        scale = abs(order[0].npv)
        if scale == 0:
            return self.margin() > 0
        return self.margin() > self.materiality * scale

    def verdict(self) -> str:
        """A one-line statement of the outcome, materiality included."""
        order = self.ranking()
        if len(order) == 1:
            return f"{order[0].name} is the only alternative."
        lead = self.margin()
        if self.is_material():
            return (
                f"{order[0].name} is ahead of {order[1].name} by {lead:,.0f} in present value "
                f"({lead / abs(order[0].npv):.1%} of its own PV): a clear difference."
            )
        return (
            f"{order[0].name} leads {order[1].name} by only {lead:,.0f} in present value, "
            f"within the {self.materiality:.0%} materiality threshold: too close to call on "
            "cost alone."
        )

    # ------------------------------------------------------------- breakdowns

    def breakdown(self, name: str) -> dict[Component, float]:
        """Labelled present-value components of one alternative, summing to its NPV."""
        return self[name].breakdown()

    def detail(self, name: str) -> tuple[ComponentPV, ...]:
        """One line per constituent flow of one alternative."""
        return self[name].detail()

    # ------------------------------------------------------------ incremental

    def incremental(self, a: str, b: str) -> Incremental:
        """The differential cash flow ``a - b``, its PV, its IRR and its payback.

        The internal rate of return is only defined where the differential
        stream changes sign.  Where it does not, or where it changes sign more
        than once and several roots exist, ``irr`` is ``None`` with the reason
        stated in ``irr_note`` — a wrong root is worse than no root.
        """
        left, right = self[a], self[b]
        timeline = left.ctx.timeline
        amounts = left.amounts - right.amounts
        pv = timeline.pv(amounts)
        irr, note = _irr(amounts, timeline.periods_per_year)
        return Incremental(
            a=a,
            b=b,
            amounts=amounts,
            pv=pv,
            irr=irr,
            irr_note=note,
            payback_years=_payback_years(amounts, timeline.periods_per_year),
        )

    # ------------------------------------------------------------- reporting

    def to_rows(self, baseline: str | None = None) -> tuple[dict[str, Any], ...]:
        """One dictionary per alternative, ranked, ready for any renderer.

        Args:
            baseline: Alternative to measure the delta against.  Defaults to the
                worst-ranked, so the deltas read as advantages.
        """
        order = self.ranking()
        base = self[baseline] if baseline is not None else order[-1]
        has_km = all(e.ctx.usage.annual_km > 0 for e in self.evaluations)
        rows: list[dict[str, Any]] = []
        for rank, e in enumerate(order, start=1):
            rows.append(
                {
                    "rank": rank,
                    "alternative": e.name,
                    "npv": e.npv,
                    "pv_of_outflows": e.pv_of_outflows,
                    "eac": e.eac(),
                    "cost_per_km": e.cost_per_km() if has_km else None,
                    "delta_vs_baseline": e.npv - base.npv,
                }
            )
        return tuple(rows)

    def to_markdown(self, baseline: str | None = None, *, decimals: int = 2) -> str:
        """A table a non-programmer can read.  Rounding happens only here."""
        rows = self.to_rows(baseline)
        base = self[baseline] if baseline is not None else self.ranking()[-1]
        has_km = rows[0]["cost_per_km"] is not None
        headers = ["#", "Alternative", "PV of outflows", "EAC / year"]
        if has_km:
            headers.append("Cost / km")
        headers.append(f"vs. {base.name}")

        def money(x: float) -> str:
            return f"{x:,.{decimals}f}"

        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---:"] + ["---"] + ["---:"] * (len(headers) - 2)) + "|",
        ]
        for row in rows:
            cells = [
                str(row["rank"]),
                str(row["alternative"]),
                money(row["pv_of_outflows"]),
                money(row["eac"]),
            ]
            if has_km:
                cells.append(f"{row['cost_per_km']:,.4f}")
            cells.append(money(row["delta_vs_baseline"]))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(self.verdict())
        constraints = self.constraints()
        if constraints:
            lines.append("")
            lines.append("Constraints:")
            for name, items in constraints.items():
                for item in items:
                    lines.append(f"- {name}: {item}")
        return "\n".join(lines)

    def to_frame(self, baseline: str | None = None) -> Any:
        """The same table as a pandas DataFrame.  Requires the ``frames`` extra."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "to_frame() needs pandas; install carbitrage with the 'frames' extra"
            ) from exc
        return pd.DataFrame(list(self.to_rows(baseline)))

    # ----------------------------------------------------------- sensitivity

    def _require_case(self, what: str) -> Case:
        """The inputs this result came from, or a clear error saying why not."""
        if self.case is None:
            raise CarbitrageError(
                f"{what} needs the inputs this result was built from.  Build the comparison with "
                "carbitrage.compare(...) rather than constructing a ComparisonResult directly."
            )
        return self.case

    def one_way(
        self, param: str, values: Sequence[float], *, metric: Metric | None = None
    ) -> OneWayGrid:
        """Sweep one parameter.  See :func:`carbitrage.study.sensitivity.one_way`."""
        # An upward edge into `study`, by design.  These six methods are
        # convenience delegators so that `result.one_way(...)` reads well; keeping
        # the import function-local means `engine` stays importable without
        # `study`, and the reverse edge (sensitivity -> engine) stays type-only.
        from ..study.sensitivity import one_way

        return one_way(self._require_case("one_way"), param, values, metric=metric)

    def two_way(
        self,
        row_param: str,
        row_values: Sequence[float],
        column_param: str,
        column_values: Sequence[float],
        *,
        metric: Metric | None = None,
    ) -> TwoWayGrid:
        """Sweep two parameters.  See :func:`carbitrage.study.sensitivity.two_way`."""
        from ..study.sensitivity import two_way

        return two_way(
            self._require_case("two_way"),
            row_param,
            row_values,
            column_param,
            column_values,
            metric=metric,
        )

    def switch_point(
        self,
        param: str,
        between: tuple[str, str],
        *,
        bounds: tuple[float, float] | None = None,
    ) -> SwitchPoint | None:
        """Solve for the value at which the ranking flips, or ``None``.

        See :func:`carbitrage.study.sensitivity.switch_point`.
        """
        from ..study.sensitivity import switch_point

        return switch_point(
            self._require_case("switch_point"), param, between, bounds=bounds
        )

    def switch_point_report(
        self,
        param: str,
        between: tuple[str, str],
        *,
        bounds: tuple[float, float] | None = None,
    ) -> SwitchPoint | NoSwitchPoint:
        """As :meth:`switch_point`, but explains itself when there is no crossing."""
        from ..study.sensitivity import switch_point_report

        return switch_point_report(
            self._require_case("switch_point"), param, between, bounds=bounds
        )

    def tornado(
        self,
        params: Sequence[str] | Mapping[str, Range],
        *,
        metric: Metric | None = None,
        default_range: Range | None = None,
    ) -> Tornado:
        """Rank drivers by swing.  See :func:`carbitrage.study.sensitivity.tornado`."""
        from ..study.sensitivity import tornado

        return tornado(
            self._require_case("tornado"),
            params,
            metric=metric,
            default_range=default_range,
        )

    def monte_carlo(
        self,
        distributions: Mapping[str, Distribution],
        *,
        between: tuple[str, str],
        n: int = 2_000,
        correlation: _npt.NDArray[np.float64] | Sequence[Sequence[float]] | None = None,
        seed: int | None = None,
    ) -> MonteCarlo:
        """Simulate with uncertain inputs.

        See :func:`carbitrage.study.sensitivity.monte_carlo`.
        """
        from ..study.sensitivity import monte_carlo

        return monte_carlo(
            self._require_case("monte_carlo"),
            distributions,
            between=between,
            n=n,
            correlation=correlation,
            seed=seed,
        )

    def breakdown_markdown(self, name: str, *, decimals: int = 2) -> str:
        """The labelled components of one alternative, with a checked total."""
        lines = ["| Component | PV |", "|---|---:|"]
        total = 0.0
        for line in self.detail(name):
            if line.pv == 0.0:
                continue
            lines.append(f"| {line.description} | {line.pv:,.{decimals}f} |")
            total += line.pv
        lines.append(f"| **Total** | **{total:,.{decimals}f}** |")
        return "\n".join(lines)


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
