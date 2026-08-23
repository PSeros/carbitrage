"""The read surface: ranking, breakdown, and the questions you can ask of it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..cashflow import Component, ComponentPV
from ..errors import CarbitrageError
from .evaluation import _DEFAULT_MATERIALITY, Evaluation, Incremental, _irr, _payback_years

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

    import numpy.typing as _npt

    from ..sensitivity import (
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
    from .case import Case

__all__ = [
    "ComparisonResult",
]


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
            raise CarbitrageError(f"materiality must lie in [0, 1), got {self.materiality!r}")
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

    def breakdown_frame(self, *, decimals: int | None = None) -> Any:
        """Every alternative's components side by side, as a pandas DataFrame.

        Rows are components, columns are alternatives in the order they were
        supplied, and a component absent from an alternative reads as ``0``
        rather than as missing data — it is a real zero, not an unknown.  Each
        column sums to that alternative's NPV.

        Args:
            decimals: Round to this many places, ``0`` giving whole currency
                units as integers.  ``None``, the default, keeps full precision;
                rounding is a presentation choice and belongs to the caller.

        Requires the ``frames`` extra.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "breakdown_frame() needs pandas; install carbitrage with the 'frames' extra"
            ) from exc
        columns = {e.name: e.breakdown() for e in self.evaluations}
        present = [c for c in Component if any(c in column for column in columns.values())]
        frame = pd.DataFrame(
            {name: [column.get(c, 0.0) for c in present] for name, column in columns.items()},
            index=pd.Index([c.name for c in present], name="component"),
        )
        if decimals is None:
            return frame
        frame = frame.round(decimals)
        return frame.astype(int) if decimals <= 0 else frame

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
        from ..sensitivity import one_way

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
        from ..sensitivity import two_way

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
        from ..sensitivity import switch_point

        return switch_point(self._require_case("switch_point"), param, between, bounds=bounds)

    def switch_point_report(
        self,
        param: str,
        between: tuple[str, str],
        *,
        bounds: tuple[float, float] | None = None,
    ) -> SwitchPoint | NoSwitchPoint:
        """As :meth:`switch_point`, but explains itself when there is no crossing."""
        from ..sensitivity import switch_point_report

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
        from ..sensitivity import tornado

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
        from ..sensitivity import monte_carlo

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
