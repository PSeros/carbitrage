"""Sampling the joint uncertainty, and how often each option wins."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ..cashflow import Component
from ..errors import CarbitrageError
from ..params import ParamName, name_of, resolve, set_params, spread_of, spreads
from .distributions import Distribution, _norm_cdf

if TYPE_CHECKING:  # pragma: no cover
    from ..comparison import Case

__all__ = [
    "MonteCarlo",
    "monte_carlo",
]

# ---------------------------------------------------------- monte carlo


@dataclass(frozen=True)
class MonteCarlo:
    """Every alternative's present value under one shared set of draws.

    Each trial evaluates *all* the alternatives against the same draw, so the
    columns of :attr:`npv` are paired row by row.  That pairing is the content
    of the simulation.  The alternatives share an energy price, a discount rate
    and a horizon, so their present values move together, and the spread of any
    one of them on its own says nothing about how often it beats another.

    Read the marginals -- :meth:`npv_percentiles` -- for how bad a single
    alternative can get in absolute euros, which is a budgeting question and a
    real one.  Take every claim about *winning* from the paired columns:
    :meth:`probability`, :meth:`win_share`, :meth:`regret`.  Two marginals that
    overlap heavily can still be decided four to one once they are paired, so
    reading overlap as if it were a probability is a mistake the paired columns
    are there to prevent.
    """

    names: tuple[str, ...]
    npv: dict[str, npt.NDArray[np.float64]]
    params: tuple[str, ...]
    draws: npt.NDArray[np.float64]

    components: tuple[Component, ...] = ()
    """The labelled parts the present values decompose into, in the order
    :class:`~carbitrage.cashflow.Component` declares them.  Empty on a
    simulation assembled by hand rather than by :func:`monte_carlo`."""

    breakdown: dict[str, npt.NDArray[np.float64]] = field(default_factory=dict)
    """Per alternative, trials down the rows and :attr:`components` across the
    columns.  Every row sums to that trial's entry in :attr:`npv`, because the
    present value is read off this decomposition rather than beside it."""

    @property
    def n(self) -> int:
        return int(self.npv[self.names[0]].size)

    # -------------------------------------------------------------- lookup

    def _column(self, name: str) -> npt.NDArray[np.float64]:
        try:
            return self.npv[name]
        except KeyError:
            raise CarbitrageError(
                f"{name!r} is not in this simulation, which covers {list(self.names)}"
            ) from None

    def _matrix(self) -> npt.NDArray[np.float64]:
        """Trials down the rows, alternatives across the columns, in name order."""
        return np.column_stack([self.npv[name] for name in self.names])

    # ------------------------------------------------------------ marginal

    def expected_npv(self) -> dict[str, float]:
        """Mean present value per alternative.

        This is the one reading that needs no pairing at all: expectation is
        linear, so ``E[a] - E[b] == E[a - b]`` whatever the correlation.  Under
        risk neutrality the highest of these *is* the decision.
        """
        return {name: float(np.mean(self.npv[name])) for name in self.names}

    def npv_percentiles(
        self, name: str, levels: Sequence[float] = (5, 25, 50, 75, 95)
    ) -> dict[float, float]:
        """Percentiles of one alternative's own present value.

        A statement about that alternative's exposure, not about the comparison:
        nothing here licenses a claim that one option beats another.
        """
        values = np.percentile(self._column(name), levels)
        return {float(level): float(value) for level, value in zip(levels, values, strict=True)}

    # ----------------------------------------------------------- components

    def _parts(self, name: str) -> npt.NDArray[np.float64]:
        """One alternative's component matrix, or a clear word on why there is none."""
        if not self.components:
            raise CarbitrageError(
                "this simulation carries no component breakdown; build it with "
                "monte_carlo(...) rather than constructing a MonteCarlo directly"
            )
        try:
            return self.breakdown[name]
        except KeyError:
            raise CarbitrageError(
                f"{name!r} is not in this simulation, which covers {list(self.names)}"
            ) from None

    def component_npv(self, name: str, component: Component) -> npt.NDArray[np.float64]:
        """One component's present value for one alternative, trial by trial.

        Paired with every other column of the simulation on the same rows, so a
        component and the total it belongs to can be read against each other.
        """
        try:
            j = self.components.index(component)
        except ValueError:
            raise CarbitrageError(
                f"{component!r} appears in no alternative here, which carries "
                f"{[c.name for c in self.components]}"
            ) from None
        return self._parts(name)[:, j]

    def expected_breakdown(self) -> dict[str, dict[Component, float]]:
        """Mean present value per component, per alternative.

        The one component reading that adds up: expectation is linear, so these
        sum down to :meth:`expected_npv` exactly.  No percentile does.
        """
        return {
            name: dict(
                zip(self.components, map(float, self._parts(name).mean(axis=0)), strict=True)
            )
            for name in self.names
        }

    def component_percentiles(
        self, name: str, levels: Sequence[float] = (5, 95)
    ) -> dict[Component, dict[float, float]]:
        """Percentiles of each component's present value, for one alternative.

        Each component read on its own, which is what makes these *not* add up:
        the fifth percentile of a sum is not the sum of the fifth percentiles,
        because the components do not all hit their bad end on the same trial.
        Take the spread of the total from :meth:`npv_percentiles`, which is the
        narrower and the honest one.
        """
        values = np.percentile(self._parts(name), levels, axis=0)
        return {
            component: {float(level): float(values[i, j]) for i, level in enumerate(levels)}
            for j, component in enumerate(self.components)
        }

    def breakdown_frame(
        self, *, levels: Sequence[float] = (5, 95), decimals: int | None = None
    ) -> Any:
        """Every alternative's components under uncertainty, as a pandas DataFrame.

        The probabilistic reading of
        :meth:`~carbitrage.comparison.ComparisonResult.breakdown_frame`: same
        rows, but each cell is a statistic over the trials rather than one
        base-case number.  Rows are components in
        :class:`~carbitrage.cashflow.Component` order with a final ``NPV`` row;
        columns are a two-level index of alternative and statistic, the
        statistics running low to high with the mean where the median would sit.
        A component no draw reaches shows the same figure in every column, which
        is a fact about the run and not a rounding artefact.

        Only the mean row adds up.  The percentile columns do not sum down to
        the ``NPV`` row and are not meant to: totalling the fifth percentiles
        would describe a trial in which every component went wrong at once,
        which is not a trial the simulation drew.  The ``NPV`` row is taken from
        the paired totals, so it is narrower than that sum, and it is the one to
        quote.

        Args:
            levels: Percentiles to report alongside the mean.  Sorted and
                de-duplicated; an empty sequence leaves the mean alone.
            decimals: Round to this many places, ``0`` giving whole currency
                units as integers.  ``None``, the default, keeps full precision.

        Raises:
            CarbitrageError: on a level outside ``[0, 100]``, or on a simulation
                that carries no breakdown.

        Requires the ``frames`` extra.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "breakdown_frame() needs pandas; install carbitrage with the 'frames' extra"
            ) from exc
        statistics = _statistics(levels)
        index = pd.Index([c.name for c in self.components] + ["NPV"], name="component")
        columns: dict[tuple[str, str], npt.NDArray[np.float64]] = {}
        for name in self.names:
            parts = self._parts(name)
            total = self._column(name)
            for label, level in statistics:
                if level is None:
                    values = np.append(parts.mean(axis=0), total.mean())
                else:
                    values = np.append(
                        np.percentile(parts, level, axis=0), np.percentile(total, level)
                    )
                columns[(name, label)] = values
        frame = pd.DataFrame(columns, index=index)
        frame.columns.names = ["alternative", "statistic"]
        if decimals is None:
            return frame
        frame = frame.round(decimals)
        return frame.astype(int) if decimals <= 0 else frame

    # ------------------------------------------------------------ pairwise

    def difference(self, a: str, b: str) -> npt.NDArray[np.float64]:
        """Trial-by-trial advantage of ``a`` over ``b``.  Positive favours ``a``."""
        return self._column(a) - self._column(b)

    def probability(self, a: str, b: str) -> float:
        """Share of trials in which ``a`` has the higher present value."""
        return float(np.mean(self.difference(a, b) > 0.0))

    def difference_percentiles(
        self, a: str, b: str, levels: Sequence[float] = (5, 25, 50, 75, 95)
    ) -> dict[float, float]:
        """Percentiles of the advantage of ``a`` over ``b``."""
        values = np.percentile(self.difference(a, b), levels)
        return {float(level): float(value) for level, value in zip(levels, values, strict=True)}

    def pairwise(self) -> dict[tuple[str, str], float]:
        """``P(i beats j)`` for every ordered pair.

        A diagnostic, not a ranking.  Each single trial orders the alternatives
        totally, but the *majority* relation across trials need not be
        transitive: i can beat j, j beat k and k beat i, which is ordinary
        Condorcet.  Rank by :meth:`expected_npv` or :meth:`regret_percentile`.
        """
        return {(a, b): self.probability(a, b) for a in self.names for b in self.names if a != b}

    # --------------------------------------------------------------- N-way

    def win_share(self) -> dict[str, float]:
        """Share of trials in which each alternative is the best of them all.

        Unit-free, and so blind to magnitude: an alternative that wins sixty per
        cent of trials by twenty euros and loses the rest by three thousand has
        the best win share and is the worst choice.  Supporting evidence, never
        the headline -- that belongs to :meth:`expected_npv`.
        """
        winners = np.argmax(self._matrix(), axis=1)
        counts = np.bincount(winners, minlength=len(self.names))
        return {name: float(counts[j]) / self.n for j, name in enumerate(self.names)}

    def win_share_error(self) -> dict[str, float]:
        """Standard error of each win share, as a share.

        A win share is a binomial proportion over the trials, so it carries
        sampling noise of its own: two alternatives closer together than a
        couple of these are not told apart by this many trials.
        """
        return {
            name: float(np.sqrt(share * (1.0 - share) / self.n))
            for name, share in self.win_share().items()
        }

    def regret(self) -> dict[str, npt.NDArray[np.float64]]:
        """Shortfall against the best alternative, trial by trial.

        Never negative, and zero in the trials the alternative wins: what it
        costs to have committed to this one once the draw is known.
        """
        matrix = self._matrix()
        best = matrix.max(axis=1)
        return {name: best - matrix[:, j] for j, name in enumerate(self.names)}

    def expected_regret(self) -> dict[str, float]:
        """Mean regret per alternative.

        Ranks identically to :meth:`expected_npv` -- ``E[max] `` is the same
        constant for every alternative -- so this adds no ordering, only a
        scale a reader feels: what being wrong costs on average.
        """
        return {name: float(np.mean(values)) for name, values in self.regret().items()}

    def regret_percentile(self, level: float = 95.0) -> dict[str, float]:
        """Regret at one percentile, per alternative.

        The worst regret over sampled trials is set by a single draw and grows
        with ``n``, so an upper percentile stands in for the maximum: same idea,
        an estimator that settles down.
        """
        return {name: float(np.percentile(values, level)) for name, values in self.regret().items()}

    # ------------------------------------------------------- decision rules

    def best_by_expected_value(self) -> str:
        """The alternative with the highest mean present value."""
        expected = self.expected_npv()
        return max(expected, key=lambda name: expected[name])

    def best_by_regret(self, level: float = 95.0) -> str:
        """The alternative whose upper-percentile regret is least bad."""
        worst = self.regret_percentile(level)
        return min(worst, key=lambda name: worst[name])

    def rules_agree(self, level: float = 95.0) -> bool:
        """Whether expected value and percentile regret pick the same alternative."""
        return self.best_by_expected_value() == self.best_by_regret(level)

    # ---------------------------------------------------------- screening

    def never_best(self) -> tuple[str, ...]:
        """Alternatives that win no trial at all.

        Not beaten on average -- beaten everywhere the sample looked, which is
        the honest licence to drop them from the conversation.
        """
        share = self.win_share()
        return tuple(name for name in self.names if share[name] == 0.0)

    def undecided(self, tolerance: float = 0.05) -> tuple[tuple[str, str], ...]:
        """Pairs the simulation cannot tell apart, as ``(a, b)`` with ``a`` before ``b``.

        Where ``P(a beats b)`` sits within ``tolerance`` of a coin flip, cost has
        stopped discriminating, and what the model refuses to monetise -- range,
        charging time, a dead car on a Monday morning -- is what is left to
        decide on.
        """
        out = []
        for i, a in enumerate(self.names):
            for b in self.names[i + 1 :]:
                if abs(self.probability(a, b) - 0.5) <= tolerance:
                    out.append((a, b))
        return tuple(out)

    # ------------------------------------------------------------ reporting

    def describe(self) -> str:
        share = self.win_share()
        error = self.win_share_error()
        ev = self.best_by_expected_value()
        expected = self.expected_npv()
        lines = []
        if len(self.names) == 2:
            a, b = self.names
            q = self.difference_percentiles(a, b, (5, 50, 95))
            lines.append(
                f"{a} beats {b} in {self.probability(a, b):.1%} of {self.n:,} trials.  "
                f"Median advantage {q[50.0]:,.0f}, 5th to 95th percentile "
                f"{q[5.0]:,.0f} to {q[95.0]:,.0f}."
            )
        else:
            lines.append(f"{len(self.names)} alternatives over {self.n:,} trials.")
        lines.append(
            f"Highest expected value: {ev} ({expected[ev]:,.0f}), best in "
            f"{share[ev]:.1%} of trials (+/- {error[ev]:.1%})."
        )
        mr = self.best_by_regret()
        if mr == ev:
            lines.append("Regret at the 95th percentile picks the same alternative.")
        else:
            lines.append(
                f"Regret at the 95th percentile picks {mr} instead.  The two rules "
                "disagree, and that disagreement is the finding."
            )
        return "  ".join(lines)

    def to_markdown(self, *, decimals: int = 0) -> str:
        """Marginal spread and paired outcome per alternative, with both rules."""
        expected = self.expected_npv()
        share = self.win_share()
        error = self.win_share_error()
        worst = self.regret_percentile()
        lines = [
            "| Alternative | Expected | p5 | p95 | Wins | Regret p95 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in self.names:
            q = self.npv_percentiles(name, (5, 95))
            lines.append(
                f"| {name} | {expected[name]:,.{decimals}f} | {q[5.0]:,.{decimals}f} | "
                f"{q[95.0]:,.{decimals}f} | {share[name]:.1%} +/- {error[name]:.1%} | "
                f"{worst[name]:,.{decimals}f} |"
            )
        lines.append("")
        lines.append("The p5 and p95 columns are marginal: each alternative on its own.")
        lines.append("Overlap between them is not a probability -- the Wins column is.")
        lines.append("")
        ev = self.best_by_expected_value()
        mr = self.best_by_regret()
        lines.append(f"Highest expected value: **{ev}** ({expected[ev]:,.{decimals}f}).")
        lines.append(f"Lowest 95th-percentile regret: **{mr}** ({worst[mr]:,.{decimals}f}).")
        if not self.rules_agree():
            lines.append(
                f"The two rules disagree: expected value favours {ev}, regret favours {mr}.  "
                "That disagreement is the finding."
            )
        dropped = self.never_best()
        if dropped:
            lines.append("Never best in any trial, so nothing rests on them:")
            lines.extend(f"- {name}" for name in dropped)
        for a, b in self.undecided():
            lines.append(
                f"{a} and {b} are within a coin flip of each other, so cost does not "
                "decide between them."
            )
        return "\n".join(lines)


def monte_carlo(
    case: Case,
    distributions: Sequence[ParamName] | Mapping[ParamName, Distribution | None] | None = None,
    *,
    between: Sequence[str] | None = None,
    n: int = 2_000,
    correlation: npt.NDArray[np.float64] | Sequence[Sequence[float]] | None = None,
    seed: int | None = None,
) -> MonteCarlo:
    """Simulate the comparison with uncertain inputs.

    Every trial evaluates every alternative asked for against the same draw, so
    the result holds a paired sample rather than one distribution per option.

    Args:
        case: The base case.
        distributions: Omit it to sample every parameter in the case whose
            mark declares a distribution -- the model already states its own
            uncertainty, and listing it again here only invites the two to
            disagree.  Otherwise parameter names, which sample what each one
            declares, or a mapping from name to a distribution.  A mapping value
            of ``None`` also defers to the declaration, which is how some
            parameters take theirs from the model while others are given one
            here.
        between: Omit it to carry every alternative in the case, in the order it
            was declared.  Name two or more to narrow the simulation to those,
            which is worth doing only to keep an irrelevant option out of the
            win shares -- it saves nothing, because a trial evaluates the whole
            case either way.
        n: Number of trials.  A win share is a binomial proportion over these,
            so telling close alternatives apart takes more of them than settling
            a single pairwise comparison does.
        correlation: Correlation matrix over the parameters, in the order they
            appear in ``distributions`` -- tree order when they were not listed,
            which :func:`carbitrage.params.spreads` reports.  Energy prices and
            residual values are not independent, and pretending otherwise
            understates the spread of the difference.  Applied as a Gaussian
            copula, so each marginal keeps its own shape.
        seed: Seed for reproducibility.

    Raises:
        CarbitrageError: on an empty specification, a case that declares nothing
            to sample, a non-positive ``n``, fewer than two alternatives, a name
            in ``between`` the case does not hold, a name left to a declaration
            that cannot be sampled, or a correlation matrix that is not
            symmetric positive definite.
    """
    if distributions is not None and not distributions:
        raise CarbitrageError("monte_carlo needs at least one parameter")
    if n <= 0:
        raise CarbitrageError(f"n must be positive, got {n!r}")
    names = _alternatives(case, between)
    given = _requested(case, distributions)
    parameters = tuple(given)
    sampled = {name: _sampled(case, name, given[name]) for name in parameters}
    for name in parameters:
        resolve(case, name)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, len(parameters)))
    if correlation is not None:
        z = z @ _cholesky(np.asarray(correlation, dtype=np.float64), len(parameters)).T
    u = _norm_cdf(z)

    draws = np.column_stack([sampled[name].ppf(u[:, j]) for j, name in enumerate(parameters)])

    columns = np.empty((n, len(names)), dtype=np.float64)
    seen: dict[str, dict[Component, npt.NDArray[np.float64]]] = {name: {} for name in names}
    for i in range(n):
        trial = set_params(case, dict(zip(parameters, draws[i], strict=True)))
        result = trial.run()
        for j, name in enumerate(names):
            decomposition = result[name].breakdown()
            collected = seen[name]
            for component, value in decomposition.items():
                if component not in collected:
                    collected[component] = np.zeros(n, dtype=np.float64)
                collected[component][i] = value
            columns[i, j] = sum(decomposition.values())

    components = tuple(c for c in Component if any(c in seen[name] for name in names))
    parts = {
        name: np.column_stack(
            [seen[name].get(c, np.zeros(n, dtype=np.float64)) for c in components]
        )
        for name in names
    }
    return MonteCarlo(
        names=names,
        npv={name: columns[:, j].copy() for j, name in enumerate(names)},
        params=tuple(name_of(name) for name in parameters),
        draws=draws,
        components=components,
        breakdown=parts,
    )


def _statistics(levels: Sequence[float]) -> tuple[tuple[str, float | None], ...]:
    """The frame's statistic columns, ordered low to high.

    The mean goes where the median would, between the levels below the halfway
    mark and those above it, so a reader running along a row meets the figures
    in the order their magnitudes usually come.
    """
    ordered = sorted(dict.fromkeys(float(level) for level in levels))
    for level in ordered:
        if not 0.0 <= level <= 100.0:
            raise CarbitrageError(f"percentile levels lie in [0, 100], got {level!r}")
    lower = [(f"p{level:g}", level) for level in ordered if level < 50.0]
    upper = [(f"p{level:g}", level) for level in ordered if level >= 50.0]
    middle: tuple[str, float | None] = ("mean", None)
    return (*lower, middle, *upper)


def _alternatives(case: Case, between: Sequence[str] | None) -> tuple[str, ...]:
    """Which alternatives the simulation carries, in the order it will report them.

    Naming none carries them all: the trial evaluates the whole case anyway, and
    a simulation that quietly dropped every option but two would make the reader
    ask for the comparison twice.
    """
    available = tuple(alt.name for alt in case.alternatives)
    if between is None:
        if len(available) < 2:
            raise CarbitrageError(
                f"a simulation compares alternatives, and this case holds {len(available)}"
            )
        return available
    names = tuple(between)
    if len(names) < 2:
        raise CarbitrageError(f"between needs at least two alternatives, got {list(names)}")
    for i, name in enumerate(names):
        if name in names[:i]:
            raise CarbitrageError(f"between names {name!r} twice")
        if name not in available:
            raise CarbitrageError(f"{name!r} is not an alternative in this case: {list(available)}")
    return names


def _requested(
    case: Case,
    distributions: Sequence[ParamName] | Mapping[ParamName, Distribution | None] | None,
) -> dict[ParamName, Distribution | None]:
    """What to sample, as a name to distribution-or-defer mapping.

    An omitted specification is read off the case itself: every mark declaring a
    distribution, in tree order.  Marks declaring a range are passed over rather
    than refused, because the caller named nothing and so asked nothing of them.
    """
    if distributions is None:
        declared: dict[ParamName, Distribution | None] = {
            label: found
            for label, found in spreads(case).items()
            if isinstance(found, Distribution)
        }
        if not declared:
            raise CarbitrageError(
                "no parameter in this case declares a distribution, so there is nothing to "
                "sample; declare one with Uncertain(value, label, spread) or name the "
                "distributions here"
            )
        return declared
    if isinstance(distributions, Mapping):
        return dict(distributions)
    return dict.fromkeys(distributions)


def _sampled(case: Case, name: ParamName, given: Distribution | None) -> Distribution:
    """The distribution to sample for one parameter, given or declared.

    A declared :class:`Range` is refused rather than widened into a uniform: a
    range says where a value lies, a simulation needs to know how likely each
    value in it is, and quietly supplying the second from the first would put a
    modelling assumption nobody made into the answer.
    """
    if given is not None:
        return given
    declared = spread_of(case, name)
    if isinstance(declared, Distribution):
        return declared
    shown = name_of(name)
    if declared is None:
        raise CarbitrageError(
            f"{shown} declares no spread, so there is nothing to sample; give it a distribution"
        )
    raise CarbitrageError(
        f"{shown} declares {declared!r}, which says where the value lies but not how likely each "
        "value in it is; give it a distribution"
    )


def _cholesky(matrix: npt.NDArray[np.float64], k: int) -> npt.NDArray[np.float64]:
    if matrix.shape != (k, k):
        raise CarbitrageError(
            f"correlation matrix must be {k}x{k} to match the parameters, got {matrix.shape}"
        )
    if not np.allclose(matrix, matrix.T):
        raise CarbitrageError("correlation matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0):
        raise CarbitrageError("correlation matrix must have ones on the diagonal")
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise CarbitrageError(
            "correlation matrix is not positive definite, so no set of correlated draws can "
            "reproduce it"
        ) from exc
