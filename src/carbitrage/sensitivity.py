"""Sensitivity analysis: grids, switch points, tornados and Monte Carlo.

The headline feature is :func:`switch_point`.  A comparison that says "the
electric car wins by EUR 1,930" is far less useful than one that says "the
electric car wins above 9,400 km a year, and below that the incumbent does",
because the second tells the reader which side of the question they are on.

Everything here re-runs the full cash-flow engine.  The reference workbook's own
sensitivity sheets use an annual closed-form approximation and note that it
differs from its monthly grid by a few hundred euros; reproducing that
approximation would make these results disagree with the library's own present
values, so the layout is shared and the arithmetic is not.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from .errors import CarbitrageError
from .params import get_param, resolve, scale_param, set_param, set_params

if TYPE_CHECKING:  # pragma: no cover
    from .analysis import Case
    from .result import ComparisonResult

__all__ = [
    "Distribution",
    "LogNormal",
    "MonteCarlo",
    "Normal",
    "OneWayGrid",
    "Range",
    "SwitchPoint",
    "Tornado",
    "TornadoBar",
    "Triangular",
    "TwoWayGrid",
    "Uniform",
    "advantage",
    "best_margin",
    "monte_carlo",
    "npv_of",
    "one_way",
    "switch_point",
    "tornado",
    "two_way",
]

Metric = Callable[["ComparisonResult"], float]
"""A scalar read off a comparison, used as the value plotted or solved for."""


def _pretty(value: float) -> str:
    """Format a parameter value for a human, never in scientific notation.

    Parameter values here span mileages in the tens of thousands and rates in the
    hundredths, so a single format specifier cannot serve both.
    """
    if value != value or value in (float("inf"), float("-inf")):
        return str(value)
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.2f}"
    if magnitude >= 1:
        return f"{value:,.4g}"
    return f"{value:.6g}"


# ---------------------------------------------------------------- metrics


def advantage(a: str, b: str) -> Metric:
    """The present-value advantage of ``a`` over ``b``.  Positive favours ``a``."""

    def metric(result: ComparisonResult) -> float:
        return result[a].npv - result[b].npv

    return metric


def npv_of(name: str) -> Metric:
    """The net present value of one alternative."""

    def metric(result: ComparisonResult) -> float:
        return result[name].npv

    return metric


def best_margin() -> Metric:
    """The winner's lead over the runner-up, whoever they turn out to be."""

    def metric(result: ComparisonResult) -> float:
        return result.margin()

    return metric


# ------------------------------------------------------------------ grids


@dataclass(frozen=True)
class OneWayGrid:
    """One parameter swept over a range of values."""

    param: str
    values: tuple[float, ...]
    names: tuple[str, ...]
    npv: npt.NDArray[np.float64]
    """Shape ``(len(values), len(names))``: net present value per point."""
    metric: tuple[float, ...]
    winners: tuple[str, ...]

    def winner_changes(self) -> tuple[tuple[float, str, str], ...]:
        """Where the ranking flips, as ``(value, from, to)`` triples."""
        out: list[tuple[float, str, str]] = []
        for i in range(1, len(self.winners)):
            if self.winners[i] != self.winners[i - 1]:
                out.append((self.values[i], self.winners[i - 1], self.winners[i]))
        return tuple(out)

    def to_markdown(self, *, decimals: int = 2) -> str:
        """The grid as a table, one row per value."""
        header = f"| {self.param} | " + " | ".join(self.names) + " | winner |"
        rule = "|---:|" + "---:|" * len(self.names) + "---|"
        lines = [header, rule]
        for i, value in enumerate(self.values):
            cells = " | ".join(f"{self.npv[i, j]:,.{decimals}f}" for j in range(len(self.names)))
            lines.append(f"| {_pretty(value)} | {cells} | {self.winners[i]} |")
        return "\n".join(lines)


@dataclass(frozen=True)
class TwoWayGrid:
    """Two parameters swept against each other, matching the workbook's layout."""

    row_param: str
    row_values: tuple[float, ...]
    column_param: str
    column_values: tuple[float, ...]
    values: npt.NDArray[np.float64]
    """Shape ``(len(row_values), len(column_values))``."""
    winners: tuple[tuple[str, ...], ...]

    def to_markdown(self, *, decimals: int = 0) -> str:
        header = (
            f"| {self.row_param} \\ {self.column_param} | "
            + " | ".join(_pretty(v) for v in self.column_values)
            + " |"
        )
        rule = "|---:|" + "---:|" * len(self.column_values)
        lines = [header, rule]
        for i, row in enumerate(self.row_values):
            cells = " | ".join(
                f"{self.values[i, j]:,.{decimals}f}" for j in range(len(self.column_values))
            )
            lines.append(f"| {_pretty(row)} | {cells} |")
        return "\n".join(lines)


def one_way(
    case: Case,
    param: str,
    values: Sequence[float],
    *,
    metric: Metric | None = None,
) -> OneWayGrid:
    """Sweep one parameter and record every alternative at every value.

    Args:
        case: The base case.
        param: Alias or dotted path of the parameter to vary.
        values: The values to evaluate.
        metric: Scalar recorded alongside the present values.  Defaults to the
            winner's margin over the runner-up.
    """
    if not values:
        raise CarbitrageError("one_way needs at least one value")
    read = best_margin() if metric is None else metric
    names = tuple(alt.name for alt in case.alternatives)
    npv = np.zeros((len(values), len(names)), dtype=np.float64)
    metrics: list[float] = []
    winners: list[str] = []
    for i, value in enumerate(values):
        result = set_param(case, param, value).run()
        for j, name in enumerate(names):
            npv[i, j] = result[name].npv
        metrics.append(read(result))
        winners.append(result.best().name)
    return OneWayGrid(
        param=param,
        values=tuple(float(v) for v in values),
        names=names,
        npv=npv,
        metric=tuple(metrics),
        winners=tuple(winners),
    )


def two_way(
    case: Case,
    row_param: str,
    row_values: Sequence[float],
    column_param: str,
    column_values: Sequence[float],
    *,
    metric: Metric | None = None,
) -> TwoWayGrid:
    """Sweep two parameters against each other.

    The reference workbook's tables 1 and 2 have this shape: mileage down the
    side, autogas price or repair cost across the top, and the present-value
    advantage of acting now in each cell.
    """
    if not row_values or not column_values:
        raise CarbitrageError("two_way needs at least one value on each axis")
    read = best_margin() if metric is None else metric
    grid = np.zeros((len(row_values), len(column_values)), dtype=np.float64)
    winners: list[tuple[str, ...]] = []
    for i, row in enumerate(row_values):
        row_winners: list[str] = []
        with_row = set_param(case, row_param, row)
        for j, column in enumerate(column_values):
            result = set_param(with_row, column_param, column).run()
            grid[i, j] = read(result)
            row_winners.append(result.best().name)
        winners.append(tuple(row_winners))
    return TwoWayGrid(
        row_param=row_param,
        row_values=tuple(float(v) for v in row_values),
        column_param=column_param,
        column_values=tuple(float(v) for v in column_values),
        values=grid,
        winners=tuple(winners),
    )


# ---------------------------------------------------------- switch points


@dataclass(frozen=True)
class SwitchPoint:
    """The parameter value at which the ranking between two alternatives flips."""

    param: str
    value: float
    between: tuple[str, str]
    base_value: float
    favoured_below: str
    favoured_above: str

    def describe(self) -> str:
        """A sentence a non-programmer can act on."""
        return (
            f"{self.param} switches the answer at {_pretty(self.value)} "
            f"(base case {_pretty(self.base_value)}): below it {self.favoured_below} wins, "
            f"above it {self.favoured_above} wins."
        )


@dataclass(frozen=True)
class NoSwitchPoint:
    """Why no crossing exists, so the caller is told rather than left guessing."""

    param: str
    between: tuple[str, str]
    reason: str
    searched: tuple[float, float]

    def describe(self) -> str:
        low, high = self.searched
        return (
            f"No switch point for {self.param} between {_pretty(low)} and {_pretty(high)}: "
            f"{self.reason}"
        )


def switch_point(
    case: Case,
    param: str,
    between: tuple[str, str],
    *,
    bounds: tuple[float, float] | None = None,
    samples: int = 33,
    tolerance: float = 1e-8,
) -> SwitchPoint | None:
    """Solve for the parameter value at which ``between`` swaps places.

    The differential ``npv[a] - npv[b]`` is a continuous function of most
    parameters, so a sign change can be located exactly rather than guessed from
    a grid.  The domain is scanned first to find a bracket, then Brent's method
    refines it.

    Args:
        case: The base case.
        param: Alias or dotted path of the parameter to solve for.
        between: The two alternatives whose order should flip.
        bounds: Domain to search.  Defaults to a decade either side of the base
            value, which covers the plausible range of most inputs.
        samples: Points used to find a sign change before refining.
        tolerance: Convergence tolerance passed to the root finder.

    Returns:
        A :class:`SwitchPoint`, or ``None`` when no crossing exists in the
        domain.  Use :func:`switch_point_report` when the reason matters.
    """
    found = switch_point_report(
        case, param, between, bounds=bounds, samples=samples, tolerance=tolerance
    )
    return found if isinstance(found, SwitchPoint) else None


def switch_point_report(
    case: Case,
    param: str,
    between: tuple[str, str],
    *,
    bounds: tuple[float, float] | None = None,
    samples: int = 33,
    tolerance: float = 1e-8,
) -> SwitchPoint | NoSwitchPoint:
    """As :func:`switch_point`, but explains itself when there is no crossing."""
    a, b = between
    base = get_param(case, param)
    low, high = _default_bounds(base) if bounds is None else bounds
    if low >= high:
        raise CarbitrageError(f"bounds must be increasing, got {(low, high)!r}")

    def differential(x: float) -> float:
        result = set_param(case, param, x).run()
        return result[a].npv - result[b].npv

    xs, ys = _scan(differential, low, high, samples)
    if len(xs) < 2:
        return NoSwitchPoint(
            param, between, "the parameter is not valid anywhere in this domain", (low, high)
        )
    bracket = _first_sign_change(xs, ys)
    if bracket is None:
        direction = "always" if ys[0] > 0 else "never"
        return NoSwitchPoint(
            param,
            between,
            f"{a} {direction} beats {b} across the whole domain, so the ranking never flips",
            (low, high),
        )
    lo, hi = bracket
    root = float(brentq(differential, lo, hi, xtol=tolerance))
    below, above = (b, a) if differential(lo) < 0 else (a, b)
    return SwitchPoint(
        param=param,
        value=root,
        between=between,
        base_value=base,
        favoured_below=below,
        favoured_above=above,
    )


def _default_bounds(base: float) -> tuple[float, float]:
    """A decade either side of the base value, or the unit interval around zero."""
    if base > 0:
        return base / 10.0, base * 10.0
    if base < 0:
        return base * 10.0, base / 10.0
    return -1.0, 1.0


def _scan(
    f: Callable[[float], float], low: float, high: float, samples: int
) -> tuple[list[float], list[float]]:
    """Evaluate ``f`` across the domain, skipping values it rejects.

    Parameters have domains — a share cannot exceed 1, a depreciation rate
    cannot reach 100 % — and the model raises rather than returning nonsense
    outside them.  Those points are dropped instead of aborting the search.
    """
    xs: list[float] = []
    ys: list[float] = []
    for x in np.linspace(low, high, samples):
        try:
            y = f(float(x))
        except CarbitrageError:
            continue
        if math.isfinite(y):
            xs.append(float(x))
            ys.append(y)
    return xs, ys


def _first_sign_change(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    for i in range(1, len(xs)):
        if ys[i - 1] == 0.0:
            return xs[max(i - 2, 0)], xs[i - 1]
        if ys[i - 1] * ys[i] < 0:
            return xs[i - 1], xs[i]
    return None


# --------------------------------------------------------------- tornado


@dataclass(frozen=True)
class Range:
    """A plausible range for one parameter.

    Either absolute (``low``/``high`` values) or relative (multiples of the base
    value).  The relative form is what makes an alias covering several unequal
    parameters — ``residual_rate`` across three different vehicles — still
    meaningful: everything scales together and the modelled differences survive.
    """

    low: float
    high: float
    relative: bool = False

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise CarbitrageError(f"range low {self.low!r} exceeds high {self.high!r}")


@dataclass(frozen=True)
class TornadoBar:
    """One parameter's contribution to the spread of the metric."""

    param: str
    low_value: float
    high_value: float
    low_metric: float
    high_metric: float
    relative: bool = False
    """Whether the range was stated as multiples of the base value.  An alias
    covering several parameters that legitimately differ has no single base
    value, so its endpoints are reported as multipliers."""

    low_factor: float = 1.0
    high_factor: float = 1.0

    def endpoints(self) -> tuple[str, str]:
        """The range endpoints, formatted for display.

        Relative ranges print as multipliers, because an alias spanning several
        unequal parameters has no single base value to multiply out.
        """
        if self.relative or not math.isfinite(self.low_value):
            return f"x{self.low_factor:,.3g}", f"x{self.high_factor:,.3g}"
        return _pretty(self.low_value), _pretty(self.high_value)

    @property
    def swing(self) -> float:
        """Absolute spread of the metric over the parameter's range.

        Absolute rather than elastic: the ranges differ in width and in units,
        and an elasticity would silently rank a parameter with a wide plausible
        range alongside one with a narrow one.
        """
        return abs(self.high_metric - self.low_metric)


@dataclass(frozen=True)
class Tornado:
    """Drivers ranked by how far they move the metric, widest first."""

    base_metric: float
    bars: tuple[TornadoBar, ...]

    def to_markdown(self, *, decimals: int = 0) -> str:
        lines = [
            "| Driver | Low | High | Metric low | Metric high | Swing |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for bar in self.bars:
            low, high = bar.endpoints()
            lines.append(
                f"| {bar.param} | {low} | {high} | "
                f"{bar.low_metric:,.{decimals}f} | {bar.high_metric:,.{decimals}f} | "
                f"{bar.swing:,.{decimals}f} |"
            )
        lines.append("")
        lines.append(f"Base case metric: {self.base_metric:,.{decimals}f}")
        return "\n".join(lines)


def tornado(
    case: Case,
    params: Sequence[str] | Mapping[str, Range],
    *,
    metric: Metric | None = None,
    default_range: Range | None = None,
) -> Tornado:
    """Rank parameters by the swing they produce, one at a time from the base case.

    Args:
        case: The base case.
        params: Either parameter names, which then use ``default_range``, or a
            mapping from name to an explicit :class:`Range`.
        metric: What to measure.  Defaults to the winner's margin over the
            runner-up, which is the decision-relevant quantity: a driver that
            moves every alternative equally changes no decision.
        default_range: Applied to any parameter given without one.  Defaults to
            plus or minus a quarter of the base value.
    """
    read = best_margin() if metric is None else metric
    span_default = Range(0.75, 1.25, relative=True) if default_range is None else default_range
    ranges: dict[str, Range] = (
        dict(params) if isinstance(params, Mapping) else dict.fromkeys(params, span_default)
    )
    if not ranges:
        raise CarbitrageError("tornado needs at least one parameter")

    base_metric = read(case.run())
    bars: list[TornadoBar] = []
    for name, span in ranges.items():
        resolve(case, name)  # fail early on an unknown name
        if span.relative:
            low_case = scale_param(case, name, span.low)
            high_case = scale_param(case, name, span.high)
            base = _base_or_nan(case, name)
            low_value, high_value = base * span.low, base * span.high
        else:
            low_case = set_param(case, name, span.low)
            high_case = set_param(case, name, span.high)
            low_value, high_value = span.low, span.high
        bars.append(
            TornadoBar(
                param=name,
                low_value=low_value,
                high_value=high_value,
                low_metric=read(low_case.run()),
                high_metric=read(high_case.run()),
                relative=span.relative,
                low_factor=span.low if span.relative else 1.0,
                high_factor=span.high if span.relative else 1.0,
            )
        )
    bars.sort(key=lambda bar: bar.swing, reverse=True)
    return Tornado(base_metric=base_metric, bars=tuple(bars))


def _base_or_nan(case: Case, name: str) -> float:
    """The base value, or NaN when an alias covers parameters that disagree."""
    try:
        return get_param(case, name)
    except CarbitrageError:
        return math.nan


# ----------------------------------------------------------- distributions


class Distribution(ABC):
    """A marginal distribution, sampled through its inverse CDF.

    Sampling by inverse transform is what lets correlated draws be produced with
    a Gaussian copula: the dependence structure is imposed on uniforms, and each
    marginal keeps its own shape.
    """

    @abstractmethod
    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Inverse CDF evaluated on uniforms in (0, 1)."""

    @abstractmethod
    def mean(self) -> float:
        """The distribution's mean, used to report the base of a run."""


@dataclass(frozen=True)
class Normal(Distribution):
    """Gaussian.  Symmetric, and unbounded in both directions."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise CarbitrageError(f"sigma must not be negative, got {self.sigma!r}")

    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.mu + self.sigma * _norm_ppf(u)

    def mean(self) -> float:
        return self.mu


@dataclass(frozen=True)
class LogNormal(Distribution):
    """Log-normal.  Strictly positive, which suits prices and residual values."""

    mu_log: float
    sigma_log: float

    def __post_init__(self) -> None:
        if self.sigma_log < 0:
            raise CarbitrageError(f"sigma_log must not be negative, got {self.sigma_log!r}")

    @classmethod
    def from_mean_cv(cls, mean: float, cv: float) -> LogNormal:
        """Parameterise by arithmetic mean and coefficient of variation."""
        if mean <= 0:
            raise CarbitrageError(f"mean must be positive, got {mean!r}")
        sigma_sq = math.log(1.0 + cv * cv)
        return cls(mu_log=math.log(mean) - sigma_sq / 2.0, sigma_log=math.sqrt(sigma_sq))

    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.exp(self.mu_log + self.sigma_log * _norm_ppf(u))

    def mean(self) -> float:
        return math.exp(self.mu_log + self.sigma_log**2 / 2.0)


@dataclass(frozen=True)
class Uniform(Distribution):
    """Uniform between two bounds.  The honest choice when only a range is known."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise CarbitrageError(f"low must be below high, got {(self.low, self.high)!r}")

    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.low + (self.high - self.low) * u

    def mean(self) -> float:
        return (self.low + self.high) / 2.0


@dataclass(frozen=True)
class Triangular(Distribution):
    """Triangular: a minimum, a most likely value and a maximum.

    The usual choice when an expert can state a best guess and two extremes but
    nothing about the shape between them.
    """

    low: float
    mode: float
    high: float

    def __post_init__(self) -> None:
        if not self.low <= self.mode <= self.high or self.low == self.high:
            raise CarbitrageError(
                f"require low <= mode <= high with low < high, got "
                f"{(self.low, self.mode, self.high)!r}"
            )

    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        span = self.high - self.low
        pivot = (self.mode - self.low) / span
        lower = self.low + np.sqrt(u * span * (self.mode - self.low))
        upper = self.high - np.sqrt((1.0 - u) * span * (self.high - self.mode))
        return np.where(u < pivot, lower, upper)

    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3.0


def _norm_ppf(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Inverse standard-normal CDF, via the Beasley-Springer-Moro algorithm.

    Implemented here rather than imported so that the core stays on numpy, with
    scipy reserved for root finding.  Accuracy is better than 1e-6 across the
    range, which is orders of magnitude finer than the input uncertainty any of
    these models carries.
    """
    a = (2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637)
    b = (-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833)
    c = (
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    )
    u = np.clip(u, 1e-15, 1.0 - 1e-15)
    y = u - 0.5
    central = np.abs(y) < 0.42

    r = np.where(central, y * y, 0.0)
    numerator = ((a[3] * r + a[2]) * r + a[1]) * r + a[0]
    denominator = (((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0
    inner = y * numerator / denominator

    tail = np.where(y > 0.0, 1.0 - u, u)
    s = np.log(-np.log(np.clip(tail, 1e-300, 1.0 - 1e-16)))
    poly = np.full_like(s, c[-1])
    for coefficient in reversed(c[:-1]):
        poly = poly * s + coefficient
    outer = np.where(y > 0.0, poly, -poly)

    return np.asarray(np.where(central, inner, outer), dtype=np.float64)


# ---------------------------------------------------------- monte carlo


@dataclass(frozen=True)
class MonteCarlo:
    """A simulated distribution of the *difference* between two alternatives.

    Reporting the difference rather than each alternative separately is the
    whole point.  Two alternatives that share an energy price move together, and
    the spread of each one on its own says nothing about how likely it is that
    one beats the other.
    """

    a: str
    b: str
    differences: npt.NDArray[np.float64]
    npv: dict[str, npt.NDArray[np.float64]]
    params: tuple[str, ...]
    draws: npt.NDArray[np.float64]

    @property
    def n(self) -> int:
        return int(self.differences.size)

    def probability_a_beats_b(self) -> float:
        """Share of trials in which ``a`` has the higher net present value."""
        return float(np.mean(self.differences > 0.0))

    def mean_difference(self) -> float:
        return float(np.mean(self.differences))

    def percentiles(
        self, levels: Sequence[float] = (5, 25, 50, 75, 95)
    ) -> dict[float, float]:
        """Percentiles of the difference."""
        values = np.percentile(self.differences, levels)
        return {float(level): float(value) for level, value in zip(levels, values, strict=True)}

    def describe(self) -> str:
        p = self.probability_a_beats_b()
        q = self.percentiles((5, 50, 95))
        return (
            f"{self.a} beats {self.b} in {p:.1%} of {self.n:,} trials.  "
            f"Median advantage {q[50.0]:,.0f}, 5th to 95th percentile "
            f"{q[5.0]:,.0f} to {q[95.0]:,.0f}."
        )


def monte_carlo(
    case: Case,
    distributions: Mapping[str, Distribution],
    *,
    between: tuple[str, str],
    n: int = 2_000,
    correlation: npt.NDArray[np.float64] | Sequence[Sequence[float]] | None = None,
    seed: int | None = None,
) -> MonteCarlo:
    """Simulate the comparison with uncertain inputs.

    Args:
        case: The base case.
        distributions: One distribution per parameter name.
        between: The two alternatives whose difference is reported.
        n: Number of trials.
        correlation: Correlation matrix over the parameters, in the order they
            appear in ``distributions``.  Energy prices and residual values are
            not independent, and pretending otherwise understates the spread of
            the difference.  Applied as a Gaussian copula, so each marginal
            keeps its own shape.
        seed: Seed for reproducibility.

    Raises:
        CarbitrageError: on an empty specification, a non-positive ``n``, or a
            correlation matrix that is not symmetric positive definite.
    """
    if not distributions:
        raise CarbitrageError("monte_carlo needs at least one distribution")
    if n <= 0:
        raise CarbitrageError(f"n must be positive, got {n!r}")
    a, b = between
    names = tuple(distributions)
    for name in names:
        resolve(case, name)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, len(names)))
    if correlation is not None:
        z = z @ _cholesky(np.asarray(correlation, dtype=np.float64), len(names)).T
    u = _norm_cdf(z)

    draws = np.column_stack(
        [distributions[name].ppf(u[:, j]) for j, name in enumerate(names)]
    )

    npv: dict[str, list[float]] = {name: [] for name in (a, b)}
    differences = np.empty(n, dtype=np.float64)
    for i in range(n):
        trial = set_params(case, dict(zip(names, draws[i], strict=True)))
        result = trial.run()
        npv_a, npv_b = result[a].npv, result[b].npv
        npv[a].append(npv_a)
        npv[b].append(npv_b)
        differences[i] = npv_a - npv_b
    return MonteCarlo(
        a=a,
        b=b,
        differences=differences,
        npv={name: np.asarray(values, dtype=np.float64) for name, values in npv.items()},
        params=names,
        draws=draws,
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


def _norm_cdf(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Standard-normal CDF, vectorised over an array."""
    return np.asarray(0.5 * (1.0 + _erf(x / math.sqrt(2.0))), dtype=np.float64)


_erf = np.vectorize(math.erf, otypes=[np.float64])
