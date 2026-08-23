"""Plots for the analyses that are hard to read as tables.

Requires the ``viz`` extra::

    pip install "carbitrage[viz]"

Every function takes an existing axis or creates one, returns it, and never
calls ``show()`` or ``savefig()``: what to do with the figure is the caller's
decision, not the library's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import CarbitrageError

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from ..comparison import ComparisonResult
    from ..sensitivity import MonteCarlo, OneWayGrid, Tornado

__all__ = [
    "difference_plot",
    "npv_density_plot",
    "one_way_plot",
    "ranking_plot",
    "tornado_plot",
]


def _axes(ax: Any, **kwargs: Any) -> Any:
    if ax is not None:
        return ax
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting needs matplotlib; install carbitrage with the 'viz' extra"
        ) from exc
    _, created = plt.subplots(**kwargs)
    return created


def tornado_plot(tornado: Tornado, ax: Any = None, *, colour: str = "#4C72B0") -> Any:
    """Horizontal bars of each driver's swing, widest at the top.

    Bars are drawn as the *distance* the metric moves from the base case, so the
    reader sees which direction each end of the range pushes the answer.
    """
    ax = _axes(ax, figsize=(8, 0.6 * len(tornado.bars) + 2))
    bars = list(reversed(tornado.bars))
    base = tornado.base_metric
    positions = np.arange(len(bars))
    for i, bar in enumerate(bars):
        low = min(bar.low_metric, bar.high_metric)
        high = max(bar.low_metric, bar.high_metric)
        ax.barh(i, high - low, left=low, color=colour, alpha=0.85, height=0.6)
        left, right = bar.endpoints()
        ax.text(low, i, f" {left} ", ha="right", va="center", fontsize=8)
        ax.text(high, i, f" {right} ", ha="left", va="center", fontsize=8)
    ax.axvline(base, color="black", linewidth=1.2, linestyle="--", label="base case")
    ax.set_yticks(positions)
    ax.set_yticklabels([bar.param for bar in bars])
    ax.set_xlabel("Metric")
    ax.set_title("Drivers ranked by absolute swing")
    ax.legend(loc="lower right", fontsize=8)
    return ax


def one_way_plot(grid: OneWayGrid, ax: Any = None) -> Any:
    """Every alternative's present value against one swept parameter.

    Where the lines cross is where the decision changes, which is the whole
    reason to look at the chart rather than the number.
    """
    ax = _axes(ax, figsize=(8, 5))
    for j, name in enumerate(grid.names):
        ax.plot(grid.values, grid.npv[:, j], marker="o", markersize=3, label=name)
    for value, before, after in grid.winner_changes():
        ax.axvline(value, color="grey", linewidth=0.8, linestyle=":")
        ax.annotate(
            f"{before} to {after}",
            xy=(value, ax.get_ylim()[0]),
            fontsize=7,
            rotation=90,
            va="bottom",
            ha="right",
        )
    ax.set_xlabel(grid.param)
    ax.set_ylabel("Net present value")
    ax.set_title(f"Sensitivity to {grid.param}")
    ax.legend(fontsize=8)
    return ax


def _pair(simulation: MonteCarlo, between: Sequence[str] | None) -> tuple[str, str]:
    """The two alternatives whose difference is being drawn.

    A difference is defined between exactly two things, so a simulation carrying
    more than two is asked which pair rather than guessed at.
    """
    if between is not None:
        names = tuple(between)
        if len(names) != 2:
            raise CarbitrageError(
                f"a difference is drawn between exactly two alternatives, got {list(names)}"
            )
        return names[0], names[1]
    if len(simulation.names) != 2:
        raise CarbitrageError(
            f"this simulation carries {len(simulation.names)} alternatives, so there is no one "
            "difference to draw; name the pair with between=(a, b), or use npv_density_plot to "
            "show all of them at once"
        )
    return simulation.names[0], simulation.names[1]


def difference_plot(
    simulation: MonteCarlo,
    ax: Any = None,
    *,
    between: Sequence[str] | None = None,
    bins: int = 40,
    colour: str = "#4C72B0",
) -> Any:
    """The distribution of the *difference* between two alternatives.

    Drawing the difference rather than two overlaid distributions is the point:
    the alternatives share inputs and move together, so the spread of each one
    separately says nothing about how often one beats the other.  Pass
    ``between`` to pick the pair out of a simulation carrying more.
    """
    a, b = _pair(simulation, between)
    values = simulation.difference(a, b)
    ax = _axes(ax, figsize=(8, 5))
    ax.hist(values, bins=bins, color=colour, alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1.2)
    median = simulation.difference_percentiles(a, b, (50,))[50.0]
    ax.axvline(median, color="#C44E52", linewidth=1.2, linestyle="--", label="median")
    probability = simulation.probability(a, b)
    ax.set_xlabel(f"NPV advantage of {a} over {b}")
    ax.set_ylabel("Trials")
    ax.set_title(f"{a} wins in {probability:.1%} of {simulation.n:,} trials")
    ax.legend(fontsize=8)
    return ax


def npv_density_plot(
    simulation: MonteCarlo,
    ax: Any = None,
    *,
    names: Sequence[str] | None = None,
    points: int = 256,
    fill: bool = True,
) -> Any:
    """A smoothed density of each alternative's own present value.

    This is the one chart that scales past two alternatives, and the one to be
    careful with.  Every curve here is *marginal*: it answers how bad a single
    alternative can get in absolute euros, which is a budgeting question the
    difference actively hides.  It does not answer which one to buy.

    Overlap between two curves is not a probability.  The alternatives share an
    energy price and a discount rate, so heavily overlapping curves can still be
    decided four to one once the trials are paired -- reading the overlap by eye
    is the overlapping-error-bars fallacy, and the answer is
    :meth:`~carbitrage.sensitivity.MonteCarlo.win_share` or
    :func:`difference_plot`, not this chart.

    An alternative that nothing sampled reaches has no density to estimate.  It
    is drawn as the spike it is and labelled *(fixed)*, because dropping it
    would read as an oversight when it is really a fact about the run.
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError as exc:  # pragma: no cover - scipy is a hard dependency
        raise ImportError("the density plot needs scipy") from exc

    chosen = tuple(names) if names is not None else simulation.names
    unknown = [name for name in chosen if name not in simulation.npv]
    if unknown:
        raise CarbitrageError(
            f"{unknown[0]!r} is not in this simulation, which covers {list(simulation.names)}"
        )
    ax = _axes(ax, figsize=(8, 5))
    columns = {name: simulation.npv[name] for name in chosen}
    low = min(float(np.min(values)) for values in columns.values())
    high = max(float(np.max(values)) for values in columns.values())
    pad = 0.05 * (high - low) if high > low else 1.0
    grid = np.linspace(low - pad, high + pad, points)

    densities = {
        name: gaussian_kde(values)(grid) for name, values in columns.items() if np.ptp(values) > 0.0
    }
    peak = max((float(np.max(d)) for d in densities.values()), default=1.0)

    for name, values in columns.items():
        density = densities.get(name)
        if density is None:
            fixed = float(values[0])
            ax.plot([fixed, fixed], [0.0, peak], linewidth=1.6, label=f"{name} (fixed)")
            continue
        (line,) = ax.plot(grid, density, linewidth=1.6, label=name)
        if fill:
            ax.fill_between(grid, density, color=line.get_color(), alpha=0.20)
        ax.axvline(float(np.median(values)), color=line.get_color(), linewidth=0.9, linestyle=":")
    ax.set_xlabel("Net present value")
    ax.set_ylabel("Density")
    ax.set_title("Marginal spread of each alternative -- overlap is not a probability")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)
    return ax


def ranking_plot(result: ComparisonResult, ax: Any = None) -> Any:
    """Present value of outflows per alternative, best first."""
    ax = _axes(ax, figsize=(8, 0.6 * len(result.evaluations) + 2))
    order = list(reversed(result.ranking()))
    values = [e.pv_of_outflows for e in order]
    colours = ["#55A868" if e is result.best() else "#4C72B0" for e in order]
    ax.barh([e.name for e in order], values, color=colours, height=0.6)
    ax.set_xlabel("Present value of outflows")
    ax.set_title(result.verdict(), fontsize=9, loc="left", wrap=True)
    return ax
