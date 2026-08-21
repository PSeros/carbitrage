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

if TYPE_CHECKING:  # pragma: no cover
    from ..engine.result import ComparisonResult
    from ..study.sensitivity import MonteCarlo, OneWayGrid, Tornado

__all__ = ["monte_carlo_plot", "one_way_plot", "ranking_plot", "tornado_plot"]


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


def monte_carlo_plot(simulation: MonteCarlo, ax: Any = None, *, bins: int = 40) -> Any:
    """The distribution of the *difference* between two alternatives.

    Plotting the difference rather than two overlaid distributions is the point:
    the alternatives share inputs, so the spread of each one separately says
    nothing about how often one beats the other.
    """
    ax = _axes(ax, figsize=(8, 5))
    ax.hist(simulation.differences, bins=bins, color="#4C72B0", alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1.2)
    median = simulation.percentiles((50,))[50.0]
    ax.axvline(median, color="#C44E52", linewidth=1.2, linestyle="--", label="median")
    probability = simulation.probability_a_beats_b()
    ax.set_xlabel(f"NPV advantage of {simulation.a} over {simulation.b}")
    ax.set_ylabel("Trials")
    ax.set_title(f"{simulation.a} wins in {probability:.1%} of {simulation.n:,} trials")
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
