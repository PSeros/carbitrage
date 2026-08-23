"""The number a sensitivity run tracks.

Every entry point in this scope takes a metric: a function of a
`ComparisonResult` returning one float.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..comparison import ComparisonResult

__all__ = [
    "Metric",
    "advantage",
    "best_margin",
    "npv_of",
]

Metric = Callable[["ComparisonResult"], float]

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
