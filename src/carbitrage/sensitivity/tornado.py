"""Ranking the drivers by how far each one moves the margin."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import CarbitrageError
from ..params import get_param, resolve, scale_param, set_param
from .metrics import Metric, best_margin
from .spec import Range, _pretty

if TYPE_CHECKING:  # pragma: no cover
    from ..comparison import Case

__all__ = [
    "Tornado",
    "TornadoBar",
    "tornado",
]


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
