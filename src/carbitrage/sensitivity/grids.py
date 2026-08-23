"""Sweeping one parameter, or two against each other."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..errors import CarbitrageError
from ..params import set_param
from .metrics import Metric, best_margin
from .spec import _pretty

if TYPE_CHECKING:  # pragma: no cover
    from ..comparison import Case

__all__ = [
    "OneWayGrid",
    "TwoWayGrid",
    "one_way",
    "two_way",
]

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
