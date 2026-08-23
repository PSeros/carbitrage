"""Solving for the value at which the ranking flips.

This is the answer the library exists to give: not "which option wins" but
"on which side of the question are you".
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from ..errors import CarbitrageError
from ..params import ParamName, get_param, name_of, set_param, spread_of
from .spec import _band, _pretty

if TYPE_CHECKING:  # pragma: no cover
    from ..comparison import Case

__all__ = [
    "NoSwitchPoint",
    "SwitchPoint",
    "switch_point",
    "switch_point_report",
]

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
    band: tuple[float, float] | None = None
    """What the parameter's declared spread calls plausible, or ``None`` when it
    declares none.  The search is never confined to it: whether the ranking can
    flip at all, and whether it flips within reach, are different findings, and
    collapsing them into one would lose the first."""

    @property
    def is_plausible(self) -> bool | None:
        """Whether the crossing lies inside the declared band.  ``None`` if undeclared."""
        if self.band is None:
            return None
        return self.band[0] <= self.value <= self.band[1]

    def describe(self) -> str:
        """A sentence a non-programmer can act on."""
        sentence = (
            f"{self.param} switches the answer at {_pretty(self.value)} "
            f"(base case {_pretty(self.base_value)}): below it {self.favoured_below} wins, "
            f"above it {self.favoured_above} wins."
        )
        if self.band is None:
            return sentence
        low, high = self.band
        where = "inside" if self.is_plausible else "outside"
        tail = "" if self.is_plausible else ", so the flip is possible but not plausible"
        return (
            f"{sentence}  That is {where} the {_pretty(low)} to {_pretty(high)} "
            f"you called plausible{tail}."
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
    param: ParamName,
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
        param: The parameter to solve for: an ``Uncertain`` mark, an alias,
            or a dotted path.
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
    param: ParamName,
    between: tuple[str, str],
    *,
    bounds: tuple[float, float] | None = None,
    samples: int = 33,
    tolerance: float = 1e-8,
) -> SwitchPoint | NoSwitchPoint:
    """As :func:`switch_point`, but explains itself when there is no crossing."""
    a, b = between
    shown = name_of(param)
    base = get_param(case, param)
    declared = spread_of(case, param)
    band = None if declared is None else _band(declared, base)
    low, high = _default_bounds(base) if bounds is None else bounds
    if low >= high:
        raise CarbitrageError(f"bounds must be increasing, got {(low, high)!r}")

    def differential(x: float) -> float:
        result = set_param(case, param, x).run()
        return result[a].npv - result[b].npv

    xs, ys = _scan(differential, low, high, samples)
    if len(xs) < 2:
        return NoSwitchPoint(
            shown, between, "the parameter is not valid anywhere in this domain", (low, high)
        )
    bracket = _first_sign_change(xs, ys)
    if bracket is None:
        direction = "always" if ys[0] > 0 else "never"
        return NoSwitchPoint(
            shown,
            between,
            f"{a} {direction} beats {b} across the whole domain, so the ranking never flips",
            (low, high),
        )
    lo, hi = bracket
    root = float(brentq(differential, lo, hi, xtol=tolerance))
    below, above = (b, a) if differential(lo) < 0 else (a, b)
    return SwitchPoint(
        param=shown,
        value=root,
        between=between,
        base_value=base,
        favoured_below=below,
        favoured_above=above,
        band=band,
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
