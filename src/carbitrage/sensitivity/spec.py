"""Describing the parameter being varied."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..errors import CarbitrageError
from .distributions import Distribution

if TYPE_CHECKING:  # pragma: no cover
    from ..params import Spread

__all__ = [
    "Range",
]


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


#: Quantiles at which an unbounded spread is read as "plausible".
_PLAUSIBLE = (0.10, 0.90)


def _band(
    spread: Spread, base: float, quantiles: tuple[float, float] = _PLAUSIBLE
) -> tuple[float, float]:
    """The plausible low and high of a declared spread.

    A bounded spread states its own plausibility: everything it can produce is
    plausible and nothing outside it is, so the band is the whole support.  An
    unbounded one says no such thing — a normal's support is the entire real
    line — so it is read at ``quantiles`` instead.

    A relative range is a pair of multipliers, so it needs the base value it
    multiplies.  That is the only use the other forms have for ``base``.
    """
    if isinstance(spread, Range):
        if spread.relative:
            return (base * spread.low, base * spread.high)
        return (spread.low, spread.high)
    if not isinstance(spread, Distribution):  # pragma: no cover - guarded at construction
        raise CarbitrageError(f"a spread must be a Distribution or a Range, got {spread!r}")
    low, high = spread.support()
    if math.isfinite(low) and math.isfinite(high):
        return (low, high)
    edges = spread.ppf(np.asarray(quantiles, dtype=np.float64))
    return (float(edges[0]), float(edges[1]))
