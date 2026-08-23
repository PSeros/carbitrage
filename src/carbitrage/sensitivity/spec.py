"""Describing the parameter being varied."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import CarbitrageError

if TYPE_CHECKING:  # pragma: no cover
    pass

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
