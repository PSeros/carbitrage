"""How an annual rate becomes a rate per period, and how a real rate relates
to a nominal one.

Getting either convention wrong is the quiet way to a plausible but wrong
answer, so both are named types rather than a boolean and a float.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

__all__ = [
    "Escalation",
    "Periodisation",
    "RateBasis",
    "fisher_inflation",
    "fisher_nominal",
    "fisher_real",
]

MONTHS_PER_YEAR: Final = 12


class Periodisation(StrEnum):
    """How an annual rate is converted to a sub-annual one."""

    CONFORMAL = "conformal"
    """``i_m = (1 + i_a) ** (1/m) - 1``.  The default, and the only convention
    that compounds back to the annual rate."""

    PROPORTIONAL = "proportional"
    """``i_m = i_a / m``.  The nominal convention used by most loan contracts.
    Not equivalent to conformal, and must be chosen deliberately."""


class RateBasis(StrEnum):
    """Whether the discount rate and the cash flows are nominal or real."""

    NOMINAL = "nominal"
    REAL = "real"


class Escalation(StrEnum):
    """How an annual growth rate is spread across sub-annual periods."""

    ANNUAL_STEP = "annual_step"
    """The index is constant within a policy year and steps at each anniversary:
    ``(1 + g) ** floor((t - 1) / m)``.  Period 0 and the whole first year of
    payments carry an index of 1, so a payment at the end of month 12 is still a
    first-year payment.  This is how tariffs behave in practice."""

    CONTINUOUS = "continuous"
    """The index compounds every period: ``(1 + g) ** (t / m)``."""


def fisher_nominal(real_rate: float, inflation: float) -> float:
    """Nominal rate from a real rate: ``(1 + i_nom) = (1 + i_real)(1 + pi)``."""
    return (1.0 + real_rate) * (1.0 + inflation) - 1.0


def fisher_real(nominal_rate: float, inflation: float) -> float:
    """Real rate from a nominal rate.  The exact Fisher inverse, not ``i - pi``."""
    return (1.0 + nominal_rate) / (1.0 + inflation) - 1.0


def fisher_inflation(nominal_rate: float, real_rate: float) -> float:
    """The inflation rate implied by a nominal/real rate pair."""
    return (1.0 + nominal_rate) / (1.0 + real_rate) - 1.0
