"""Errors and warnings raised by :mod:`carbitrage`.

The library prefers loud failure over silent approximation.  Every condition
here corresponds to a modelling mistake that would otherwise produce a
plausible-looking but wrong number.
"""

from __future__ import annotations

__all__ = [
    "CarbitrageError",
    "DoubleCountingWarning",
    "EligibilityError",
    "InconsistentRateBasisError",
    "TimelineError",
    "UnequalLivesError",
]


class CarbitrageError(Exception):
    """Base class for every error raised by this library."""


class UnequalLivesError(CarbitrageError):
    """Alternatives with different useful lives were compared by raw NPV.

    Two remedies exist and both are supported: equalise the lives with a
    :class:`~carbitrage.chain.ReplacementChain`, or compare equivalent annual
    cost.  Silently truncating the longer-lived alternative is not one of them.
    """


class InconsistentRateBasisError(CarbitrageError):
    """Real cash flows were mixed with a nominal discount rate, or vice versa.

    Either escalate the flows with growth rates and discount at a nominal rate,
    or leave the flows in today's money and discount at a real rate.  The Fisher
    relation ``(1 + i_nom) = (1 + i_real)(1 + pi)`` converts between the two.
    """


class TimelineError(CarbitrageError):
    """A cash flow could not be resolved against the timeline it was given."""


class EligibilityError(CarbitrageError):
    """An incentive was asked for flows although the alternative is not eligible."""


class DoubleCountingWarning(UserWarning):
    """The same economic benefit appears to be counted twice.

    Raised most importantly when an explicit subsidy is combined with an
    advertised German lease rate: those rates almost always have the subsidy
    already baked in as a capitalised initial payment.
    """
