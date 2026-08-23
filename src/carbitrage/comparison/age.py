"""How long to keep the asset: equivalent annual cost by holding period."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..context import Context, Household, Incumbent, Usage
from ..errors import CarbitrageError
from ..rates import Timeline
from ..tax import PrivateHousehold, TaxTreatment
from .alternative import Evaluable

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

__all__ = [
    "ReplacementAgeTable",
    "optimal_replacement_age",
]


@dataclass(frozen=True)
class ReplacementAgeTable:
    """Equivalent annual cost by holding period, and the period that minimises it.

    Attributes:
        ages: The holding periods evaluated, in years.
        eac: Equivalent annual **cost**, positive for a net cost, matching the
            sign convention of :meth:`carbitrage.engine.result.Evaluation.eac`.  This is
            the series to minimise; minimising a signed NPV annuity instead would
            pick the *most* expensive age.
        pv: Net present value per holding period, negative for a net cost.
    """

    ages: tuple[float, ...]
    eac: tuple[float, ...]
    pv: tuple[float, ...]

    @property
    def best_age(self) -> float:
        """The holding period with the lowest equivalent annual cost."""
        return self.ages[min(range(len(self.eac)), key=lambda i: self.eac[i])]

    @property
    def best_eac(self) -> float:
        """The equivalent annual cost at :attr:`best_age`."""
        return min(self.eac)

    def as_rows(self) -> tuple[tuple[float, float, float], ...]:
        """``(age, pv, eac)`` per candidate, in order."""
        return tuple(zip(self.ages, self.pv, self.eac, strict=True))


def optimal_replacement_age(
    alternative: Evaluable,
    *,
    timeline: Timeline,
    candidates: Sequence[float],
    usage: Usage | None = None,
    household: Household | None = None,
    tax: TaxTreatment | None = None,
    incumbent: Incumbent | None = None,
) -> ReplacementAgeTable:
    """The holding period that minimises equivalent annual cost.

    This is the textbook answer to "how long should I keep it".  Each candidate
    age is evaluated as a self-contained holding period — acquire, run, dispose
    — and converted to an annuity.  Minimising that annuity is equivalent to
    minimising the present value of an infinitely repeated chain of identical
    replacements, which is the assumption the method rests on and the reason it
    should not be used when successive replacements differ materially.

    The classical U-shaped cost curve — capital cost falling with age, running
    costs rising — only appears when running costs actually grow with age.  With
    flat costs the annuity falls monotonically and the "optimal" age is simply
    the longest candidate, which is a statement about the inputs rather than
    about replacement policy.  Set ``cost_escalation`` on the timeline, or use a
    residual curve with a steep early drop, to get a genuine interior optimum.

    Args:
        alternative: The course of action to hold.
        timeline: Supplies the rate, periodisation and escalations.  Its horizon
            is replaced by each candidate age in turn.
        candidates: Holding periods in years to evaluate.
        usage: Mileage profile.  Defaults to no mileage.
        household: The buyer.  Defaults to an unstated household.
        incumbent: The asset already owned, if the holding period displaces one.
        tax: Tax treatment, defaulting to a private household.

    Raises:
        CarbitrageError: if ``candidates`` is empty or contains a non-positive age.
    """
    if not candidates:
        raise CarbitrageError("optimal_replacement_age needs at least one candidate age")
    if any(age <= 0 for age in candidates):
        raise CarbitrageError(f"candidate ages must be positive, got {tuple(candidates)!r}")

    treatment = PrivateHousehold() if tax is None else tax
    profile = Usage() if usage is None else usage
    buyer = Household() if household is None else household
    ages: list[float] = []
    pvs: list[float] = []
    eacs: list[float] = []
    for age in candidates:
        horizon = replace(timeline, horizon_years=age)
        ctx = Context(
            timeline=horizon,
            usage=profile,
            household=buyer,
            tax=treatment,
            incumbent=incumbent,
            start=0,
            end=horizon.n_periods,
        )
        pv = alternative.flows(ctx).pv(horizon)
        ages.append(age)
        pvs.append(pv)
        # Positive for a net cost, so that the minimum is the cheapest age.
        eacs.append(-pv * horizon.capital_recovery_factor(age))
    return ReplacementAgeTable(tuple(ages), tuple(eacs), tuple(pvs))
