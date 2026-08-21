"""Replacement chains, and the classical optimal replacement age.

A chain is how alternatives with unequal lives are made comparable: run the
incumbent for its remaining life, then acquire a successor, so that every
alternative covers the same horizon.

Three things have to be right at the handover, and all three are easy to get
wrong:

* the outgoing asset's disposal proceeds and the incoming asset's acquisition
  land in the **same period**;
* the successor's price is the price **at the date it is bought**, escalated
  from today, not today's price;
* the successor's terminal value is taken at **its own age** at the horizon —
  four years for a vehicle bought after two years of a six-year horizon — not at
  the horizon's length.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .alternative import Alternative, Evaluable
from .cashflow import CashFlowSeries
from .context import Context, Household, Incumbent, Usage
from .errors import CarbitrageError, UnequalLivesError
from .tax import PrivateHousehold, TaxTreatment
from .timeline import Timeline

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

__all__ = ["ReplacementAgeTable", "ReplacementChain", "optimal_replacement_age"]


@dataclass(frozen=True, init=False)
class ReplacementChain:
    """Alternatives run one after another to cover a common horizon.

    Every leg but the last must state a ``life_years``; the last runs to the
    horizon.  Only the first leg may dispose of the incumbent, because by the
    time a later leg begins the incumbent is already gone.

    Args:
        first: The leg that starts at t=0.
        then: The successor, or several in order.
        label: Display name.  Defaults to naming the legs in order.
        escalate_successor_price: Whether a successor's price is escalated from
            today's price to its acquisition date at the timeline's
            ``vehicle_price_escalation``.  Leaving this on is the honest
            default: buying the same car in two years costs more.
    """

    legs: tuple[Alternative, ...]
    label: str | None
    escalate_successor_price: bool

    def __init__(
        self,
        first: Alternative,
        then: Alternative | Sequence[Alternative] | None = None,
        *,
        label: str | None = None,
        escalate_successor_price: bool = True,
    ) -> None:
        successors: tuple[Alternative, ...]
        if then is None:
            successors = ()
        elif isinstance(then, Alternative):
            successors = (then,)
        else:
            successors = tuple(then)
        legs = (first, *successors)
        if len(legs) < 2:
            raise CarbitrageError(
                "a replacement chain needs at least two legs; a single alternative is not a chain"
            )
        for leg in legs[:-1]:
            if leg.life_years is None:
                raise UnequalLivesError(
                    f"leg {leg.name!r} has no life_years, so the chain cannot know when it hands "
                    "over.  Set life_years on every leg but the last."
                )
        object.__setattr__(self, "legs", legs)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "escalate_successor_price", escalate_successor_price)

    def __replace__(self, **changes: object) -> ReplacementChain:
        """Rebuild with some fields changed, honouring the custom constructor.

        ``dataclasses.replace`` cannot be used on this class because its
        ``__init__`` takes ``(first, then)`` rather than ``legs``.  Parameter
        overrides in :mod:`carbitrage.params` reach inside a chain's legs, so the
        rebuild has to work.
        """
        legs = changes.pop("legs", self.legs)
        assert isinstance(legs, tuple)
        label = changes.pop("label", self.label)
        escalate = changes.pop("escalate_successor_price", self.escalate_successor_price)
        if changes:
            raise TypeError(f"unexpected fields for ReplacementChain: {sorted(changes)}")
        return ReplacementChain(
            legs[0],
            legs[1:],
            label=label,  # type: ignore[arg-type]
            escalate_successor_price=bool(escalate),
        )

    @property
    def name(self) -> str:
        return self.label or " then ".join(leg.name for leg in self.legs)

    @property
    def life_years(self) -> float | None:
        """A chain always runs to the horizon, so it has no life of its own."""
        return None

    # ----------------------------------------------------------------- layout

    def windows(self, ctx: Context) -> tuple[tuple[Alternative, int, int], ...]:
        """Each leg with the absolute period window it occupies."""
        per_year = ctx.timeline.periods_per_year
        out: list[tuple[Alternative, int, int]] = []
        start = ctx.start
        for leg in self.legs[:-1]:
            assert leg.life_years is not None  # enforced in __init__
            end = start + round(leg.life_years * per_year)
            if end >= ctx.last:
                raise UnequalLivesError(
                    f"leg {leg.name!r} already reaches the horizon at period {end}, so the "
                    f"remaining {len(self.legs) - len(out) - 1} leg(s) would never run.  "
                    "Shorten the earlier legs or lengthen the horizon."
                )
            out.append((leg, start, end))
            start = end
        out.append((self.legs[-1], start, ctx.last))
        return tuple(out)

    # ------------------------------------------------------------------ flows

    def flows(self, ctx: Context) -> CashFlowSeries:
        series = CashFlowSeries()
        for index, (leg, start, end) in enumerate(self.windows(ctx)):
            series += self._prepare(leg, index, start, ctx).flows(ctx.window(start, end))
        return series

    def constraints(self, ctx: Context) -> tuple[str, ...]:
        out: tuple[str, ...] = ()
        for index, (leg, start, end) in enumerate(self.windows(ctx)):
            out += self._prepare(leg, index, start, ctx).constraints(ctx.window(start, end))
        return out

    def _prepare(self, leg: Alternative, index: int, start: int, ctx: Context) -> Alternative:
        """The leg as it is actually acquired: right vintage, right incumbent rules."""
        if index == 0:
            return leg
        vehicle = leg.vehicle
        if self.escalate_successor_price and ctx.timeline.vehicle_price_escalation:
            years = (start - ctx.start) / ctx.timeline.periods_per_year
            factor = (1.0 + ctx.timeline.vehicle_price_escalation) ** years
            vehicle = replace(vehicle, price=vehicle.price * factor)
        # A later leg cannot sell an asset the first leg has already disposed of.
        return replace(leg, vehicle=vehicle, disposes_incumbent=False)


@dataclass(frozen=True)
class ReplacementAgeTable:
    """Equivalent annual cost by holding period, and the period that minimises it.

    Attributes:
        ages: The holding periods evaluated, in years.
        eac: Equivalent annual **cost**, positive for a net cost, matching the
            sign convention of :meth:`carbitrage.result.Evaluation.eac`.  This is
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
