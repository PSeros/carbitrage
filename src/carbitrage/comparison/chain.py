"""Alternatives run back to back, so that unequal lives can be compared."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..cashflow import CashFlowSeries
from ..context import Context
from ..errors import CarbitrageError, UnequalLivesError
from .alternative import Alternative

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

__all__ = [
    "ReplacementChain",
]


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
        overrides in :mod:`carbitrage.study.params` reach inside a chain's legs, so the
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
