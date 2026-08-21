"""The entry point: compare mutually exclusive alternatives on one timeline.

The one thing this module refuses to do is compare alternatives with unequal
lives by raw net present value.  That comparison is meaningless — the
longer-lived option is being credited with service the other does not deliver —
and there are exactly two legitimate remedies, both supported:

* equalise the lives with a :class:`~carbitrage.engine.chain.ReplacementChain`, so every
  alternative covers the same horizon; or
* compare equivalent annual cost, which annualises each alternative over its own
  life and is licensed when replacements repeat identically.

Silently truncating the longer alternative at the shorter horizon is not one of
them, so an unequal-lives comparison raises unless the caller says which remedy
it wants.

A :class:`Case` is the whole comparison as one immutable value.  Keeping the
inputs alongside the outputs is what lets sensitivity analysis perturb one
parameter and re-run, rather than asking the caller to rebuild the model by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core.timeline import Timeline
from ..domain.context import Context, Household, Incumbent, Usage
from ..domain.tax import PrivateHousehold, TaxTreatment
from ..errors import UnequalLivesError
from .result import ComparisonResult, Evaluation

if TYPE_CHECKING:  # pragma: no cover
    from .alternative import Evaluable

__all__ = ["Case", "compare"]


@dataclass(frozen=True)
class Case:
    """A complete, re-runnable comparison: the alternatives and their assumptions.

    Every field is immutable, so a sensitivity run produces a *new* case rather
    than mutating the one it came from.  That is what makes one-way grids,
    switch points and Monte Carlo runs reproducible.
    """

    alternatives: tuple[Evaluable, ...]
    timeline: Timeline
    usage: Usage = field(default_factory=Usage)
    household: Household = field(default_factory=Household)
    incumbent: Incumbent | None = None
    tax: TaxTreatment = field(default_factory=PrivateHousehold)
    materiality: float = 0.03
    allow_unequal_lives: bool = False

    def context(self) -> Context:
        """The root context every alternative is evaluated against."""
        return Context(
            timeline=self.timeline,
            usage=self.usage,
            household=self.household,
            tax=self.tax,
            incumbent=self.incumbent,
            start=0,
            end=self.timeline.n_periods,
        )

    def run(self) -> ComparisonResult:
        """Evaluate every alternative and return the comparison."""
        if not self.alternatives:
            raise UnequalLivesError("nothing to compare: alternatives is empty")
        if not self.allow_unequal_lives:
            _reject_unequal_lives(self.alternatives, self.timeline)
        ctx = self.context()
        evaluations = tuple(
            Evaluation(
                name=alt.name,
                series=alt.flows(ctx),
                ctx=ctx,
                constraints=alt.constraints(ctx),
            )
            for alt in self.alternatives
        )
        return ComparisonResult(evaluations=evaluations, materiality=self.materiality, case=self)


def compare(
    alternatives: Sequence[Evaluable],
    timeline: Timeline,
    *,
    usage: Usage | None = None,
    household: Household | None = None,
    incumbent: Incumbent | None = None,
    tax: TaxTreatment | None = None,
    materiality: float = 0.03,
    allow_unequal_lives: bool = False,
) -> ComparisonResult:
    """Evaluate every alternative over one horizon and return the comparison.

    Args:
        alternatives: The mutually exclusive options.  Each must be an
            :class:`~carbitrage.engine.alternative.Alternative` or a
            :class:`~carbitrage.engine.chain.ReplacementChain`.
        timeline: The common horizon, rate and escalations.
        usage: Mileage profile applied to every alternative.
        household: The buyer, for means-tested incentives.
        incumbent: The asset already owned.  Its market value is an inflow at
            t=0 for every alternative that disposes of it, and its absence is
            the opportunity cost borne by the one that keeps it.
        tax: Tax treatment, defaulting to a private household.
        materiality: Fraction of the winner's present value below which its lead
            is reported as too close to call.
        allow_unequal_lives: Compare anyway, accepting that only the equivalent
            annual costs are meaningful and the present values are not.

    Raises:
        UnequalLivesError: if any alternative's life is shorter than the horizon
            and it is not wrapped in a chain, unless ``allow_unequal_lives``.
        CarbitrageError: on duplicate alternative names.

    Returns:
        A :class:`~carbitrage.engine.result.ComparisonResult`.
    """
    return Case(
        alternatives=tuple(alternatives),
        timeline=timeline,
        usage=Usage() if usage is None else usage,
        household=Household() if household is None else household,
        incumbent=incumbent,
        tax=PrivateHousehold() if tax is None else tax,
        materiality=materiality,
        allow_unequal_lives=allow_unequal_lives,
    ).run()


def _reject_unequal_lives(alternatives: Sequence[Evaluable], timeline: Timeline) -> None:
    """Raise unless every alternative covers the whole horizon."""
    short = {
        alt.name: alt.life_years
        for alt in alternatives
        if alt.life_years is not None and alt.life_years < timeline.horizon_years
    }
    if not short:
        return
    listed = ", ".join(f"{name} ({life:g} y)" for name, life in short.items())
    raise UnequalLivesError(
        f"these alternatives are shorter than the {timeline.horizon_years:g}-year horizon: "
        f"{listed}.  Comparing them by present value would credit the others with service "
        "these do not deliver.  Either wrap each in a ReplacementChain so that all alternatives "
        "cover the horizon, or pass allow_unequal_lives=True and read only the equivalent "
        "annual costs."
    )
