"""The facts a cash-flow builder needs beyond the asset itself.

Every builder in the library takes ``(subject, ctx)``.  The brief sketched
narrower signatures — ``flows(vehicle, usage, timeline)`` — but those cannot
express a replacement chain, and the missing fact is not plumbing: the period at
which a leg *starts* is an economic quantity.  It fixes the asset's age at the
horizon and therefore its residual, it fixes the vintage of the price paid, and
it fixes whether and when a subsidy disburses.  A chain's second leg starting at
month 24 pays its subsidy in month 28, not month 4, and is four years old at a
six-year horizon, not six.

So the window travels with the context, and every builder sees the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .errors import CarbitrageError
from .tax import PrivateHousehold, TaxTreatment
from .timeline import Timeline
from .vehicle import Vehicle

__all__ = ["Context", "Household", "Incumbent", "Usage"]


@dataclass(frozen=True)
class Usage:
    """How the asset is used.  The dominant driver of running-cost differences."""

    annual_km: float = 0.0

    def __post_init__(self) -> None:
        if self.annual_km < 0:
            raise CarbitrageError(f"annual_km must not be negative, got {self.annual_km!r}")

    def km_per_period(self, timeline: Timeline) -> float:
        """Mileage attributable to one period."""
        return self.annual_km / timeline.periods_per_year


@dataclass(frozen=True)
class Household:
    """Who is buying, to the extent means-tested incentives depend on it.

    Means-testing data belongs here rather than on the vehicle: the same vehicle
    attracts a different grant for a different buyer.

    Args:
        taxable_income: Taxable household income per year.  Under the 2026
            German scheme this is the average of the two most recent tax
            assessments, with both partners' incomes added together.
        children: Children under 18 in the household.
    """

    taxable_income: float = 0.0
    children: int = 0

    def __post_init__(self) -> None:
        if self.taxable_income < 0:
            raise CarbitrageError(
                f"taxable_income must not be negative, got {self.taxable_income!r}"
            )
        if self.children < 0:
            raise CarbitrageError(f"children must not be negative, got {self.children!r}")


@dataclass(frozen=True)
class Incumbent:
    """The asset already owned — the "defender" in replacement analysis.

    Its **current market value** is the economically relevant figure: an inflow
    at t=0 for every alternative that disposes of it, and a forgone inflow for
    the alternative that keeps it.  What was paid for it in the past never
    enters, and neither does its book value.

    Args:
        vehicle: The asset, carrying its own running costs and residual curve.
        market_value: What it would fetch today.
    """

    vehicle: Vehicle
    market_value: float = 0.0

    def __post_init__(self) -> None:
        if self.market_value < 0:
            raise CarbitrageError(f"market_value must not be negative, got {self.market_value!r}")


@dataclass(frozen=True)
class Context:
    """Everything a cash-flow builder needs, including the window it builds for.

    Args:
        timeline: The period grid every flow resolves against.
        usage: Mileage profile.
        household: The buyer, for means-tested incentives.
        tax: How the buyer is taxed.
        incumbent: The asset already owned, if any.
        start: First period of this leg.  Acquisition lands here.
        end: Last period of this leg.  The terminal value lands here.
    """

    timeline: Timeline
    usage: Usage = Usage()
    household: Household = Household()
    tax: TaxTreatment = field(default_factory=PrivateHousehold)
    incumbent: Incumbent | None = None
    start: int = 0
    end: int | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise CarbitrageError(f"start must not be negative, got {self.start!r}")
        self.timeline.check_period(self.start)
        if self.end is not None:
            self.timeline.check_period(self.end)
            if self.end < self.start:
                raise CarbitrageError(f"end {self.end} precedes start {self.start}")

    @property
    def last(self) -> int:
        """Last period of this leg, defaulting to the horizon."""
        return self.timeline.n_periods if self.end is None else self.end

    @property
    def n_periods(self) -> int:
        """Length of this leg in periods."""
        return self.last - self.start

    @property
    def years_held(self) -> float:
        """Length of this leg in years."""
        return self.n_periods / self.timeline.periods_per_year

    def window(self, start: int, end: int | None = None) -> Context:
        """The same context over a different window."""
        return replace(self, start=start, end=end)
