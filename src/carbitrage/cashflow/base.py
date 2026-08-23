"""What a cash flow is, and the labels a present value decomposes into."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..errors import TimelineError
from ..rates import Timeline

__all__ = [
    "CashFlow",
    "Component",
    "Frequency",
]


class Component(StrEnum):
    """The labelled parts a present value decomposes into."""

    ACQUISITION = "acquisition"
    FINANCING = "financing"
    LEASE = "lease"
    ENERGY = "energy"
    INSURANCE = "insurance"
    MAINTENANCE = "maintenance"
    TAX = "tax"
    SUBSIDY = "subsidy"
    RESIDUAL = "residual"
    DISPOSAL = "disposal"
    OTHER = "other"


class Frequency(StrEnum):
    """How often a recurring flow is paid."""

    PER_PERIOD = "per_period"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMIANNUAL = "semiannual"
    ANNUAL = "annual"

    def stride(self, timeline: Timeline) -> int:
        """Number of timeline periods between two consecutive payments.

        Raises:
            TimelineError: if the frequency is not representable on the grid,
                e.g. monthly payments on an annual timeline.
        """
        if self is Frequency.PER_PERIOD:
            return 1
        per_year = {
            Frequency.MONTHLY: 12,
            Frequency.QUARTERLY: 4,
            Frequency.SEMIANNUAL: 2,
            Frequency.ANNUAL: 1,
        }[self]
        stride, remainder = divmod(timeline.periods_per_year, per_year)
        if remainder or stride < 1:
            raise TimelineError(
                f"{self.value} payments are not representable on a timeline with "
                f"{timeline.periods_per_year} periods per year"
            )
        return stride


@runtime_checkable
class CashFlow(Protocol):
    """A stream of amounts that can be resolved against a timeline.

    Declared with read-only members so that frozen dataclasses satisfy it.
    """

    @property
    def label(self) -> Component:
        """Which component of the breakdown this flow belongs to."""
        ...

    @property
    def description(self) -> str:
        """Human-readable provenance, shown in breakdowns."""
        ...

    def amounts(self, timeline: Timeline) -> npt.NDArray[np.float64]:
        """Signed amounts per period, indexed ``0 .. timeline.n_periods``."""
        ...


class ResidualValueModelLike(Protocol):
    """The part of a residual value model :class:`Terminal` depends on."""

    def value_after(
        self, price_paid: float, *, years_held: float, age_at_acquisition: float = 0.0
    ) -> float: ...
