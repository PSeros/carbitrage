"""Residual value models.

Every model takes an **age**, never a position in the comparison horizon.  That
is what keeps a replacement chain consistent: the successor vehicle bought at
month 24 is four years old at a six-year horizon, and must be valued at four
years on the same curve as any other four-year-old asset.

Terminal values are therefore derived rather than entered as free parameters —
see :meth:`carbitrage.cashflow.Terminal.from_residual`.

**Real or nominal.**  A depreciation rate read off observed used-asset prices is
a *nominal* rate of price decline: it already contains inflation.  Applying it to
today's nominal price gives a future nominal price, which is correctly
discounted at a nominal rate, and that is what the reference workbook does.
Working on a real basis means restating the rate in real terms as well,
``1 - (1 - nominal) / (1 + inflation)``.  The library cannot tell which one it
was handed, so the choice is documented here rather than guessed at.
"""

from __future__ import annotations

import bisect
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .errors import CarbitrageError

__all__ = [
    "FirstYearDropThenGeometric",
    "GeometricDecline",
    "ResidualValueModel",
    "TabulatedResiduals",
]


class ResidualValueModel(ABC):
    """Maps an acquisition price and an age to a market value."""

    @abstractmethod
    def value(self, price: float, age_years: float) -> float:
        """Market value of an asset that cost ``price`` when new, at ``age_years``."""

    def retained(self, age_years: float) -> float:
        """Fraction of the as-new price retained at ``age_years`` of asset age."""
        return self.value(1.0, age_years)

    def value_after(
        self, price_paid: float, *, years_held: float, age_at_acquisition: float = 0.0
    ) -> float:
        """Value after ``years_held`` of an asset bought at ``age_at_acquisition``.

        The curve's age axis is *absolute asset age*, but the price actually paid
        need not sit on it — a used vehicle is bought at a market price at age 3,
        not at the as-new price.  The curve is therefore rescaled to pass through
        ``price_paid`` at ``age_at_acquisition``, which is what keeps a new
        vehicle, a used one and a chain successor on one consistent curve.

        Raises:
            CarbitrageError: if the curve has already reached zero at the age the
                asset was acquired, leaving nothing to rescale.
        """
        self._check_age(years_held)
        self._check_age(age_at_acquisition)
        base = self.retained(age_at_acquisition)
        if base <= 0.0:
            raise CarbitrageError(
                f"{type(self).__name__} retains nothing at age {age_at_acquisition:g}, so a price "
                "paid at that age cannot be placed on the curve"
            )
        return price_paid * self.retained(age_at_acquisition + years_held) / base

    def _check_age(self, age_years: float) -> None:
        if age_years < 0:
            raise CarbitrageError(f"age_years must not be negative, got {age_years!r}")


@dataclass(frozen=True)
class GeometricDecline(ResidualValueModel):
    """Constant proportional decline: ``price * (1 - rate) ** age``.

    Simple and scale-free, but it understates the first year of a new asset's
    depreciation.  For a new battery-electric vehicle prefer
    :class:`FirstYearDropThenGeometric`; using a constant rate there flatters
    buying now relative to waiting.
    """

    rate: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate < 1.0:
            raise CarbitrageError(f"rate must lie in [0, 1), got {self.rate!r}")

    def value(self, price: float, age_years: float) -> float:
        self._check_age(age_years)
        return float(price * (1.0 - self.rate) ** age_years)


@dataclass(frozen=True)
class FirstYearDropThenGeometric(ResidualValueModel):
    """A discrete first-year drop, then a constant rate on the reduced base.

    ``value = price * (1 - drop) * (1 - rate) ** max(age - 1, 0)``, with the drop
    prorated linearly within the first year.  Electric vehicles in particular
    lose disproportionately in year one, and a single constant rate cannot
    represent that.
    """

    drop: float
    rate: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.drop < 1.0:
            raise CarbitrageError(f"drop must lie in [0, 1), got {self.drop!r}")
        if not 0.0 <= self.rate < 1.0:
            raise CarbitrageError(f"rate must lie in [0, 1), got {self.rate!r}")

    def value(self, price: float, age_years: float) -> float:
        self._check_age(age_years)
        if age_years <= 1.0:
            return price * (1.0 - self.drop * age_years)
        return float(price * (1.0 - self.drop) * (1.0 - self.rate) ** (age_years - 1.0))


@dataclass(frozen=True)
class TabulatedResiduals(ResidualValueModel):
    """Retained-value fractions by age, linearly interpolated.

    This is the shape Schwacke- and DAT-style curves come in.  Ages beyond the
    last tabulated point continue at the rate implied by the final segment
    rather than clamping, so a chain that runs past the table does not silently
    freeze the asset's value.

    Args:
        table: Mapping of age in years to the fraction of price retained.
    """

    table: tuple[tuple[float, float], ...]

    def __init__(self, table: dict[float, float] | tuple[tuple[float, float], ...]) -> None:
        items = tuple(sorted(table.items() if isinstance(table, dict) else table))
        if len(items) < 2:
            raise CarbitrageError("a residual table needs at least two points")
        if any(age < 0 for age, _ in items):
            raise CarbitrageError("residual table ages must not be negative")
        object.__setattr__(self, "table", items)

    @classmethod
    def from_values(cls, price: float, values: dict[float, float]) -> TabulatedResiduals:
        """Build a table from absolute values, e.g. ``{0: 1500, 2: 800}``.

        Convenient when the input is a dealer quote rather than a depreciation
        curve.  Age 0 defaults to the full price when not supplied.
        """
        if price <= 0:
            raise CarbitrageError(f"price must be positive to build a table, got {price!r}")
        table = {age: value / price for age, value in values.items()}
        table.setdefault(0.0, 1.0)
        return cls(table)

    def value(self, price: float, age_years: float) -> float:
        self._check_age(age_years)
        ages = [age for age, _ in self.table]
        fractions = [fraction for _, fraction in self.table]
        if age_years <= ages[0]:
            return price * fractions[0]
        i = bisect.bisect_right(ages, age_years)
        if i >= len(ages):
            i = len(ages) - 1  # extrapolate along the final segment
        lo_age, hi_age = ages[i - 1], ages[i]
        lo, hi = fractions[i - 1], fractions[i]
        weight = (age_years - lo_age) / (hi_age - lo_age)
        return price * max(lo + weight * (hi - lo), 0.0)
