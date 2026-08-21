"""Period indexing, discount factors and escalation indices.

Every cash flow in the library is resolved against a :class:`Timeline`.  It owns
three things and nothing else: the mapping between wall-clock years and period
indices, the discount factor for a period, and the escalation index for a period
and a growth rate.

Two conventions here are correctness requirements rather than defaults.

**Conformal periodisation.**  The sub-annual rate is ``(1 + i_a) ** (1/m) - 1``,
so that compounding ``m`` sub-periods reproduces the annual rate exactly.  The
proportional convention ``i_a / m`` overstates the effective rate and is
available only by explicitly choosing :attr:`Periodisation.PROPORTIONAL`.

**Real versus nominal.**  Either escalate the cash flows with growth rates and
discount at a nominal rate, or leave them in today's money and discount at a real
rate.  Mixing the two is caught at construction time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from ..errors import InconsistentRateBasisError, TimelineError

__all__ = [
    "Escalation",
    "Periodisation",
    "RateBasis",
    "Timeline",
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


@dataclass(frozen=True)
class Timeline:
    """A discrete, evenly spaced period grid with discounting and escalation.

    Args:
        horizon_years: Length of the comparison horizon.  Must be positive.
        periods_per_year: Sub-periods per year (12 monthly, 1 annual).
        rate: Discount rate per annum, expressed on ``basis``.
        basis: Whether ``rate`` and the cash flows are nominal or real.
        inflation: General inflation.  Required when ``basis`` is real, so that
            the price level the flows are stated in is on the record.
        energy_escalation: Growth rate of energy prices per annum.
        vehicle_price_escalation: Growth rate of asset acquisition prices per
            annum.  Applies to assets acquired later in a replacement chain.
        cost_escalation: Growth rate of other recurring costs per annum.
        periodisation: Conformal (default) or proportional rate conversion.
        escalation_mode: How growth is spread within a year.

    Raises:
        TimelineError: on a non-positive horizon or period count.
        InconsistentRateBasisError: if nominal discounting is combined with an
            explicit inflation assumption but no escalation anywhere, or if a
            real basis is used without stating the inflation it is real to.
    """

    horizon_years: float
    periods_per_year: int = MONTHS_PER_YEAR
    rate: float = 0.0
    basis: RateBasis = RateBasis.NOMINAL
    inflation: float | None = None
    energy_escalation: float = 0.0
    vehicle_price_escalation: float = 0.0
    cost_escalation: float = 0.0
    periodisation: Periodisation = Periodisation.CONFORMAL
    escalation_mode: Escalation = Escalation.ANNUAL_STEP

    _factors: npt.NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.horizon_years <= 0:
            raise TimelineError(f"horizon_years must be positive, got {self.horizon_years!r}")
        if self.periods_per_year <= 0:
            raise TimelineError(f"periods_per_year must be positive, got {self.periods_per_year!r}")
        if self.rate <= -1.0:
            raise TimelineError(f"rate must exceed -100 %, got {self.rate!r}")
        self._validate_rate_basis()
        t = np.arange(self.n_periods + 1, dtype=np.float64)
        object.__setattr__(self, "_factors", (1.0 + self.periodic_rate) ** (-t))

    # ------------------------------------------------------------------ basis

    @property
    def escalations(self) -> tuple[float, ...]:
        """Every growth rate this timeline carries, in declaration order."""
        return (self.energy_escalation, self.vehicle_price_escalation, self.cost_escalation)

    def _validate_rate_basis(self) -> None:
        if self.basis is RateBasis.REAL:
            if self.inflation is None:
                raise InconsistentRateBasisError(
                    "A real discount rate needs an explicit `inflation` so that the price level "
                    "the flows are stated in is on the record.  Use Timeline.real(...); growth "
                    "rates are then read as real growth."
                )
            return
        if self.inflation is not None and not any(g != 0.0 for g in self.escalations):
            raise InconsistentRateBasisError(
                "Nominal discounting with an inflation assumption but no escalation anywhere: the "
                "cash flows are in today's money while the rate is not.  Either escalate the flows "
                "(energy_escalation / cost_escalation / vehicle_price_escalation) or switch to "
                "Timeline.real(rate=fisher_real(nominal, inflation), inflation=...)."
            )

    @classmethod
    def real(
        cls,
        horizon_years: float,
        *,
        rate: float,
        inflation: float,
        periods_per_year: int = MONTHS_PER_YEAR,
        **kwargs: Any,
    ) -> Timeline:
        """Construct a real-basis timeline.  Growth rates are read as *real* growth."""
        return cls(
            horizon_years=horizon_years,
            periods_per_year=periods_per_year,
            rate=rate,
            basis=RateBasis.REAL,
            inflation=inflation,
            **kwargs,
        )

    def to_nominal(self) -> Timeline:
        """Convert a real timeline to its nominal twin via the Fisher relation."""
        if self.basis is RateBasis.NOMINAL:
            return self
        pi = self.inflation
        assert pi is not None  # guaranteed by _validate_rate_basis
        return Timeline(
            horizon_years=self.horizon_years,
            periods_per_year=self.periods_per_year,
            rate=fisher_nominal(self.rate, pi),
            basis=RateBasis.NOMINAL,
            inflation=None,
            energy_escalation=fisher_nominal(self.energy_escalation, pi),
            vehicle_price_escalation=fisher_nominal(self.vehicle_price_escalation, pi),
            cost_escalation=fisher_nominal(self.cost_escalation, pi),
            periodisation=self.periodisation,
            escalation_mode=self.escalation_mode,
        )

    # ------------------------------------------------------------------ shape

    @property
    def n_periods(self) -> int:
        """Number of periods after t=0.  The grid runs ``0 .. n_periods``."""
        return round(self.horizon_years * self.periods_per_year)

    @property
    def periods(self) -> npt.NDArray[np.int64]:
        """The period indices ``0 .. n_periods`` inclusive."""
        return np.arange(self.n_periods + 1, dtype=np.int64)

    @property
    def periodic_rate(self) -> float:
        """The per-period discount rate implied by :attr:`periodisation`."""
        if self.periodisation is Periodisation.PROPORTIONAL:
            return self.rate / self.periods_per_year
        return float((1.0 + self.rate) ** (1.0 / self.periods_per_year)) - 1.0

    # ------------------------------------------------------------ conversions

    def index_of(self, years: float) -> int:
        """The period index at ``years`` from t=0.

        Raises:
            TimelineError: if ``years`` lies outside ``[0, horizon_years]`` or
                does not land on a period boundary.
        """
        if years < 0 or years > self.horizon_years + 1e-12:
            raise TimelineError(
                f"{years!r} years is outside the horizon of {self.horizon_years} years"
            )
        exact = years * self.periods_per_year
        rounded = round(exact)
        if not math.isclose(exact, rounded, abs_tol=1e-9):
            raise TimelineError(
                f"{years!r} years is {exact:.6f} periods, which is not a period boundary at "
                f"{self.periods_per_year} periods per year"
            )
        return int(rounded)

    def years_at(self, t: int) -> float:
        """Wall-clock years elapsed at period ``t``."""
        return t / self.periods_per_year

    def year_index(self, t: int) -> int:
        """Zero-based policy year of period ``t`` under :attr:`Escalation.ANNUAL_STEP`."""
        return int(max(t - 1, 0) // self.periods_per_year)

    # ------------------------------------------------------------- arithmetic

    def discount_factor(self, t: int) -> float:
        """Present value at t=0 of one unit received at period ``t``."""
        self.check_period(t)
        return float(self._factors[t])

    def discount_factors(self) -> npt.NDArray[np.float64]:
        """Discount factors for every period ``0 .. n_periods``."""
        return self._factors.copy()

    def escalation(self, t: int, g: float) -> float:
        """Price index at period ``t`` for a flow growing at ``g`` per annum."""
        self.check_period(t)
        if self.escalation_mode is Escalation.CONTINUOUS:
            return float((1.0 + g) ** (t / self.periods_per_year))
        return float((1.0 + g) ** self.year_index(t))

    def escalation_index(self, g: float) -> npt.NDArray[np.float64]:
        """Vectorised :meth:`escalation` over every period."""
        t = self.periods
        if self.escalation_mode is Escalation.CONTINUOUS:
            exponent = t.astype(np.float64) / self.periods_per_year
        else:
            exponent = np.maximum(t - 1, 0) // self.periods_per_year
        return np.asarray((1.0 + g) ** exponent, dtype=np.float64)

    def annuity_factor(self, years: float | None = None) -> float:
        """PV of one unit per year for ``years`` years (default: the horizon).

        The reciprocal of :meth:`capital_recovery_factor`.
        """
        n = self.horizon_years if years is None else years
        i = self.rate
        if i == 0.0:
            return float(n)
        return float(((1.0 + i) ** n - 1.0) / (i * (1.0 + i) ** n))

    def capital_recovery_factor(self, years: float | None = None) -> float:
        """``CRF = i(1+i)^n / ((1+i)^n - 1)``: the annuity per unit of present value."""
        n = self.horizon_years if years is None else years
        i = self.rate
        if i == 0.0:
            return 1.0 / float(n)
        return float(i * (1.0 + i) ** n / ((1.0 + i) ** n - 1.0))

    def pv(self, amounts: npt.NDArray[np.float64]) -> float:
        """Present value of an amount vector indexed by period."""
        if amounts.shape != self._factors.shape:
            raise TimelineError(
                f"amount vector of length {amounts.shape[0]} does not match a timeline of "
                f"{self._factors.shape[0]} periods"
            )
        return float(np.dot(amounts, self._factors))

    def zeros(self) -> npt.NDArray[np.float64]:
        """An all-zero amount vector shaped for this timeline."""
        return np.zeros(self.n_periods + 1, dtype=np.float64)

    def check_period(self, t: int) -> None:
        """Raise :class:`TimelineError` unless ``t`` is a valid period index."""
        if t < 0 or t > self.n_periods:
            raise TimelineError(f"period {t} is outside 0..{self.n_periods}")
