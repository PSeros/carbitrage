"""Marginals, sampled through their inverse CDF."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..errors import CarbitrageError

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = [
    "Distribution",
    "LogNormal",
    "Normal",
    "Triangular",
    "Uniform",
]

# ----------------------------------------------------------- distributions


class Distribution(ABC):
    """A marginal distribution, sampled through its inverse CDF.

    Sampling by inverse transform is what lets correlated draws be produced with
    a Gaussian copula: the dependence structure is imposed on uniforms, and each
    marginal keeps its own shape.
    """

    @abstractmethod
    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Inverse CDF evaluated on uniforms in (0, 1)."""

    @abstractmethod
    def mean(self) -> float:
        """The distribution's mean, used to report the base of a run."""

    def support(self) -> tuple[float, float]:
        """Where this distribution can produce values at all.

        Unbounded by default, which is the safe answer: a caller that reads a
        support as finite will treat everything outside it as impossible, and
        claiming that wrongly is worse than claiming nothing.  It cannot be
        inferred from :meth:`ppf`, whose inverse normal clamps its argument
        away from 0 and 1 and so reports a finite value for an infinite tail.
        """
        return (-math.inf, math.inf)


@dataclass(frozen=True)
class Normal(Distribution):
    """Gaussian.  Symmetric, and unbounded in both directions."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise CarbitrageError(f"sigma must not be negative, got {self.sigma!r}")

    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.mu + self.sigma * _norm_ppf(u)

    def mean(self) -> float:
        return self.mu


@dataclass(frozen=True)
class LogNormal(Distribution):
    """Log-normal.  Strictly positive, which suits prices and residual values."""

    mu_log: float
    sigma_log: float

    def __post_init__(self) -> None:
        if self.sigma_log < 0:
            raise CarbitrageError(f"sigma_log must not be negative, got {self.sigma_log!r}")

    @classmethod
    def from_mean_cv(cls, mean: float, cv: float) -> LogNormal:
        """Parameterise by arithmetic mean and coefficient of variation."""
        if mean <= 0:
            raise CarbitrageError(f"mean must be positive, got {mean!r}")
        sigma_sq = math.log(1.0 + cv * cv)
        return cls(mu_log=math.log(mean) - sigma_sq / 2.0, sigma_log=math.sqrt(sigma_sq))

    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.exp(self.mu_log + self.sigma_log * _norm_ppf(u))

    def mean(self) -> float:
        return math.exp(self.mu_log + self.sigma_log**2 / 2.0)

    def support(self) -> tuple[float, float]:
        return (0.0, math.inf)


@dataclass(frozen=True)
class Uniform(Distribution):
    """Uniform between two bounds.  The honest choice when only a range is known."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise CarbitrageError(f"low must be below high, got {(self.low, self.high)!r}")

    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.low + (self.high - self.low) * u

    def mean(self) -> float:
        return (self.low + self.high) / 2.0

    def support(self) -> tuple[float, float]:
        return (self.low, self.high)


@dataclass(frozen=True)
class Triangular(Distribution):
    """Triangular: a minimum, a most likely value and a maximum.

    The usual choice when an expert can state a best guess and two extremes but
    nothing about the shape between them.
    """

    low: float
    mode: float
    high: float

    def __post_init__(self) -> None:
        if not self.low <= self.mode <= self.high or self.low == self.high:
            raise CarbitrageError(
                f"require low <= mode <= high with low < high, got "
                f"{(self.low, self.mode, self.high)!r}"
            )

    def ppf(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        span = self.high - self.low
        pivot = (self.mode - self.low) / span
        lower = self.low + np.sqrt(u * span * (self.mode - self.low))
        upper = self.high - np.sqrt((1.0 - u) * span * (self.high - self.mode))
        return np.where(u < pivot, lower, upper)

    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3.0

    def support(self) -> tuple[float, float]:
        return (self.low, self.high)


def _norm_ppf(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Inverse standard-normal CDF, via the Beasley-Springer-Moro algorithm.

    Implemented here rather than imported so that the core stays on numpy, with
    scipy reserved for root finding.  Accuracy is better than 1e-6 across the
    range, which is orders of magnitude finer than the input uncertainty any of
    these models carries.
    """
    a = (2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637)
    b = (-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833)
    c = (
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    )
    u = np.clip(u, 1e-15, 1.0 - 1e-15)
    y = u - 0.5
    central = np.abs(y) < 0.42

    r = np.where(central, y * y, 0.0)
    numerator = ((a[3] * r + a[2]) * r + a[1]) * r + a[0]
    denominator = (((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0
    inner = y * numerator / denominator

    tail = np.where(y > 0.0, 1.0 - u, u)
    s = np.log(-np.log(np.clip(tail, 1e-300, 1.0 - 1e-16)))
    poly = np.full_like(s, c[-1])
    for coefficient in reversed(c[:-1]):
        poly = poly * s + coefficient
    outer = np.where(y > 0.0, poly, -poly)

    return np.asarray(np.where(central, inner, outer), dtype=np.float64)


def _norm_cdf(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Standard-normal CDF, vectorised over an array."""
    return np.asarray(0.5 * (1.0 + _erf(x / math.sqrt(2.0))), dtype=np.float64)


_erf = np.vectorize(math.erf, otypes=[np.float64])
