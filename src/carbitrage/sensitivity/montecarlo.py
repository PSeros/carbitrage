"""Sampling the joint uncertainty, and how often each option wins."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..errors import CarbitrageError
from ..params import ParamName, name_of, resolve, set_params, spread_of
from .distributions import Distribution, _norm_cdf

if TYPE_CHECKING:  # pragma: no cover
    from ..comparison import Case

__all__ = [
    "MonteCarlo",
    "monte_carlo",
]

# ---------------------------------------------------------- monte carlo


@dataclass(frozen=True)
class MonteCarlo:
    """A simulated distribution of the *difference* between two alternatives.

    Reporting the difference rather than each alternative separately is the
    whole point.  Two alternatives that share an energy price move together, and
    the spread of each one on its own says nothing about how likely it is that
    one beats the other.
    """

    a: str
    b: str
    differences: npt.NDArray[np.float64]
    npv: dict[str, npt.NDArray[np.float64]]
    params: tuple[str, ...]
    draws: npt.NDArray[np.float64]

    @property
    def n(self) -> int:
        return int(self.differences.size)

    def probability_a_beats_b(self) -> float:
        """Share of trials in which ``a`` has the higher net present value."""
        return float(np.mean(self.differences > 0.0))

    def mean_difference(self) -> float:
        return float(np.mean(self.differences))

    def percentiles(self, levels: Sequence[float] = (5, 25, 50, 75, 95)) -> dict[float, float]:
        """Percentiles of the difference."""
        values = np.percentile(self.differences, levels)
        return {float(level): float(value) for level, value in zip(levels, values, strict=True)}

    def describe(self) -> str:
        p = self.probability_a_beats_b()
        q = self.percentiles((5, 50, 95))
        return (
            f"{self.a} beats {self.b} in {p:.1%} of {self.n:,} trials.  "
            f"Median advantage {q[50.0]:,.0f}, 5th to 95th percentile "
            f"{q[5.0]:,.0f} to {q[95.0]:,.0f}."
        )


def monte_carlo(
    case: Case,
    distributions: Sequence[ParamName] | Mapping[ParamName, Distribution | None],
    *,
    between: tuple[str, str],
    n: int = 2_000,
    correlation: npt.NDArray[np.float64] | Sequence[Sequence[float]] | None = None,
    seed: int | None = None,
) -> MonteCarlo:
    """Simulate the comparison with uncertain inputs.

    Args:
        case: The base case.
        distributions: Either parameter names, which then sample the
            distribution each one declares for itself, or a mapping from name to
            a distribution.  A mapping value of ``None`` also defers to the
            declaration, which is how some parameters take theirs from the model
            while others are given one here.
        between: The two alternatives whose difference is reported.
        n: Number of trials.
        correlation: Correlation matrix over the parameters, in the order they
            appear in ``distributions``.  Energy prices and residual values are
            not independent, and pretending otherwise understates the spread of
            the difference.  Applied as a Gaussian copula, so each marginal
            keeps its own shape.
        seed: Seed for reproducibility.

    Raises:
        CarbitrageError: on an empty specification, a non-positive ``n``, a name
            left to a declaration that cannot be sampled, or a correlation
            matrix that is not symmetric positive definite.
    """
    if not distributions:
        raise CarbitrageError("monte_carlo needs at least one parameter")
    if n <= 0:
        raise CarbitrageError(f"n must be positive, got {n!r}")
    a, b = between
    given: dict[ParamName, Distribution | None] = (
        dict(distributions) if isinstance(distributions, Mapping) else dict.fromkeys(distributions)
    )
    names = tuple(given)
    sampled = {name: _sampled(case, name, given[name]) for name in names}
    for name in names:
        resolve(case, name)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, len(names)))
    if correlation is not None:
        z = z @ _cholesky(np.asarray(correlation, dtype=np.float64), len(names)).T
    u = _norm_cdf(z)

    draws = np.column_stack([sampled[name].ppf(u[:, j]) for j, name in enumerate(names)])

    npv: dict[str, list[float]] = {name: [] for name in (a, b)}
    differences = np.empty(n, dtype=np.float64)
    for i in range(n):
        trial = set_params(case, dict(zip(names, draws[i], strict=True)))
        result = trial.run()
        npv_a, npv_b = result[a].npv, result[b].npv
        npv[a].append(npv_a)
        npv[b].append(npv_b)
        differences[i] = npv_a - npv_b
    return MonteCarlo(
        a=a,
        b=b,
        differences=differences,
        npv={name: np.asarray(values, dtype=np.float64) for name, values in npv.items()},
        params=tuple(name_of(name) for name in names),
        draws=draws,
    )


def _sampled(case: Case, name: ParamName, given: Distribution | None) -> Distribution:
    """The distribution to sample for one parameter, given or declared.

    A declared :class:`Range` is refused rather than widened into a uniform: a
    range says where a value lies, a simulation needs to know how likely each
    value in it is, and quietly supplying the second from the first would put a
    modelling assumption nobody made into the answer.
    """
    if given is not None:
        return given
    declared = spread_of(case, name)
    if isinstance(declared, Distribution):
        return declared
    shown = name_of(name)
    if declared is None:
        raise CarbitrageError(
            f"{shown} declares no spread, so there is nothing to sample; give it a distribution"
        )
    raise CarbitrageError(
        f"{shown} declares {declared!r}, which says where the value lies but not how likely each "
        "value in it is; give it a distribution"
    )


def _cholesky(matrix: npt.NDArray[np.float64], k: int) -> npt.NDArray[np.float64]:
    if matrix.shape != (k, k):
        raise CarbitrageError(
            f"correlation matrix must be {k}x{k} to match the parameters, got {matrix.shape}"
        )
    if not np.allclose(matrix, matrix.T):
        raise CarbitrageError("correlation matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0):
        raise CarbitrageError("correlation matrix must have ones on the diagonal")
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise CarbitrageError(
            "correlation matrix is not positive definite, so no set of correlated draws can "
            "reproduce it"
        ) from exc
