"""Parameter overrides, grids, switch points, tornados and Monte Carlo."""

from __future__ import annotations

import numpy as np
import pytest

from carbitrage.errors import CarbitrageError
from carbitrage.sensitivity import LogNormal, Normal, Range, Triangular, Uniform

# ------------------------------------------------------------------ params


# ------------------------------------------------------------------- grids


# ----------------------------------------------------------- switch points


# ---------------------------------------------------------------- tornado


def test_a_range_must_be_ordered() -> None:
    with pytest.raises(CarbitrageError, match="exceeds high"):
        Range(5.0, 1.0)


# ----------------------------------------------------------- distributions


def test_uniform_inverse_cdf_spans_its_range() -> None:
    u = np.array([0.0, 0.5, 1.0])
    assert Uniform(10.0, 20.0).ppf(u) == pytest.approx([10.0, 15.0, 20.0])


def test_normal_inverse_cdf_is_symmetric() -> None:
    normal = Normal(mu=5.0, sigma=2.0)
    assert normal.ppf(np.array([0.5]))[0] == pytest.approx(5.0, abs=1e-9)
    assert normal.ppf(np.array([0.975]))[0] == pytest.approx(5.0 + 2.0 * 1.959964, abs=1e-6)


def test_lognormal_from_mean_cv_reproduces_its_mean() -> None:
    dist = LogNormal.from_mean_cv(mean=0.99, cv=0.25)
    draws = dist.ppf(np.random.default_rng(0).uniform(1e-9, 1 - 1e-9, 200_000))
    assert float(np.mean(draws)) == pytest.approx(0.99, rel=0.02)
    assert dist.mean() == pytest.approx(0.99, rel=1e-9)
    assert np.all(draws > 0)


def test_triangular_respects_its_bounds_and_mode() -> None:
    dist = Triangular(low=0.8, mode=1.0, high=1.6)
    draws = dist.ppf(np.random.default_rng(1).uniform(0, 1, 50_000))
    assert draws.min() >= 0.8
    assert draws.max() <= 1.6
    assert float(np.mean(draws)) == pytest.approx(dist.mean(), rel=0.02)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: Normal(0.0, -1.0), "sigma must not be negative"),
        (lambda: Uniform(2.0, 1.0), "low must be below high"),
        (lambda: Triangular(1.0, 0.5, 2.0), "low <= mode <= high"),
        (lambda: LogNormal.from_mean_cv(0.0, 0.2), "mean must be positive"),
    ],
)
def test_distributions_validate_their_parameters(factory, match: str) -> None:
    with pytest.raises(CarbitrageError, match=match):
        factory()


# ---------------------------------------------------------- monte carlo


# ------------------------------------------------- the result-level surface
