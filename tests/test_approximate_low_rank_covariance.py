"""The approximate backtest's low-rank covariance answers what the dense matrix answers (#323).

`LowRankCovariance` holds every decision bar's covariance as `B F B' +
diag(D)` without building the `[symbols, symbols]` matrix; its `variance` (the
diagonal) and `times` (the covariance times a book) are what the approximate
mean-variance reads. Both are locked here against the dense matrix, with a
factor covariance and with the identity, on random bars.

Everything is synthetic, CPU-only and offline.
"""

import numpy as np
import pytest

from quantlab.portfolio.approximate import LowRankCovariance


def _covariance(identity: bool) -> LowRankCovariance:
    rng = np.random.default_rng(7)
    bars, symbols, k = 3, 9, 4
    exposures = rng.normal(size=(bars, symbols, k))
    root = rng.normal(size=(bars, k, k))
    factor = None if identity else root @ np.swapaxes(root, 1, 2)
    return LowRankCovariance(
        covered=np.ones((bars, symbols), dtype=bool),
        exposures=exposures,
        factor_covariance=factor,
        specific_variance=rng.uniform(0.01, 0.1, size=(bars, symbols)),
    )


def _dense(covariance: LowRankCovariance) -> np.ndarray:
    b = covariance.exposures
    f = covariance.factor_covariance
    if f is None:
        f = np.broadcast_to(np.eye(b.shape[2]), (b.shape[0], b.shape[2], b.shape[2]))
    return b @ f @ np.swapaxes(b, 1, 2) + np.stack([np.diag(d) for d in covariance.specific_variance])


@pytest.mark.parametrize("identity", [False, True])
def test_the_variance_is_the_dense_diagonal(identity):
    covariance = _covariance(identity)

    np.testing.assert_allclose(
        covariance.variance, np.diagonal(_dense(covariance), axis1=1, axis2=2), rtol=1e-12
    )


@pytest.mark.parametrize("identity", [False, True])
def test_times_is_the_dense_product(identity):
    covariance = _covariance(identity)
    weights = np.random.default_rng(3).normal(size=covariance.specific_variance.shape)

    np.testing.assert_allclose(
        covariance.times(weights), (_dense(covariance) @ weights[..., None])[..., 0], rtol=1e-12
    )


def test_the_variance_of_a_scaled_covariance_scales():
    covariance = _covariance(False)

    np.testing.assert_allclose(covariance.scaled(5.0).variance, 5.0 * covariance.variance, rtol=1e-12)
