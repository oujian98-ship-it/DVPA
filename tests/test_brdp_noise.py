import math

import numpy as np
import pytest
from scipy.integrate import quad

from src.brdp_noise import absolute_tail, log_density, moments, sample_scalar_noise


@pytest.mark.parametrize('r,tau', [(0., 1.), (.1, .5), (.4, 1.), (.8, 2.)])
def test_density_normalizes_and_sampler_matches_analytic_moments(r, tau):
    sigma = 1.3
    cuts = [-math.inf, -sigma*tau, sigma*tau, math.inf]
    integral = sum(quad(lambda z: math.exp(float(log_density(z, sigma, tau, r))), lo, hi)[0]
                   for lo, hi in zip(cuts, cuts[1:]))
    assert integral == pytest.approx(1., abs=1e-9)
    values = sample_scalar_noise(sigma, tau, r, np.random.default_rng(42), 150000)
    theoretical = moments(sigma, tau, r)
    assert abs(values.mean()) < .015*sigma
    assert values.var() == pytest.approx(theoretical['variance'], rel=.025)
    for threshold in [sigma*.3, sigma*tau, sigma*2.]:
        assert np.mean(np.abs(values) > threshold) == pytest.approx(
            absolute_tail(threshold, sigma, tau, r), abs=.005)


def test_gaussian_limit_exact_rng_and_no_global_numpy_rng_used():
    a = sample_scalar_noise(2., 1., 0., np.random.default_rng(51), 20)
    b = np.random.default_rng(51).normal(0., 2., 20)
    assert np.array_equal(a, b)
    np.random.seed(99)
    before = np.random.get_state()
    sample_scalar_noise(1., 1., .4, np.random.default_rng(1), 30)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]


@pytest.mark.parametrize('sigma,tau,r', [(0., 1., .1), (1., .001, .1),
    (1., 9., .1), (1., 1., 1.), (1., 1., -.1), (1., 1., True), (float('nan'), 1., 0.)])
def test_invalid_distribution_rejected(sigma, tau, r):
    with pytest.raises(ValueError):
        sample_scalar_noise(sigma, tau, r, np.random.default_rng(1))


def test_extreme_supported_tails_and_shapes():
    rng = np.random.default_rng(42)
    assert sample_scalar_noise(1., .05, .99, rng, (3, 4)).shape == (3, 4)
    assert np.isfinite(sample_scalar_noise(1., 8., .99, rng, 100)).all()
    assert isinstance(sample_scalar_noise(1., 1., .1, rng), float)
    with pytest.raises(ValueError, match='private'):
        sample_scalar_noise(1., 1., .1, None)
