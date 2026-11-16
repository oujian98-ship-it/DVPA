"""Symmetric scalar soft recycling; private randomness is supplied explicitly.

Research floating-point sampler, not a constant-time or cryptographic DP runtime.
The returned value is noise. Never publish it or its component-selection bits.
"""
import math

import numpy as np
from scipy.special import ndtri_exp
from scipy.stats import norm, truncnorm

from .st_accountant import positive


def parameters(sigma, tau, recycle_prob):
    sigma = positive(sigma, 'sigma')
    tau = positive(tau, 'tau')
    if not .05 <= tau <= 8:
        raise ValueError('This implementation supports tau in [0.05, 8]')
    if isinstance(recycle_prob, bool) or not isinstance(recycle_prob, (int, float)):
        raise ValueError('recycle_prob must be a number in [0,1)')
    r = float(recycle_prob)
    if not math.isfinite(r) or not 0 <= r < 1:
        raise ValueError('recycle_prob must be in [0,1)')
    p_out = math.erfc(tau / math.sqrt(2))
    log_z = math.log1p(-r * p_out)
    return sigma, tau, r, p_out, log_z


def moments(sigma, tau, recycle_prob):
    sigma, tau, r, p_out, log_z = parameters(sigma, tau, recycle_prob)
    z = math.exp(log_z)
    variance = sigma**2 * (1-r*(p_out+2*tau*norm.pdf(tau))) / z
    return dict(mean=0., variance=float(variance), normalizer=z,
                log_normalizer=log_z, outside_probability=(1-r)*p_out/z)


def absolute_tail(threshold, sigma, tau, recycle_prob):
    sigma, tau, r, p_out, log_z = parameters(sigma, tau, recycle_prob)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError('threshold must be finite and nonnegative')
    x = threshold / sigma
    if x >= tau:
        result = (1-r)*2*norm.sf(x) / math.exp(log_z)
    else:
        result = (2*(norm.sf(x)-norm.sf(tau))+(1-r)*p_out) / math.exp(log_z)
    return min(1., max(0., float(result)))


def log_density(value, sigma, tau, recycle_prob):
    sigma, tau, r, _, log_z = parameters(sigma, tau, recycle_prob)
    values = np.asarray(value, dtype=np.float64)
    weight = np.where(np.abs(values) <= tau*sigma, 0., math.log1p(-r))
    return norm.logpdf(values, scale=sigma) + weight - log_z


def sample_scalar_noise(sigma, tau, recycle_prob, rng, size=None):
    """Direct mixture, equivalent to unlimited soft-rejection sampling.

    No finite retry cap, public RNG, or hard clipping of sampled noise is used.
    ndtri_exp evaluates the tail inverse without subtracting from one.
    """
    sigma, tau, r, p_out, log_z = parameters(sigma, tau, recycle_prob)
    if not isinstance(rng, np.random.Generator):
        raise ValueError('Supply a private numpy Generator explicitly')
    if r == 0:
        return rng.normal(0., sigma, size=size)
    shape = () if size is None else ((size,) if isinstance(size, int) else tuple(size))
    count = math.prod(shape) if shape else 1
    inside = rng.random(count) < (1-p_out)/math.exp(log_z)
    u = np.maximum(rng.random(count), np.nextafter(0., 1.))
    result = np.empty(count, dtype=np.float64)
    result[inside] = truncnorm.ppf(u[inside], -tau, tau)
    outside = ~inside
    if outside.any():
        magnitude = -ndtri_exp(norm.logsf(tau) + np.log(u[outside]))
        signs = np.where(rng.random(int(outside.sum())) < .5, -1., 1.)
        result[outside] = signs*magnitude
    result *= sigma
    if not np.isfinite(result).all():
        raise RuntimeError('Scalar sampler produced a nonfinite value')
    return float(result[0]) if size is None else result.reshape(shape)
