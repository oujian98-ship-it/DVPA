"""Analytic conservative RDP certificates and NON-certifying tight diagnostics.

For density G(v)*a(v_parallel)/Z with 1-r <= a <= 1, its RDP is at most
alpha*Delta^2/(2*min_sigma^2) - log(1-r) - log(Z)/(alpha-1).
The fixed-offset scalar diagnostic integrates its four breakpoints exactly
in real arithmetic. Floating-point grids are NEVER accepted as certificates.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
from scipy.special import log_ndtr, logsumexp

from .brdp_noise import absolute_tail, moments, parameters
from .st_accountant import positive

BOUND_MODE = 'conservative_density_ratio_v1'


def _alpha(alpha):
    alpha = positive(alpha, 'alpha')
    if alpha <= 1:
        raise ValueError('alpha must exceed one')
    return alpha


def conservative_rdp(alpha, sensitivity, dimension, sigma_parallel, sigma_orth,
                     tau, recycle_prob):
    alpha = _alpha(alpha)
    if type(dimension) is not int or dimension < 1:
        raise ValueError('dimension must be a positive integer')
    if not math.isfinite(sensitivity) or sensitivity < 0:
        raise ValueError('sensitivity must be nonnegative and finite')
    sp, _, r, _, log_z = parameters(sigma_parallel, tau, recycle_prob)
    so = positive(sigma_orth, 'sigma_orth')
    if sensitivity == 0:
        return 0.
    smallest = sp if dimension == 1 else min(sp, so)
    value = alpha/2*(sensitivity/smallest)**2 - math.log1p(-r) - log_z/(alpha-1)
    # Numerical research margin. This is not interval-certified transcendental
    # arithmetic and does not claim finite-precision/cryptographic DP security.
    return math.nextafter(value*(1+1e-12)+1e-15, math.inf)


def _logdiff(a, b):
    if b == -math.inf:
        return a
    if b >= a:
        return -math.inf
    return a + math.log(-math.expm1(b-a))


def _log_interval(lo, hi):
    if lo >= 0:
        return _logdiff(float(log_ndtr(-lo)), float(log_ndtr(-hi)))
    return _logdiff(float(log_ndtr(hi)), float(log_ndtr(lo)))


def scalar_rdp_diagnostic(alpha, sigma, tau, recycle_prob, shift):
    """Fixed-offset formula; diagnostic floating-point result, not a bound."""
    alpha = _alpha(alpha)
    sigma, tau, r, _, log_z = parameters(sigma, tau, recycle_prob)
    if not math.isfinite(shift):
        raise ValueError('shift must be finite')
    h = abs(float(shift))
    if h == 0:
        return 0.
    if r == 0:
        return alpha/2*(h/sigma)**2
    theta = sigma*tau
    points = [-math.inf, *sorted({-theta, theta, -h-theta, -h+theta}), math.inf]
    mu = (alpha-1)*h
    terms = []
    for lo, hi in zip(points, points[1:]):
        mid = hi-sigma if lo == -math.inf else (lo+sigma if hi == math.inf else (lo+hi)/2)
        log_a = 0. if abs(mid) <= theta else math.log1p(-r)
        log_b = 0. if abs(mid+h) <= theta else math.log1p(-r)
        terms.append(alpha*log_a+(1-alpha)*log_b+_log_interval((lo-mu)/sigma, (hi-mu)/sigma))
    value = alpha/2*(h/sigma)**2 + (float(logsumexp(terms))-log_z)/(alpha-1)
    if value < -1e-9 or not math.isfinite(value):
        raise ArithmeticError('Unreliable scalar RDP diagnostic')
    return max(0., value)


def joint_grid_diagnostic(alpha, sensitivity, dimension, sigma_parallel, sigma_orth,
                          tau, recycle_prob, points=257):
    if type(points) is not int or points < 2:
        raise ValueError('At least two grid points required')
    conservative_rdp(alpha, sensitivity, dimension, sigma_parallel, sigma_orth, tau, recycle_prob)
    values = []
    for h in np.linspace(0., sensitivity, points):
        value = scalar_rdp_diagnostic(alpha, sigma_parallel, tau, recycle_prob, h)
        if dimension > 1:
            value += alpha*(sensitivity**2-h**2)/(2*sigma_orth**2)
        values.append(value)
    index = int(np.argmax(values))
    return dict(certified=False, diagnostic_grid_max=float(values[index]),
                maximizing_grid_shift=float(np.linspace(0., sensitivity, points)[index]),
                points=points)


@dataclass(frozen=True)
class MechanismCertificate:
    alpha: float
    sensitivity: float
    dimension: int
    rdp_budget: float
    sigma_parallel: float
    sigma_orth: float
    recycle_prob: float
    tau: float
    rdp_upper: float
    bound_mode: str = BOUND_MODE
    status: str = 'CERTIFIED'

    @property
    def certificate_id(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
                                         allow_nan=False).encode()).hexdigest()

    @property
    def expected_noise_energy(self):
        return moments(self.sigma_parallel, self.tau, self.recycle_prob)['variance'] + (self.dimension-1)*self.sigma_orth**2


class BoundTooLoose(ValueError):
    """This certificate cannot fit; does NOT imply the actual mechanism cannot."""


def calibrate(alpha, sensitivity, dimension, rdp_budget, recycle_prob=0., tau=1.):
    """V1 isotropic base. All charging uses the analytic density-ratio bound."""
    alpha = _alpha(alpha)
    sensitivity = positive(sensitivity, 'sensitivity')
    budget = positive(rdp_budget, 'rdp_budget')
    _, _, r, _, log_z = parameters(1., tau, recycle_prob)
    penalty = -math.log1p(-r)-log_z/(alpha-1)
    available = budget*(1-1e-8)-penalty
    if available <= 0:
        raise BoundTooLoose('BOUND_TOO_LOOSE: conservative penalty exhausts unit quota')
    sigma = sensitivity*math.sqrt(alpha/(2*available))*(1+1e-9)
    cost = conservative_rdp(alpha, sensitivity, dimension, sigma, sigma, tau, r)
    if not math.isfinite(cost) or cost > budget:
        raise ArithmeticError('Numerical calibration failed to fit certified quota')
    return MechanismCertificate(alpha, sensitivity, dimension, budget, sigma, sigma,
                                r, float(tau), cost,
                                status='GAUSSIAN_FALLBACK' if r == 0 else 'CERTIFIED')


def verify_certificate(certificate):
    if not isinstance(certificate, MechanismCertificate) or certificate.bound_mode != BOUND_MODE:
        raise ValueError('Only analytic conservative certificates are supported')
    if certificate.status != ('GAUSSIAN_FALLBACK' if certificate.recycle_prob == 0 else 'CERTIFIED'):
        raise ValueError('Certificate status disagrees with mechanism')
    positive(certificate.rdp_budget, 'rdp_budget')
    cost = conservative_rdp(certificate.alpha, certificate.sensitivity, certificate.dimension,
                            certificate.sigma_parallel, certificate.sigma_orth,
                            certificate.tau, certificate.recycle_prob)
    if cost > certificate.rdp_budget or cost != certificate.rdp_upper:
        raise ValueError('Certificate cost mismatch or quota exceeded')
    return cost


def select_certificate(alpha, sensitivity, dimension, budget, noise, selection,
                       allow_recycling=True):
    baseline = calibrate(alpha, sensitivity, dimension, budget)
    if not allow_recycling:
        return baseline, 'WARMUP_OR_INVALID_DIRECTION'
    if selection['policy'] == 'fixed_candidate':
        try:
            return calibrate(alpha, sensitivity, dimension, budget,
                             selection['fixed_recycle_prob'], selection['fixed_tau']), 'FIXED_CANDIDATE'
        except BoundTooLoose:
            return baseline, 'BOUND_TOO_LOOSE_GAUSSIAN_FALLBACK'
    threshold = baseline.sigma_parallel*selection['tail_threshold_sigma']
    best, reason = baseline, 'NO_IMPROVEMENT_UNDER_CONSERVATIVE_BOUND'
    best_tail = absolute_tail(threshold, baseline.sigma_parallel, baseline.tau, 0.)
    for r in noise['recycle_prob_candidates']:
        if r == 0:
            continue
        for tau in noise['tau_candidates']:
            try:
                candidate = calibrate(alpha, sensitivity, dimension, budget, r, tau)
            except BoundTooLoose:
                continue
            parallel_variance = moments(candidate.sigma_parallel, tau, r)['variance']
            if candidate.expected_noise_energy > baseline.expected_noise_energy*selection['max_energy_ratio']:
                continue
            if parallel_variance > baseline.sigma_parallel**2*selection['max_parallel_variance_ratio']:
                continue
            tail = absolute_tail(threshold, candidate.sigma_parallel, tau, r)
            if tail < best_tail-1e-12:
                best, best_tail, reason = candidate, tail, 'PUBLIC_TAIL_OPTIMIZATION'
    return best, reason
