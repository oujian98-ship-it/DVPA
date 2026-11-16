from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.integrate import quad

from src.brdp_budget import DirectionalReleasePlan, LayerMechanismPlan
from src.brdp_filter import FixedOrderRDPFilter
from src.brdp_noise import log_density
from src.brdp_rdp import (BoundTooLoose, calibrate, conservative_rdp,
    joint_grid_diagnostic, scalar_rdp_diagnostic, verify_certificate)


@pytest.mark.parametrize('alpha,tau,r,h', [(2., .5, .4, .7), (3.725, 1., .1, 1.2), (8., 2., .01, .1)])
def test_scalar_piecewise_expression_against_independent_quadrature(alpha, tau, r, h):
    sigma = 1.3
    cuts = [-math.inf, *sorted({-sigma*tau, sigma*tau, -h-sigma*tau, -h+sigma*tau}), math.inf]
    def integrand(z):
        return math.exp(alpha*float(log_density(z, sigma, tau, r))
                        +(1-alpha)*float(log_density(z+h, sigma, tau, r)))
    value = sum(quad(integrand, lo, hi, epsabs=1e-11)[0] for lo, hi in zip(cuts, cuts[1:]))
    assert scalar_rdp_diagnostic(alpha, sigma, tau, r, h) == pytest.approx(math.log(value)/(alpha-1), abs=1e-9)


def test_conservative_bound_and_gaussian_limits_cover_offsets():
    for r in [0., .01, .1, .4]:
        for tau in [.5, 1., 2.]:
            for h in np.linspace(0, 1.2, 21):
                cost = scalar_rdp_diagnostic(3.725, 1.3, tau, r, h)
                assert cost <= conservative_rdp(3.725, h, 1, 1.3, 1.3, tau, r)+1e-10
                assert cost == pytest.approx(scalar_rdp_diagnostic(3.725, 1.3, tau, r, -h))
    assert scalar_rdp_diagnostic(4., 2., 1., 0., .7) == pytest.approx(4*.7**2/8)
    assert scalar_rdp_diagnostic(4., 2., 1., .5, 0.) == 0
    grid = joint_grid_diagnostic(4., 1., 2, 2., 3., 1., 0.)
    assert grid['certified'] is False and 'rdp_upper' not in grid
    assert grid['diagnostic_grid_max'] == pytest.approx(4/8)


def test_calibration_quota_and_conservative_failure_is_named():
    c = calibrate(3.725, .5, 500, .2, .01, 1.)
    assert verify_certificate(c) <= c.rdp_budget
    g = calibrate(3.725, .5, 500, .2)
    assert c.sigma_parallel > g.sigma_parallel
    with pytest.raises(BoundTooLoose, match='BOUND_TOO_LOOSE'):
        calibrate(3.725, 1., 10, .02, .1, 1.)
    with pytest.raises(ValueError):
        verify_certificate(replace(c, sigma_parallel=c.sigma_parallel/2))
    with pytest.raises(ValueError):
        verify_certificate(replace(c, bound_mode='grid_maximum'))


def _plan(ledger, budget, t=1):
    c = calibrate(ledger.alpha, .25, 2, budget, .01, 1.)
    p = LayerMechanismPlan('a', True, 2, .25/ledger.adjacency_factor, 1., t-1, 'test_direction_hash', c, 'test')
    return DirectionalReleasePlan(t, (p,))


def test_filter_is_atomic_and_verifies_actual_params():
    ledger = FixedOrderRDPFilter(10., 1e-5)
    with pytest.raises(RuntimeError, match='budget exceeded'):
        ledger.accept(_plan(ledger, ledger.cap*2))
    assert ledger.spent == 0 and ledger.history == []
    plan = _plan(ledger, ledger.cap/3)
    tampered = replace(plan, layers=(replace(plan.layers[0], certificate=replace(plan.layers[0].certificate, recycle_prob=.2)),))
    with pytest.raises(ValueError):
        ledger.accept(tampered)
    assert ledger.spent == 0
    row = ledger.accept(plan)
    assert row['epsilon_guarantee'] == 10
    assert row['epsilon_path_conversion'] < 10
    with pytest.raises(ValueError, match='sequential'):
        ledger.accept(plan)
    future = _plan(ledger, ledger.cap/4, 2)
    with pytest.raises(ValueError, match='Current/future'):
        ledger.accept(replace(future, layers=(replace(future.layers[0], direction_source_round=2),)))


def test_unsupported_order_and_adjacency():
    with pytest.raises(ValueError):
        FixedOrderRDPFilter(1., 1e-5, alpha=2.)
    with pytest.raises(ValueError):
        FixedOrderRDPFilter(10., 1e-5, adjacency='local_dp')


def test_vector_cannot_claim_scalar_sensitivity_geometry():
    ledger = FixedOrderRDPFilter(10., 1e-5)
    plan = _plan(ledger, ledger.cap/3)
    scalar = replace(plan.layers[0].certificate, dimension=1)
    with pytest.raises(ValueError, match='dimension'):
        ledger.accept(replace(plan, layers=(replace(plan.layers[0], certificate=scalar),)))
    assert ledger.spent == 0


def test_vector_bound_covers_parallel_orthogonal_and_mixed_neighbor_shifts():
    for parallel, orthogonal in [(1.3, 1.3), (1.1, 2.), (2., 1.1)]:
        for r in [.005, .1, .4]:
            bound = conservative_rdp(3.725, 1.2, 2, parallel, orthogonal, 1., r)
            grid = joint_grid_diagnostic(3.725, 1.2, 2, parallel, orthogonal, 1., r, 129)
            assert grid['diagnostic_grid_max'] <= bound+1e-10
