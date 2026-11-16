import numpy as np
from functools import lru_cache
from opacus.accountants.analysis import rdp

ORDERS = np.array([1.25, 1.5, 1.75] + list(range(2, 65)) + [96, 128, 256], dtype=float)

@lru_cache(maxsize=8192)
def increment(q, sigma, steps=1):
    result = np.asarray(rdp.compute_rdp(q=float(q), noise_multiplier=float(sigma), steps=steps, orders=ORDERS))
    result.setflags(write=False)
    return result

def epsilon(vector, delta):
    if not np.any(vector):
        return 0.0
    return float(rdp.get_privacy_spent(orders=ORDERS, rdp=vector, delta=delta)[0])

@lru_cache(maxsize=4096)
def calibrate(target, delta, q, steps):
    if target <= 0 or steps < 1 or not 0 < q <= 1:
        raise ValueError('Invalid privacy calibration parameters')
    lo, hi = 0.001, 1.0
    while epsilon(increment(q, hi, steps), delta) > target:
        hi *= 2
        if hi > 1e8:
            raise ValueError('Cannot calibrate noise')
    for _ in range(45):
        mid = (lo + hi) / 2
        if epsilon(increment(q, mid, steps), delta) > target:
            lo = mid
        else:
            hi = mid
    return hi

def calibrate_remaining(target, delta, q, steps, spent):
    """Calibrate the whole remaining RDP trajectory, not epsilon differences."""
    lo, hi = .001, 1.
    while epsilon(spent+increment(q, hi, steps), delta) > target:
        hi *= 2
        if hi > 1e8:
            raise ValueError('No feasible remaining RDP budget')
    for _ in range(45):
        mid = (lo+hi)/2
        if epsilon(spent+increment(q, mid, steps), delta) > target:
            lo = mid
        else:
            hi = mid
    return hi

class Ledger:
    def __init__(self, targets, delta, mode='aggregate_only', adjacency='add_remove_client_contribution'):
        if mode != 'aggregate_only' or adjacency != 'add_remove_client_contribution':
            raise ValueError('Accountant supports aggregate-only add/remove contribution adjacency only')
        self.targets = np.array(targets)
        self.delta = delta
        self.rdp = np.zeros((len(targets), len(ORDERS)))

    def epsilons(self):
        return np.array([epsilon(v, self.delta) for v in self.rdp])

    def record(self, params):
        candidate = self.rdp + np.array([increment(q, s) for q, s in zip(params.q_n, params.sigma_n)])
        if any(epsilon(v, self.delta) > e + 1e-6 for v, e in zip(candidate, self.targets)):
            raise RuntimeError('Privacy budget exceeded; release aborted')
        self.rdp = candidate
