"""Fixed-order Gaussian RDP privacy filter for adaptive time/layer allocation.

The observer sees aggregates only. Client slots and aggregation denominators
are public and fixed. Sampling amplification is deliberately NOT used.
"""
import math
from collections.abc import Mapping


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite positive number')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive number')
    return value


def gaussian_rho(sensitivity, noise_std):
    """Product-of-balls sensitivity after whitening independent block noise.

    Sensitivities refer to the FINAL aggregate, after its public denominator.
    Frozen blocks have BOTH zero sensitivity and zero noise.
    """
    if not isinstance(sensitivity, Mapping) or set(sensitivity) != set(noise_std):
        raise ValueError('Sensitivity and noise layer names must match')
    if not sensitivity:
        raise ValueError('Empty release')
    terms = []
    for name, delta in sensitivity.items():
        std = noise_std[name]
        if not math.isfinite(delta) or not math.isfinite(std) or delta < 0 or std < 0:
            raise ValueError('Sensitivity/noise must be finite and nonnegative')
        if delta == 0:
            continue
        if std == 0:
            raise ValueError('A sensitive layer cannot be released without noise')
        terms.append(0.5 * (delta / std) ** 2)
    result = math.fsum(terms)
    if not math.isfinite(result):
        raise ValueError('Non-finite Gaussian privacy cost')
    return result


class GaussianRDPFilter:
    """One PRESELECTED order; never minimize orders on an adaptive trajectory.

    Each conditional Gaussian release has RDP <= alpha * rho. Before a
    release, enforce a deterministic cap on the sum for every history.
    This is a fixed-order RDP filter, not a realized-loss odometer.
    """
    def __init__(self, epsilon, delta, adjacency='add_remove_client_contribution'):
        self.target = positive(epsilon, 'epsilon')
        self.delta = positive(delta, 'delta')
        if self.delta >= 1:
            raise ValueError('delta must be below one')
        if adjacency not in {'add_remove_client_contribution', 'replace_client'}:
            raise ValueError('Unsupported client adjacency')
        self.adjacency = adjacency
        self.adjacency_factor = 1.0 if adjacency == 'add_remove_client_contribution' else 2.0
        log_delta = math.log(1 / self.delta)
        # Stable solution of rho + 2 sqrt(rho log(1/delta)) = epsilon.
        root = self.target / (math.sqrt(log_delta + self.target) + math.sqrt(log_delta))
        self.alpha = 1 + math.sqrt(log_delta) / root
        self.rho_cap = root ** 2
        self.conversion = log_delta / (self.alpha - 1)
        if not all(math.isfinite(x) and x > 0 for x in (self.alpha, self.rho_cap)):
            raise ValueError('Privacy parameters outside numerical range')
        self.spent = 0.0
        self.history = []

    @property
    def remaining(self):
        return max(0.0, self.rho_cap - self.spent)

    @property
    def epsilon_bound(self):
        # For adaptive allocation the observed charge is NOT itself a uniform
        # bound across all histories. The filter certifies the preselected cap.
        return 0.0 if self.spent == 0 else self.target

    @property
    def epsilon_accounting_value(self):
        """Diagnostic conversion of the current charge, not an adaptive guarantee."""
        return 0.0 if self.spent == 0 else self.alpha * self.spent + self.conversion

    def record(self, round_index, sensitivity, noise_std):
        if type(round_index) is not int or round_index != len(self.history) + 1:
            raise ValueError('Privacy releases must be sequential')
        cost = gaussian_rho(sensitivity, noise_std)
        total = math.fsum([self.spent, cost])
        if total > self.rho_cap:
            raise RuntimeError('Privacy filter rejected release: total budget exceeded')
        # Mutate only after the filter has accepted the actual mechanism.
        self.spent = total
        row = dict(round=round_index, rho_increment=cost, rho_spent=total,
                   rho_remaining=self.remaining, alpha=self.alpha,
                   rdp_increment=self.alpha * cost, rdp_spent=self.alpha * total,
                   rdp_filter_cap=self.alpha * self.rho_cap,
                   epsilon_accounting_value=self.epsilon_accounting_value,
                   epsilon_bound=self.epsilon_bound, epsilon_target=self.target,
                   delta=self.delta)
        self.history.append(row)
        return row.copy()
