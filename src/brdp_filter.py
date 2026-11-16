"""General fixed-order conditional RDP filter, debited BEFORE private access."""
import math

from .brdp_rdp import verify_certificate
from .st_accountant import positive


class FixedOrderRDPFilter:
    def __init__(self, epsilon, delta, adjacency='add_remove_client_contribution', alpha=None):
        self.target = positive(epsilon, 'epsilon')
        self.delta = positive(delta, 'delta')
        if self.delta >= 1:
            raise ValueError('delta must be below one')
        if adjacency not in {'add_remove_client_contribution', 'replace_client'}:
            raise ValueError('Unsupported client adjacency')
        self.adjacency = adjacency
        self.adjacency_factor = 1. if adjacency == 'add_remove_client_contribution' else 2.
        log_delta = math.log(1/self.delta)
        root = epsilon/(math.sqrt(log_delta+epsilon)+math.sqrt(log_delta))
        self.alpha = 1+math.sqrt(log_delta)/root if alpha is None else positive(alpha, 'alpha')
        if self.alpha <= 1:
            raise ValueError('alpha must exceed one')
        self.conversion = log_delta/(self.alpha-1)
        self.cap = self.target-self.conversion
        if not math.isfinite(self.cap) or self.cap <= 0:
            raise ValueError('Preselected alpha leaves no RDP quota')
        self.spent = 0.
        self.history = []

    @property
    def remaining(self):
        return max(0., self.cap-self.spent)

    def accept(self, plan):
        if type(plan.round_index) is not int or plan.round_index != len(self.history)+1:
            raise ValueError('Releases must be sequential and unique')
        if not plan.layers or len({p.layer for p in plan.layers}) != len(plan.layers):
            raise ValueError('Layer coverage must be nonempty and unique')
        costs = []
        for layer in plan.layers:
            if type(layer.active) is not bool or type(layer.dimension) is not int or layer.dimension < 1:
                raise ValueError('Invalid layer activity or dimension')
            if not layer.active:
                if layer.certificate is not None or layer.clip_radius != 0:
                    raise ValueError('Frozen layer must have no new release')
                continue
            certificate = layer.certificate
            if certificate is None or certificate.alpha != self.alpha:
                raise ValueError('Missing certificate or changed RDP order')
            if certificate.dimension != layer.dimension:
                raise ValueError('Certificate dimension differs from released layer')
            expected = self.adjacency_factor*positive(layer.clip_radius, 'clip radius')/positive(layer.denominator, 'denominator')
            if certificate.sensitivity != expected:
                raise ValueError('Certificate sensitivity does not match clipped aggregate')
            if type(layer.direction_source_round) is not int or layer.direction_source_round != plan.round_index-1:
                raise ValueError('Current/future direction used for current release')
            if certificate.recycle_prob > 0 and layer.direction_hash == 'none':
                raise ValueError('Non-Gaussian release requires a past-DP direction')
            costs.append(verify_certificate(certificate))
        increment = math.fsum(costs)
        if increment <= 0:
            raise ValueError('At least one sensitive active release required')
        total = math.fsum([self.spent, increment])
        if total > self.cap:
            raise RuntimeError('Privacy filter rejected release: total budget exceeded')
        self.spent = total
        row = dict(round=plan.round_index, alpha=self.alpha, rdp_increment=increment,
                   rdp_charged_total=total, rdp_remaining=self.remaining,
                   rdp_filter_cap=self.cap, epsilon_target=self.target, delta=self.delta,
                   epsilon_guarantee=self.target, epsilon_path_conversion=total+self.conversion)
        self.history.append(row)
        return row.copy()
