"""Predictable round x layer budgets; clipping weights are separate inputs."""
import math
from dataclasses import dataclass
from .st_accountant import positive


@dataclass(frozen=True)
class ReleasePlan:
    round_index: int
    rho: dict
    clip_radius: dict
    sensitivity: dict
    noise_std: dict


class TimeLayerBudget:
    def __init__(self, names, rounds, config):
        self.names = list(names)
        if not self.names or len(set(names)) != len(names):
            raise ValueError('Require distinct layer names')
        if type(rounds) is not int or rounds < 1:
            raise ValueError('rounds must be positive integer')
        self.rounds = rounds
        allowed = {'time_weights', 'layer_weights', 'layer_decay', 'feedback',
                   'feedback_ema', 'feedback_floor', 'feedback_gain', 'time_feedback_gain'}
        if set(config) - allowed:
            raise ValueError('Unknown time/layer budget fields')
        self.time = config.get('time_weights', [1.] * rounds)
        if len(self.time) != rounds:
            raise ValueError('One public time weight is required per round')
        self.time = [positive(x, 'time weight') for x in self.time]
        self.base = self._mapping(config.get('layer_weights', dict.fromkeys(names, 1.)), 'layer weights')
        self.decay = self._mapping(config.get('layer_decay', dict.fromkeys(names, 0.)), 'layer decay', True)
        self.feedback = config.get('feedback', False)
        if type(self.feedback) is not bool:
            raise ValueError('feedback must be boolean')
        self.ema_decay = float(config.get('feedback_ema', .9))
        self.floor = positive(config.get('feedback_floor', .05), 'feedback floor')
        self.gain = float(config.get('feedback_gain', 1.))
        self.time_gain = float(config.get('time_feedback_gain', 0.))
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError('feedback_ema must be in [0,1)')
        if any(not math.isfinite(x) or not 0 <= x <= 10 for x in (self.gain, self.time_gain)):
            raise ValueError('Feedback gains must be in [0,10]')
        self.scores = dict.fromkeys(names, 0.)
        self.last_plan = 0
        self.last_observed = 0

    def _mapping(self, values, label, zero=False):
        if set(values) != set(self.names):
            raise ValueError(f'{label} must name every layer exactly')
        result = {}
        for g, value in values.items():
            if zero and value == 0:
                result[g] = 0.
            else:
                result[g] = positive(value, label)
        return result

    def plan(self, t, active, remaining, clip_radius, denominator, adjacency_factor):
        if type(t) is not int or t != self.last_plan + 1 or not 1 <= t <= self.rounds:
            raise ValueError('Budget plans must be sequential within the horizon')
        if self.feedback and self.last_observed != t - 1:
            raise ValueError('Budget feedback requires the preceding released round')
        if set(active) != set(self.names) or not any(active.values()):
            raise ValueError('Invalid active layer set')
        radii = self._mapping(clip_radius, 'clip radii')
        denominator = positive(denominator, 'fixed denominator')
        remaining = positive(remaining, 'remaining rho')
        if adjacency_factor not in (1., 2.):
            raise ValueError('Unsupported adjacency factor')
        enabled = [g for g in self.names if active[g]]
        urgency = 1.
        if self.feedback:
            urgency += self.time_gain * sum(self.scores[g] for g in enabled) / len(enabled)
        current_weight = self.time[t-1] * urgency
        # Recede over the remaining public schedule. Historical feedback can
        # move unspent quota toward the current round or leave it for later.
        share = current_weight / math.fsum([current_weight, *self.time[t:]])
        # A tiny numerical reserve prevents roundoff exceeding the strict cap.
        round_rho = remaining * share * (1 - 1e-10)
        logs = {g: math.log(self.base[g]) - self.decay[g] * (t-1)
                + (self.gain * math.log(self.floor + self.scores[g]) if self.feedback else 0.)
                for g in enabled}
        peak = max(logs.values())
        # Positive budget on every active block: zero-budget blocks must freeze.
        values = {g: math.exp(max(-60., v - peak)) for g, v in logs.items()}
        total = math.fsum(values.values())
        rho = {g: round_rho * values[g] / total if active[g] else 0. for g in self.names}
        sensitivity = {g: adjacency_factor * radii[g] / denominator if active[g] else 0.
                       for g in self.names}
        std = {g: sensitivity[g] / math.sqrt(2 * rho[g]) if active[g] else 0.
               for g in self.names}
        if any(not math.isfinite(x) or x <= 0 for g in enabled for x in (rho[g], std[g])):
            raise ValueError('Budget or noise outside numerical range')
        self.last_plan = t
        return ReleasePlan(t, rho, {g: radii[g] if active[g] else 0. for g in self.names},
                           sensitivity, std)

    def observe(self, t, released, variance):
        """Only previously released DP aggregates, with public per-layer noise."""
        if t != self.last_plan or t != self.last_observed + 1:
            raise ValueError('Observe each completed release exactly once')
        if set(released) != set(self.names) or set(variance) != set(self.names):
            raise ValueError('Feedback layers must match the registry')
        scores = {}
        for g, value in released.items():
            energy = float(value.detach().double().square().sum())
            var = float(variance[g])
            if not math.isfinite(energy) or not math.isfinite(var) or var < 0:
                raise ValueError('Invalid released feedback')
            noise_energy = value.numel() * var
            # Bounded, heuristic signal-to-noise score; no raw client statistics.
            score = max(0., energy - noise_energy) / max(energy + noise_energy, 1e-30)
            scores[g] = self.ema_decay * self.scores[g] + (1-self.ema_decay) * score
        self.scores = scores
        self.last_observed = t
