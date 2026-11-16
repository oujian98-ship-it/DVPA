"""Predictable round x layer RDP allocation and immutable noise plans."""
from dataclasses import dataclass
import math

from .brdp_direction import direction_hash
from .brdp_rdp import MechanismCertificate, select_certificate
from .st_budget import TimeLayerBudget


@dataclass(frozen=True)
class LayerMechanismPlan:
    layer: str
    active: bool
    dimension: int
    clip_radius: float
    denominator: float
    direction_source_round: int
    direction_hash: str
    certificate: MechanismCertificate | None
    selection_reason: str


@dataclass(frozen=True)
class DirectionalReleasePlan:
    round_index: int
    layers: tuple[LayerMechanismPlan, ...]

    @property
    def clip_radius(self):
        return {p.layer: p.clip_radius for p in self.layers}

    @property
    def expected_noise_energy(self):
        return {p.layer: p.certificate.expected_noise_energy if p.active else 0. for p in self.layers}


class DirectionalBudget:
    def __init__(self, dimensions, rounds, config, mechanism, alpha, adjacency_factor):
        self.dimensions = dict(dimensions)
        self.names = list(dimensions)
        # Reuse strict validation of the public schedule, not its Gaussian cost.
        validated = TimeLayerBudget(self.names, rounds, config)
        self.rounds, self.time = rounds, validated.time
        self.base, self.decay = validated.base, validated.decay
        self.feedback, self.ema_decay = validated.feedback, validated.ema_decay
        self.gain, self.time_gain = validated.gain, validated.time_gain
        self.floor = validated.floor
        self.mechanism = mechanism
        self.alpha, self.adjacency_factor = alpha, adjacency_factor
        self.scores = dict.fromkeys(self.names, 0.)
        self.last_plan = self.last_observed = 0

    def plan(self, t, active, remaining, clip_radius, denominator, directions):
        if t != self.last_plan+1 or t != self.last_observed+1 or t > self.rounds:
            raise ValueError('Plans require the preceding released round')
        if any(set(x) != set(self.names) for x in (active, clip_radius, directions)) or not any(active.values()):
            raise ValueError('Invalid layer coverage')
        if not math.isfinite(remaining) or remaining <= 0 or not math.isfinite(denominator) or denominator <= 0:
            raise ValueError('Positive finite quota and public denominator required')
        enabled = [g for g in self.names if active[g]]
        urgency = 1+self.time_gain*sum(self.scores[g] for g in enabled)/len(enabled) if self.feedback else 1.
        weight = self.time[t-1]*urgency
        quota = remaining*weight/math.fsum([weight, *self.time[t:]])*(1-1e-10)
        logs = {g: math.log(self.base[g])-self.decay[g]*(t-1)
                +(self.gain*math.log(self.floor+self.scores[g]) if self.feedback else 0.) for g in enabled}
        peak = max(logs.values())
        weights = {g: math.exp(max(-60., logs[g]-peak)) for g in enabled}
        total = math.fsum(weights.values())
        layers = []
        for g in self.names:
            radius = float(clip_radius[g]) if active[g] else 0.
            cert, reason = None, 'FROZEN'
            if active[g]:
                if not math.isfinite(radius) or radius <= 0:
                    raise ValueError('Active clip radius must be positive')
                cert, reason = select_certificate(
                    self.alpha, self.adjacency_factor*radius/denominator,
                    self.dimensions[g], quota*weights[g]/total,
                    self.mechanism['noise'], self.mechanism['selection'],
                    allow_recycling=t > self.mechanism['direction']['warmup_rounds'] and directions[g] is not None)
            layers.append(LayerMechanismPlan(g, bool(active[g]), self.dimensions[g], radius,
                          denominator, t-1, direction_hash(directions[g]), cert, reason))
        self.last_plan = t
        return DirectionalReleasePlan(t, tuple(layers))

    def observe(self, t, released, expected_noise_energy):
        if t != self.last_plan or t != self.last_observed+1:
            raise ValueError('Observe each accepted release exactly once')
        if set(released) != set(self.names) or set(expected_noise_energy) != set(self.names):
            raise ValueError('Observation coverage mismatch')
        scores = {}
        for g, value in released.items():
            energy = float(value.detach().double().square().sum())
            noise = float(expected_noise_energy[g])
            if not math.isfinite(energy) or not math.isfinite(noise) or noise < 0:
                raise ValueError('Invalid DP observation or expected noise moment')
            # Bounded heuristic, not a confidence interval or unbiased utility.
            score = max(0., energy-noise)/max(energy+noise, 1e-30)
            scores[g] = self.ema_decay*self.scores[g]+(1-self.ema_decay)*score
        self.scores = scores
        self.last_observed = t

    def state_dict(self):
        return dict(scores=self.scores.copy(), last_plan=self.last_plan,
                    last_observed=self.last_observed)
