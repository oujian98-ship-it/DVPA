"""Directions computed ONLY from earlier released DP updates."""
import hashlib
import math

import torch


def direction_hash(vector):
    if vector is None:
        return 'none'
    return hashlib.sha256(vector.detach().cpu().double().contiguous().numpy().tobytes()).hexdigest()


class DirectionState:
    def __init__(self, dimensions, ema_decay=.9, norm_floor=1e-12):
        if not math.isfinite(ema_decay) or not 0 <= ema_decay < 1:
            raise ValueError('Direction EMA must be in [0,1)')
        if not math.isfinite(norm_floor) or norm_floor <= 0:
            raise ValueError('Direction norm floor must be positive')
        self.dimensions = dict(dimensions)
        self.decay = float(ema_decay)
        self.floor = float(norm_floor)
        self.ema = {g: torch.zeros(d, dtype=torch.float64) for g, d in dimensions.items()}
        self.last_observed = 0

    def for_round(self, t):
        if type(t) is not int or t != self.last_observed+1:
            raise ValueError('Directions require exactly the preceding DP history')
        result = {}
        for g, value in self.ema.items():
            norm = float(torch.linalg.vector_norm(value))
            result[g] = (value/norm).clone() if math.isfinite(norm) and norm > self.floor else None
        return result

    def observe(self, t, released):
        if t != self.last_observed+1 or set(released) != set(self.dimensions):
            raise ValueError('Observe released layers exactly once and sequentially')
        converted = {}
        for g, value in released.items():
            value = value.detach().cpu().double().flatten()
            if value.numel() != self.dimensions[g] or not torch.isfinite(value).all():
                raise ValueError('Invalid released direction observation')
            converted[g] = self.decay*self.ema[g]+(1-self.decay)*value
        self.ema = converted
        self.last_observed = t

    def state_dict(self):
        return dict(ema={g: x.clone() for g, x in self.ema.items()},
                    last_observed=self.last_observed)
