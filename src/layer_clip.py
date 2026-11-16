import math
import torch

def validate(active, weights):
    if not any(active.values()):
        raise ValueError('At least one layer must remain active')
    if set(weights) != set(active) or any(not math.isfinite(v) or v < 0 or (not active[k] and v != 0) for k, v in weights.items()):
        raise ValueError('Invalid layer weights')
    if not math.isclose(sum(weights.values()), 1., abs_tol=1e-8):
        raise ValueError('Weights must sum to one')

def clip(delta, radius, active, weights, mode='equal_active'):
    validate(active, weights)
    if radius < 0 or not math.isfinite(radius):
        raise ValueError('Invalid clipping radius')
    if any(not torch.isfinite(v).all() for v in delta.values()):
        raise ValueError('Non-finite update')
    if mode == 'original':
        norm = sum(v.double().square().sum() for g, v in delta.items() if active[g]).sqrt().item()
        factor = min(1., radius/(norm+1e-12))
        return {g: v*factor if active[g] else torch.zeros_like(v) for g, v in delta.items()}
    return {g: v*min(1., radius*math.sqrt(weights[g])/(v.double().norm().item()+1e-12)) if active[g] else torch.zeros_like(v) for g, v in delta.items()}
