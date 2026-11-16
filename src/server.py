import math
import torch

def add_client_noise(delta, scale, n, active, generator):
    return {g: v + torch.randn(v.shape, dtype=v.dtype, device=v.device, generator=generator)*(scale/math.sqrt(n)) if active[g] else torch.zeros_like(v) for g, v in delta.items()}

def aggregate(messages, template, active, n, q_bar, scale, generator):
    if len(messages) > n or n <= 0 or not 0 < q_bar <= 1:
        raise ValueError('Invalid aggregation')
    total = {g: torch.zeros_like(v) for g, v in template.items()}
    for message in messages:
        for g in total:
            if active[g]:
                total[g] += message[g]
    missing_std = scale*math.sqrt((n-len(messages))/n)
    for g, v in total.items():
        if active[g]:
            v.add_(torch.randn(v.shape, dtype=v.dtype, device=v.device, generator=generator)*missing_std)
            v.div_(n*q_bar)
    return total, (scale/(n*q_bar))**2
