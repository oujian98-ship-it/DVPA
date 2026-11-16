"""Trusted-server client-level DP aggregate with a predictable noise direction."""
import math

import torch

from .brdp_direction import direction_hash
from .brdp_noise import sample_scalar_noise
from .brdp_rdp import verify_certificate


def release_directional(private_sum, plan, directions, scalar_rng, orthogonal_rng):
    if set(private_sum) != {p.layer for p in plan.layers} or set(directions) != set(private_sum):
        raise ValueError('Release layer coverage mismatch')
    released = {}
    for p in plan.layers:
        value = private_sum[p.layer]
        if value.numel() != p.dimension or value.ndim != 1 or not torch.isfinite(value).all():
            raise ValueError('Invalid private aggregate shape or numerics')
        if direction_hash(directions[p.layer]) != p.direction_hash:
            raise ValueError('Direction changed after certification')
        if not p.active:
            if p.certificate is not None or p.clip_radius != 0:
                raise ValueError('Frozen layer has a release certificate')
            released[p.layer] = torch.zeros_like(value)
            continue
        cert = p.certificate
        verify_certificate(cert)
        if cert.dimension != p.dimension:
            raise ValueError('Certificate dimension differs from actual release')
        if not math.isfinite(p.denominator) or p.denominator <= 0:
            raise ValueError('Public denominator must be positive')
        # Compute noise geometry in float64, then cast to the model dtype.
        g = torch.randn(value.shape, generator=orthogonal_rng,
                        device=value.device, dtype=torch.float64)*cert.sigma_orth
        if cert.recycle_prob == 0 and cert.sigma_parallel == cert.sigma_orth:
            noise = g
        else:
            u = directions[p.layer]
            if u is None:
                raise ValueError('Soft recycling requires a past-DP direction')
            u = u.to(device=value.device, dtype=torch.float64)
            if u.shape != value.shape or not torch.isfinite(u).all() or abs(float(u.norm())-1) > 1e-10:
                raise ValueError('Expected a finite unit direction')
            z = sample_scalar_noise(cert.sigma_parallel, cert.tau, cert.recycle_prob, scalar_rng)
            noise = g-torch.dot(g, u)*u+z*u
        result = (value.double()/p.denominator+noise).to(value.dtype)
        if not torch.isfinite(result).all():
            raise RuntimeError('Nonfinite DP output; terminate without publishing it')
        released[p.layer] = result
    return released
