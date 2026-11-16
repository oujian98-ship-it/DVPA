"""Trusted-server, independent block Gaussian noise on a fixed-denominator sum."""
import math
import torch


def clip_blocks(update, active, radii):
    if set(update) != set(active) or set(update) != set(radii):
        raise ValueError('Update, mask, and radius layers must match')
    result = {}
    for g, value in update.items():
        radius = radii[g]
        if not math.isfinite(radius) or radius < 0:
            raise ValueError('Invalid clipping radius')
        if active[g] and radius <= 0:
            raise ValueError('Active layers require positive clipping radii')
        # A bounded deterministic local map, including failed local numerics.
        # Do not expose a private-data-dependent non-finite-update exception.
        if not torch.isfinite(value).all():
            result[g] = torch.zeros_like(value)
            continue
        norm = float(torch.linalg.vector_norm(value.double()))
        # Small safety contraction keeps finite-precision clipping inside ball.
        factor = min(1., radius / max(norm, 1e-300)) * (1 - 1e-7)
        result[g] = value * factor if active[g] else torch.zeros_like(value)
    return result


def release_sum(clipped_sum, active, plan, denominator, generator):
    """Input is a trusted PRIVATE sum; return only its noised aggregate.

    The caller must run the privacy filter BEFORE calling this function.
    No selected-client count is used as a denominator.
    """
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError('Invalid public denominator')
    if set(clipped_sum) != set(active) or set(clipped_sum) != set(plan.noise_std):
        raise ValueError('Release layers must match')
    result = {}
    for g, value in clipped_sum.items():
        if not torch.isfinite(value).all():
            raise ValueError('Non-finite private aggregate')
        if not active[g]:
            if plan.sensitivity[g] != 0 or plan.noise_std[g] != 0:
                raise ValueError('Frozen layers must have no private release')
            result[g] = torch.zeros_like(value)
        else:
            if plan.noise_std[g] <= 0:
                raise ValueError('Active layer requires positive noise')
            noise = torch.randn(value.shape, generator=generator, device=value.device, dtype=value.dtype)
            result[g] = value / denominator + noise * plan.noise_std[g]
    return result
