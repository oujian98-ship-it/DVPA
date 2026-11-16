"""Trusted offline utilities. Outputs are private oracle diagnostics, not DP logs."""
import math
import numpy as np
import torch
from .p1_freezing import group_counts, joint_group_clip
from .layer_clip import validate


def oracle_moment_weights(second_moments, active, rho=2., *, offline_only=False):
    """Use A_k=sqrt(E||u_k||²), hence omega proportional to M_k^(1/(rho+1))."""
    if not offline_only:raise ValueError('oracle_moment requires explicit offline_only=True')
    if set(second_moments)!=set(active) or not any(active.values()):raise ValueError('Invalid oracle registry')
    if not math.isfinite(rho) or rho<=0:raise ValueError('Invalid rho')
    if any(not math.isfinite(v) or v<0 for v in second_moments.values()):raise ValueError('Invalid second moments')
    names=list(active);scores=np.full(len(names),-np.inf)
    for i,g in enumerate(names):
        if active[g] and second_moments[g]>0:scores[i]=math.log(second_moments[g])/(rho+1)
    if not np.isfinite(scores).any():
        return {g:float(active[g])/sum(bool(v) for v in active.values()) for g in names}
    values=np.exp(scores-np.max(scores))
    return dict(zip(names,(values/values.sum()).tolist()))


def clipping_diagnostics(updates, masks, radius, weights, *, offline_only=False):
    """Per-client block clipping factors/errors for the SAME input update.

    Does not estimate shadow unfrozen local-training drift. Averaging was_clipped
    over clients gives layer clipping frequency; aggregate signed errors separately
    to estimate bias of a mean (mean squared errors are not squared mean bias).
    """
    if not offline_only:raise ValueError('Raw diagnostics require offline_only=True')
    if set(updates)!=set(masks) or any(not torch.isfinite(v).all() for v in updates.values()):
        raise ValueError('Diagnostics require finite, registered updates')
    counts=group_counts(masks);validate({g:bool(d) for g,d in counts.items()},weights)
    clipped={n:v.detach().clone() for n,v in updates.items()}
    joint_group_clip(clipped,radius,masks,weights)
    result={}
    for g in counts:
        norm_sq=clip_error_sq=freeze_error_sq=0.
        for n,v in updates.items():
            if n.split('.')[0]!=g:continue
            m=masks[n].to(v.device)
            norm_sq+=v[m].double().square().sum().item()
            clip_error_sq+=(clipped[n][m]-v[m]).double().square().sum().item()
            freeze_error_sq+=v[~m].double().square().sum().item()
        norm=math.sqrt(norm_sq);threshold=radius*math.sqrt(weights[g])
        result[g]=dict(active_update_norm=norm,clip_radius=threshold,
                       was_clipped=bool(norm>threshold),clip_factor=min(1.,threshold/(norm+1e-12)),
                       clip_error_sq=clip_error_sq,freeze_error_sq=freeze_error_sq)
    return dict(scope='private offline diagnostics; never a DP release',layers=result,clipped=clipped)
