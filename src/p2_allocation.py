"""P2 public allocation, independent of client updates and privacy accounting."""
import math
from collections.abc import Mapping
import numpy as np

PUBLIC_METHODS = {'equal_active', 'param_count', 'exponential_depth_time', 'prior_decay_4A'}


def validate_allocation(config):
    if not isinstance(config, dict) or config.get('layer_clip') not in PUBLIC_METHODS:
        raise ValueError('P2 requires a public allocation; oracle_moment belongs offline')
    allowed={'layer_clip','depth_rate','time_scale','rho','prior_amplitude','prior_decay','prior_source'}
    if set(config)-allowed:
        raise ValueError('Unknown allocation fields')
    if config['layer_clip']=='prior_decay_4A':
        if config.get('prior_source') not in {'assumed','public_information','independent_public_data'}:
            raise ValueError('Declare prior_source; private-data tuning is not a public prior')
        if not {'prior_amplitude','prior_decay'} <= set(config):
            raise ValueError('4A requires explicit amplitude and decay priors')


def _vector(value, names, label):
    if isinstance(value, Mapping):
        if set(value)!=set(names):raise ValueError(f'{label} must name every layer')
        value=[value[g] for g in names]
    array=np.asarray(value,dtype=float)
    if array.shape!=(len(names),) or not np.isfinite(array).all():
        raise ValueError(f'{label} requires one finite value per layer')
    return array


def weights(names, dimensions, active, t, config):
    names=list(names)
    if not names or len(set(names))!=len(names) or set(active)!=set(names) or set(dimensions)!=set(names):
        raise ValueError('Layer registry, dimensions and active mask must match')
    if not math.isfinite(t) or t<0:raise ValueError('Time must be finite and nonnegative')
    dims=_vector(dimensions,names,'dimensions')
    enabled=np.array([bool(active[g]) for g in names])
    if (dims<0).any() or (dims[enabled]<=0).any() or not enabled.any():
        raise ValueError('No valid active dimensions')
    method=config.get('layer_clip','equal_active')
    log_score=np.zeros(len(names))
    if method in {'original','equal_active'}:pass  # legacy prototype alias
    elif method=='param_count':log_score[enabled]=np.log(dims[enabled])
    elif method=='exponential_depth_time':
        rate=float(config.get('depth_rate',1.));scale=float(config.get('time_scale',25.))
        if not math.isfinite(rate) or not math.isfinite(scale) or scale<=0:
            raise ValueError('Invalid depth/time hyperparameters')
        # Absolute zero-based block depth; do not reindex after freezing.
        log_score=np.arange(len(names),dtype=float)*rate*(t/scale)
    elif method=='prior_decay_4A':
        a=_vector(config.get('prior_amplitude',[1.]*len(names)),names,'prior_amplitude')
        gamma=_vector(config.get('prior_decay',[.1]*len(names)),names,'prior_decay')
        rho=float(config.get('rho',2.))
        if (a<=0).any() or (gamma<0).any() or not math.isfinite(rho) or rho<=0:
            raise ValueError('Invalid public prior')
        log_score=(np.log(a)-gamma*t)*(2/(rho+1))
    else:raise ValueError(f'Unknown public allocation: {method}; oracle belongs offline')
    if not np.isfinite(log_score[enabled]).all():raise ValueError('Allocation log scores overflowed')
    score=np.zeros(len(names))
    # Stable softmax: clipping logs to [-100,100] would flatten distinct priors.
    with np.errstate(under='ignore'):
        score[enabled]=np.exp(log_score[enabled]-log_score[enabled].max())
    return dict(zip(names,(score/score.sum()).tolist()))


def allocation_plan(config):
    """Inspect all masks/weights without datasets, clients, training or noise."""
    from .upstream_runner import source_model
    from .p1_freezing import StaticMask, group_counts
    validate_allocation(config['allocation'])
    model=source_model()('fmnist',config['batch_size'])
    mask=StaticMask(model,config['mode'],config['freeze_rounds'],config.get('mask_seed',config['seed']))
    rows=[]
    for t in range(1,config['rounds']+1):
        counts=group_counts(mask.decide(t))
        omega=weights(mask.names,counts,{g:bool(d) for g,d in counts.items()},t,config['allocation'])
        for g in mask.names:
            rows.append(dict(round=t,layer=g,active_params=counts[g],weight=omega[g],
                             clip_radius_fraction=math.sqrt(omega[g])))
    return dict(stage='P2',scope='configuration preview only; no training or privacy release',
                allocation=config['allocation'],dimensions=mask.dims,rows=rows)
