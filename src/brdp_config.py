"""Strict configuration for the separate BR-DP federated runner."""
import copy
import math

from .brdp_direction import DirectionState
from .brdp_filter import FixedOrderRDPFilter
from .brdp_noise import parameters
from .brdp_rdp import BOUND_MODE
from .models import build_model
from .layer_registry import LayerRegistry
from .st_budget import TimeLayerBudget
from .st_runner import validate_config as validate_st


def default_mechanism():
    return dict(name='directional_soft_recycling', version=1,
        direction=dict(source='past_dp_update_ema', ema_decay=.9, norm_floor=1e-12,
                       warmup_rounds=3, rank=1, invalid_direction_policy='gaussian'),
        noise=dict(sampler='direct_truncated_mixture', recycle_prob_candidates=[0., .005, .01, .02, .04],
                   tau_candidates=[.5, 1., 2.], isotropic_base=True),
        accounting=dict(family='fixed_order_rdp', alpha_policy='preselected_before_training',
                        alpha=None, bound_mode=BOUND_MODE, require_certificate=True,
                        sampling_amplification=False),
        selection=dict(policy='public_offline_calibration', fallback='gaussian',
                       threshold_units='aggregate_update', tail_threshold_sigma=2.,
                       max_energy_ratio=1.05, max_parallel_variance_ratio=1.05,
                       fixed_recycle_prob=.005, fixed_tau=1.),
        output=dict(save_released_dp_updates=True, save_raw_updates=False,
                    save_noise_or_rng_state=False))


def _merge_known(base, given, label):
    if not isinstance(given, dict) or set(given)-set(base):
        raise ValueError(f'Unknown fields or nonmapping in {label}')
    result = copy.deepcopy(base)
    for key, value in given.items():
        if isinstance(base[key], dict):
            result[key] = _merge_known(base[key], value, f'{label}.{key}')
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_config(config):
    if not isinstance(config, dict) or 'mechanism' not in config:
        raise ValueError('BR-DP runner requires a mechanism section')
    cfg = validate_st({k: v for k, v in config.items() if k != 'mechanism'})
    m = _merge_known(default_mechanism(), config['mechanism'], 'mechanism')
    if m['name'] != 'directional_soft_recycling' or type(m['version']) is not int or m['version'] != 1:
        raise ValueError('Unsupported BR-DP mechanism/version')
    d, noise, acc, sel, out = (m[k] for k in ('direction', 'noise', 'accounting', 'selection', 'output'))
    if d['source'] != 'past_dp_update_ema' or d['invalid_direction_policy'] != 'gaussian' or type(d['rank']) is not int or d['rank'] != 1:
        raise ValueError('Only a single past-DP direction with Gaussian fallback is supported')
    if type(d['warmup_rounds']) is not int or not 1 <= d['warmup_rounds'] <= cfg['train']['rounds']:
        raise ValueError('warmup_rounds must be within the horizon and at least one')
    DirectionState({'validation': 1}, d['ema_decay'], d['norm_floor'])
    if noise['sampler'] != 'direct_truncated_mixture' or noise['isotropic_base'] is not True:
        raise ValueError('V1 supports direct mixture and isotropic base only')
    if not isinstance(noise['recycle_prob_candidates'], list) or not noise['recycle_prob_candidates'] or 0. not in noise['recycle_prob_candidates']:
        raise ValueError('Candidate list must include Gaussian r=0')
    if not isinstance(noise['tau_candidates'], list) or not noise['tau_candidates']:
        raise ValueError('Nonempty tau list required')
    for r in noise['recycle_prob_candidates']:
        for tau in noise['tau_candidates']:
            parameters(1., tau, r)
    if acc['family'] != 'fixed_order_rdp' or acc['alpha_policy'] != 'preselected_before_training' or acc['bound_mode'] != BOUND_MODE:
        raise ValueError('Only the analytic conservative fixed-order RDP accountant is implemented')
    if acc['require_certificate'] is not True or acc['sampling_amplification'] is not False:
        raise ValueError('Certificates required; sampling amplification unsupported')
    FixedOrderRDPFilter(cfg['privacy']['epsilon'], cfg['privacy']['delta'],
                        cfg['privacy']['adjacency'], acc['alpha'])
    if sel['policy'] not in {'public_offline_calibration', 'fixed_candidate'} or sel['fallback'] != 'gaussian' or sel['threshold_units'] != 'aggregate_update':
        raise ValueError('Unsupported public mechanism-selection policy')
    for key in ('tail_threshold_sigma', 'max_energy_ratio', 'max_parallel_variance_ratio'):
        value = sel[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite')
    parameters(1., sel['fixed_tau'], sel['fixed_recycle_prob'])
    if out != dict(save_released_dp_updates=True, save_raw_updates=False, save_noise_or_rng_state=False):
        raise ValueError('Replay audit requires DP updates; raw/noise/RNG artifacts prohibited')
    registry = LayerRegistry(build_model(cfg))
    TimeLayerBudget(registry.names, cfg['train']['rounds'], cfg['budget'])
    if set(cfg['clipping']['weights']) != set(registry.names):
        raise ValueError('Clipping weights must match all model layers')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in cfg['clipping']['weights'].values()):
        raise ValueError('Positive clipping weights required')
    events = cfg['freezing']['freeze_rounds']
    if set(events)-set(registry.names) or len(events) >= len(registry.names):
        raise ValueError('Invalid freeze layer set')
    if any(type(t) is not int or not 1 <= t <= cfg['train']['rounds'] for t in events.values()):
        raise ValueError('Freeze rounds must lie within horizon')
    cfg['mechanism'] = m
    return cfg
