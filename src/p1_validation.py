"""Locked, user-requested P1 comparison; does not mark the P0 gate passed."""
import hashlib
import json
from pathlib import Path
import numpy as np

ALLOWED_MODES={'NF','BU','TD','RAND'}

def check_protocol(cfg):
    path=Path(cfg['validation_protocol'])
    protocol=json.loads(path.read_text(encoding='utf-8'))
    kind=protocol.get('kind')
    feedback_kind=kind=='controlled_feedback_comparison_v1'
    diagnostic_kind=kind=='controlled_freezing_diagnostics_v1'
    joint_kind=kind=='controlled_joint_allocation_freezing_v1'
    if kind not in {'controlled_p1_comparison_v1','controlled_allocation_comparison_v1','controlled_bu_allocation_comparison_v1','controlled_feedback_comparison_v1','controlled_freezing_diagnostics_v1','controlled_joint_allocation_freezing_v1'}:raise ValueError('Unknown validation protocol')
    ignored=({'mode','seed','mask_seed','validation_protocol'} if kind=='controlled_p1_comparison_v1' else
             {'seed','mask_seed','allocation','validation_protocol'})
    if feedback_kind:ignored={'seed','mask_seed','mode','freeze_rounds','feedback','validation_protocol'}
    if diagnostic_kind:ignored={'seed','mask_seed','mode','freeze_rounds','feedback','diagnostic_freezing','diagnostic_rng_path','static_order','validation_protocol'}
    if joint_kind:ignored={'seed','mask_seed','mode','freeze_rounds','feedback','diagnostic_freezing','diagnostic_rng_path','static_order','allocation','validation_protocol'}
    if {k:v for k,v in cfg.items() if k not in ignored}!=protocol['fixed_config']:
        raise ValueError('Run configuration differs from locked comparison protocol')
    if cfg['mode'] not in (ALLOWED_MODES|{'ADAPTIVE'} if feedback_kind or diagnostic_kind or joint_kind else ALLOWED_MODES) or cfg['seed'] not in protocol['training_seeds']:
        raise ValueError('Unregistered method or training seed')
    if feedback_kind:
        method={k:cfg[k] for k in ('mode','freeze_rounds','feedback') if k in cfg}
        if method not in protocol['methods'].values() or cfg.get('mask_seed')!=cfg['seed']:
            raise ValueError('Unregistered feedback method or mask seed')
        if cfg.get('allocation')!={'layer_clip':'equal_active'}:
            raise ValueError('4B-only requires equal_active clipping')
    if diagnostic_kind:
        fields={'mode','freeze_rounds','feedback','diagnostic_freezing','diagnostic_rng_path','static_order'}
        method={k:cfg[k] for k in fields if k in cfg}
        if method not in protocol['methods'].values() or cfg.get('mask_seed')!=cfg['seed']:raise ValueError('Unregistered diagnostic method or seed')
        if cfg.get('allocation')!={'layer_clip':'equal_active'} or protocol.get('data_scope')!='public_benchmark':raise ValueError('Diagnostic scope requires public benchmark and equal clipping')
        if cfg.get('diagnostic_rng_path'):
            secret=Path(cfg['diagnostic_rng_path'])
            if hashlib.sha256(secret.read_bytes()).hexdigest()!=protocol['d1_rng_sha256']:raise ValueError('D1 random state changed')
    if joint_kind:
        fields={'mode','freeze_rounds','feedback','diagnostic_freezing','diagnostic_rng_path','static_order','allocation'}
        method={k:cfg[k] for k in fields if k in cfg}
        registered=protocol['methods_by_seed'].get(str(cfg['seed']),{})
        if method not in registered.values() or cfg.get('mask_seed')!=cfg['seed']:raise ValueError('Unregistered joint method or seed')
        if protocol.get('data_scope')!='public_benchmark':raise ValueError('Joint experiment requires public benchmark scope')
        secret=Path(cfg['diagnostic_rng_path'])
        expected=protocol['private_rng_sha256'].get(str(secret))
        if expected is None or hashlib.sha256(secret.read_bytes()).hexdigest()!=expected:raise ValueError('Joint random state changed')
    if cfg['mode']=='RAND' and cfg['mask_seed'] not in protocol['random_mask_seeds']:
        raise ValueError('Unregistered random mask seed')
    if kind in {'controlled_allocation_comparison_v1','controlled_bu_allocation_comparison_v1'}:
        if kind=='controlled_allocation_comparison_v1' and (cfg['mode']!='NF' or cfg.get('freeze_rounds') or 'feedback' in cfg):
            raise ValueError('Allocation comparison forbids freezing/feedback')
        if kind=='controlled_bu_allocation_comparison_v1' and (cfg['mode']!='BU' or not cfg.get('freeze_rounds') or 'feedback' in cfg):
            raise ValueError('BU allocation comparison requires fixed BU freezing without feedback')
        if cfg.get('allocation') not in protocol['allocations'].values() or cfg.get('mask_seed')!=cfg['seed']:
            raise ValueError('Unregistered allocation or mask seed')
    for name,digest in protocol['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Source changed since protocol lock: {name}')
    archive=path.parent/protocol['privacy_preflight']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=protocol['privacy_preflight_sha256']:
        raise ValueError('Preflight ledger archive changed')
    return np.load(archive)

def check_round(archive,t,clients,targets,q,sigmas,radii):
    np.testing.assert_array_equal(np.asarray(clients[0].alphas),archive['orders'])
    for i,client in enumerate(clients):
        group=int(np.flatnonzero(archive['targets']==targets[i])[0])
        np.testing.assert_allclose(client.rdp_eps_spent,archive['rdp'][t-1,group],rtol=0,atol=1e-12)
        np.testing.assert_allclose([q[i],sigmas[i],radii[i]],archive['params'][t-1,group],rtol=0,atol=1e-12)
