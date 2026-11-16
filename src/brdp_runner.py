"""Client-level trusted-server DP-FL with directional soft recycling.

Run from BR_DP. This entry point does not modify/use the trynew checkout.
No privacy amplification, raw telemetry, public private-RNG seeds, or grid
diagnostics are used to certify releases.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import secrets
import shutil

import numpy as np
import scipy
import torch
import yaml

from .brdp_budget import DirectionalBudget
from .brdp_config import validate_config
from .brdp_direction import DirectionState
from .brdp_filter import FixedOrderRDPFilter
from .brdp_mechanism import release_directional
from .brdp_noise import moments
from .clients import local_update
from .layer_registry import LayerRegistry
from .metrics import evaluate
from .models import build_model
from .st_mechanism import clip_blocks
from .st_runner import _datasets, round_inputs, write_csv


def make_components(cfg):
    model = build_model(cfg)
    if list(model.named_buffers()):
        raise ValueError('Private buffers unsupported')
    registry = LayerRegistry(model)
    ledger = FixedOrderRDPFilter(cfg['privacy']['epsilon'], cfg['privacy']['delta'],
                               cfg['privacy']['adjacency'], cfg['mechanism']['accounting']['alpha'])
    direction_cfg = cfg['mechanism']['direction']
    directions = DirectionState(registry.dimensions, direction_cfg['ema_decay'], direction_cfg['norm_floor'])
    allocator = DirectionalBudget(registry.dimensions, cfg['train']['rounds'], cfg['budget'],
                                  cfg['mechanism'], ledger.alpha, ledger.adjacency_factor)
    return model, registry, ledger, directions, allocator


def plan_rows(plan, cfg):
    rows = []
    for p in plan.layers:
        c = p.certificate
        row = dict(round=plan.round_index, layer=p.layer, active=int(p.active),
                   dimension=p.dimension, clip_radius=p.clip_radius, denominator=p.denominator,
                   direction_source_round=p.direction_source_round, direction_hash=p.direction_hash,
                   selection_reason=p.selection_reason, sample_rate=cfg['privacy']['sample_rate'])
        row.update(asdict(c) if c is not None else dict(alpha=0., sensitivity=0.,
            dimension=p.dimension, rdp_budget=0., sigma_parallel=0., sigma_orth=0.,
            recycle_prob=0., tau=0., rdp_upper=0., bound_mode='frozen', status='FROZEN'))
        row['certificate_id'] = c.certificate_id if c is not None else 'none'
        row['expected_noise_energy'] = c.expected_noise_energy if c is not None else 0.
        row['parallel_variance'] = moments(c.sigma_parallel, c.tau, c.recycle_prob)['variance'] if c is not None else 0.
        rows.append(row)
    return rows


def _private_rngs(device):
    return (np.random.default_rng(secrets.randbits(128)),
            np.random.default_rng(secrets.randbits(128)),
            torch.Generator(device=device).manual_seed(secrets.randbits(63)))


def run(config, output):
    cfg = validate_config(config)
    torch.set_num_threads(1)
    torch.manual_seed(cfg['experiment']['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, registry, ledger, direction_state, allocator = make_components(cfg)
    device = cfg['train']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    config_path = out/'config.yaml'
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    manifest = {}
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(source, snapshot/source.name)
        manifest[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (out/'implementation.json').write_text(json.dumps(dict(
        source_sha256=manifest, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        torch=torch.__version__, scipy=scipy.__version__,
        trust_model='trusted_server_aggregate_only', privacy_unit='one fixed client slot',
        sampling_amplification=False, accountant=cfg['mechanism']['accounting']['bound_mode'],
        noise_rng='independent private OS seeds; research NumPy/Torch floating-point RNGs',
        guarantee_scope='ideal real-valued mechanism; not a hardened finite-precision implementation'), indent=2), encoding='utf-8')
    train_data, test_data = _datasets(cfg)
    n = cfg['data']['clients']
    if len(train_data) < n:
        raise ValueError('At least one training sample per fixed slot required')
    parts = [list(range(i, len(train_data), n)) for i in range(n)]
    sample_rng, scalar_rng, orthogonal_rng = _private_rngs(device)
    rounds, layers, released_history = [], [], {}
    for t in range(1, cfg['train']['rounds']+1):
        directions = direction_state.for_round(t)
        active, radii, denominator = round_inputs(cfg, registry.names, cfg['clipping']['weights'],
                                                  cfg['freezing']['freeze_rounds'], t)
        plan = allocator.plan(t, active, ledger.remaining, radii, denominator, directions)
        ledger.accept(plan)
        # Persist accepted cost before current private access; no refund/retry.
        write_csv(out/'privacy.csv', ledger.history)
        layers.extend(plan_rows(plan, cfg))
        write_csv(out/'layer_mechanisms.csv', layers)
        private_sum = {g: torch.zeros_like(v) for g, v in registry.pack(model).items()}
        selected = np.flatnonzero(sample_rng.random(n) < cfg['privacy']['sample_rate'])
        for i in selected:
            update = local_update(model, registry, train_data, parts[int(i)], active, cfg['train'],
                                  cfg['experiment']['seed']+t*n+int(i), device, cfg['train']['lr'])
            clipped = clip_blocks(update, active, plan.clip_radius)
            for g in registry.names:
                private_sum[g].add_(clipped[g])
            del update, clipped
        # Noise and cost are present even with no selected clients.
        released = release_directional(private_sum, plan, directions, scalar_rng, orthogonal_rng)
        del private_sum, selected
        registry.apply(model, released)
        direction_state.observe(t, released)
        allocator.observe(t, released, plan.expected_noise_energy)
        released_history[str(t)] = {g: x.detach().cpu().clone() for g, x in released.items()}
        loss, accuracy = evaluate(model, test_data, device)
        rounds.append(dict(round=t, public_test_loss=loss, public_test_accuracy=accuracy,
            rdp_round=ledger.history[-1]['rdp_increment'], rdp_charged_total=ledger.spent,
            epsilon_guarantee=ledger.target, epsilon_path_conversion=ledger.history[-1]['epsilon_path_conversion'],
            expected_noise_energy=sum(plan.expected_noise_energy.values())))
        write_csv(out/'rounds.csv', rounds)
        print(f'BRDP round={t} public_accuracy={accuracy:.2f}% rdp_charged={ledger.spent:.6f}', flush=True)
    torch.save(model.state_dict(), out/'model.pt')
    torch.save(released_history, out/'released_updates.pt')
    torch.save(direction_state.state_dict(), out/'directions_state.pt')
    (out/'allocator_state.json').write_text(json.dumps(allocator.state_dict(), indent=2), encoding='utf-8')
    non_gaussian = sum(row['active'] and row['recycle_prob'] > 0 for row in layers)
    summary = dict(status='completed', mechanism='directional_soft_recycling', version=1,
        scope='synthetic engineering smoke' if cfg['data']['dataset'] == 'synthetic' else 'public dataset research experiment',
        trust_model='trusted_server_aggregate_only', privacy_unit='one fixed client slot',
        adjacency=ledger.adjacency, partition='fixed public index modulo client count',
        sampling_amplification=False, alpha=ledger.alpha, epsilon_target=ledger.target,
        epsilon_guarantee=ledger.target, epsilon_path_conversion=ledger.history[-1]['epsilon_path_conversion'],
        delta=ledger.delta, rdp_filter_cap=ledger.cap, rdp_charged_total=ledger.spent,
        rdp_remaining=ledger.remaining, rounds=len(rounds),
        accountant=cfg['mechanism']['accounting']['bound_mode'],
        selection_policy=cfg['mechanism']['selection']['policy'],
        non_gaussian_layer_releases=non_gaussian,
        gaussian_layer_releases=sum(row['active'] and row['recycle_prob'] == 0 for row in layers),
        final_test_accuracy=rounds[-1]['public_test_accuracy'],
        best_test_accuracy=max(row['public_test_accuracy'] for row in rounds),
        runtime_privacy_scope='research floating-point implementation, not cryptographic certification')
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out/'certificate_metadata.json').write_text(json.dumps(dict(
        bound_mode=cfg['mechanism']['accounting']['bound_mode'],
        numerical_margin='1e-12 relative plus 1e-15 absolute, rounded upward',
        interval_arithmetic=False, grid_diagnostics_used_for_charging=False,
        proof_assumptions=['predictable direction/parameters', 'bounded client blocks',
                           'public fixed denominator', 'independent fresh noise',
                           'ideal Gaussian and soft-mixture distributions']), indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.smoke:
        cfg['data'].update(dataset='synthetic', clients=4, limit_train=32, limit_test=16)
        cfg['model']['name'] = 'prototype_cnn'
        cfg['train'].update(rounds=4, batch_size=8, local_epochs_or_iterations=1, device='cpu')
        cfg['budget']['time_weights'] = [1., 1., 2., 3.]
        cfg['freezing']['freeze_rounds'] = dict(conv1=3, conv2=4)
        cfg['mechanism'].setdefault('direction', {})['warmup_rounds'] = 1
    print(json.dumps(run(cfg, args.output), indent=2))


if __name__ == '__main__':
    main()
