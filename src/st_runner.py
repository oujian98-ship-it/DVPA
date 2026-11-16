"""Independent time x layer budget experiment; never calls the old scheduler."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import secrets

import numpy as np
import torch
import yaml
from torch.utils.data import TensorDataset

from .clients import local_update
from .data import load_data
from .layer_registry import LayerRegistry
from .metrics import evaluate
from .models import build_model
from .st_accountant import GaussianRDPFilter, gaussian_rho, positive
from .st_budget import TimeLayerBudget
from .st_mechanism import clip_blocks, release_sum


def write_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def validate_config(config):
    cfg = copy.deepcopy(config)
    required = {'experiment', 'model', 'data', 'train', 'privacy', 'budget', 'clipping', 'freezing'}
    if set(cfg) != required:
        raise ValueError(f'Expected exactly these config sections: {sorted(required)}')
    allowed = {
        'experiment': {'seed'}, 'model': {'name'},
        'data': {'dataset', 'root', 'clients', 'limit_train', 'limit_test', 'public_test'},
        'train': {'rounds', 'batch_size', 'local_epochs_or_iterations', 'local_semantics',
                  'lr', 'momentum', 'device', 'cpu_threads'},
        'privacy': {'epsilon', 'delta', 'adjacency', 'sample_rate', 'clip_norm'},
        'clipping': {'weights'}, 'freezing': {'freeze_rounds'},
    }
    for section, keys in allowed.items():
        if set(cfg[section]) - keys:
            raise ValueError(f'Unknown fields in {section}')
    if cfg['model']['name'] not in {'prototype_cnn', 'fedavg_cnn'}:
        raise ValueError('New runner supports buffer-free grayscale CNNs only')
    if cfg['data']['dataset'] not in {'fmnist', 'mnist', 'synthetic'}:
        raise ValueError('New runner supports fmnist, mnist, synthetic')
    if cfg['data'].get('public_test') is not True:
        raise ValueError('Evaluation requires an explicitly public test set')
    for key in ('limit_train', 'limit_test'):
        if key in cfg['data'] and (type(cfg['data'][key]) is not int or cfg['data'][key] < 1):
            raise ValueError(f'data.{key} must be a positive integer')
    for section, key in [('data', 'clients'), ('train', 'rounds'), ('train', 'batch_size'),
                         ('train', 'local_epochs_or_iterations'), ('train', 'cpu_threads')]:
        value = cfg[section][key]
        if type(value) is not int or value < 1:
            raise ValueError(f'{section}.{key} must be a positive integer')
    if cfg['train']['cpu_threads'] != 1:
        raise ValueError('This Windows runtime requires cpu_threads=1')
    if type(cfg['experiment']['seed']) is not int or cfg['experiment']['seed'] < 0:
        raise ValueError('Public experiment seed must be nonnegative integer')
    positive(cfg['train']['lr'], 'learning rate')
    momentum = cfg['train']['momentum']
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError('momentum must be in [0,1)')
    if cfg['train']['local_semantics'] not in {'epochs', 'steps'}:
        raise ValueError('Unsupported local training semantics')
    q = positive(cfg['privacy']['sample_rate'], 'sample rate')
    if q > 1:
        raise ValueError('sample_rate cannot exceed one')
    positive(cfg['privacy']['clip_norm'], 'clip norm')
    GaussianRDPFilter(cfg['privacy']['epsilon'], cfg['privacy']['delta'], cfg['privacy']['adjacency'])
    return cfg


def make_components(cfg):
    model = build_model(cfg)
    if list(model.named_buffers()):
        raise ValueError('Private model buffers are not supported')
    registry = LayerRegistry(model)
    allocator = TimeLayerBudget(registry.names, cfg['train']['rounds'], cfg['budget'])
    clipping = cfg['clipping']['weights']
    if set(clipping) != set(registry.names):
        raise ValueError('Clipping weights must name every model layer')
    clipping = {g: positive(v, 'clipping weight') for g, v in clipping.items()}
    events = cfg['freezing']['freeze_rounds']
    if set(events) - set(registry.names):
        raise ValueError('Unknown frozen layer')
    if any(type(t) is not int or not 1 <= t <= cfg['train']['rounds'] for t in events.values()):
        raise ValueError('Freeze rounds must lie within training horizon')
    if len(events) >= len(registry.names):
        raise ValueError('Freezing must leave at least one active layer')
    ledger = GaussianRDPFilter(cfg['privacy']['epsilon'], cfg['privacy']['delta'], cfg['privacy']['adjacency'])
    return model, registry, allocator, clipping, events, ledger


def round_inputs(cfg, names, clipping, events, t):
    active = {g: g not in events or t < events[g] for g in names}
    total = math.fsum(clipping[g] for g in names if active[g])
    # This is a clipping allocation, independent of PRIVACY budget allocation.
    radii = {g: cfg['privacy']['clip_norm'] * math.sqrt(clipping[g] / total) for g in names}
    denominator = cfg['data']['clients'] * cfg['privacy']['sample_rate']
    return active, radii, denominator


def layer_rows(t, registry, active, plan, denominator, cfg):
    total = math.fsum(plan.rho.values())
    return [dict(round=t, layer=g, active=int(active[g]), dimensions=registry.dimensions[g],
                 rho_allocated=plan.rho[g], budget_fraction=plan.rho[g]/total,
                 clip_radius=plan.clip_radius[g], sensitivity=plan.sensitivity[g],
                 noise_std=plan.noise_std[g], noise_variance=plan.noise_std[g]**2,
                 expected_noise_energy=registry.dimensions[g]*plan.noise_std[g]**2,
                 denominator=denominator, sample_rate=cfg['privacy']['sample_rate'])
            for g in registry.names]


def preview(config):
    cfg = validate_config(config)
    torch.set_num_threads(1)
    torch.manual_seed(cfg['experiment']['seed'])
    _, registry, _, clipping, events, ledger = make_components(cfg)
    # No invented observations: preview explicitly disables online feedback.
    allocator = TimeLayerBudget(registry.names, cfg['train']['rounds'], {**cfg['budget'], 'feedback': False})
    rows = []
    for t in range(1, cfg['train']['rounds']+1):
        active, radii, denominator = round_inputs(cfg, registry.names, clipping, events, t)
        plan = allocator.plan(t, active, ledger.remaining, radii, denominator, ledger.adjacency_factor)
        ledger.record(t, plan.sensitivity, plan.noise_std)
        rows.extend(layer_rows(t, registry, active, plan, denominator, cfg))
    return dict(scope='public prior preview; no data/training/release; online feedback disabled',
                alpha=ledger.alpha, rho_cap=ledger.rho_cap, epsilon_bound=ledger.epsilon_bound,
                epsilon_accounting_value=ledger.epsilon_accounting_value,
                layers=rows, privacy=ledger.history)


def _datasets(cfg):
    if cfg['data']['dataset'] != 'synthetic':
        return load_data(cfg['data'])
    generator = torch.Generator().manual_seed(cfg['experiment']['seed'])
    datasets = []
    for count in (cfg['data'].get('limit_train', 96), cfg['data'].get('limit_test', 32)):
        datasets.append(TensorDataset(torch.randn(count, 1, 28, 28, generator=generator),
                                     torch.randint(0, 10, (count,), generator=generator)))
    return datasets


def run(config, output):
    cfg = validate_config(config)
    torch.set_num_threads(1)
    torch.manual_seed(cfg['experiment']['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, registry, allocator, clipping, events, ledger = make_components(cfg)
    device = cfg['train']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    train_data, test_data = _datasets(cfg)
    n = cfg['data']['clients']
    if len(train_data) < n:
        raise ValueError('Require at least one training record per fixed client slot')
    # Fixed public index rule, never label-dependent repartition or donor repair.
    # A neighboring client changes only its assigned slot's local dataset.
    parts = [list(range(i, len(train_data), n)) for i in range(n)]
    sample_rng = np.random.default_rng(secrets.randbits(128))
    noise_rng = torch.Generator(device=device).manual_seed(secrets.randbits(63))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob('*.py')}
    (out/'implementation.json').write_text(json.dumps(dict(source_sha256=manifest,
        torch=torch.__version__, trust_model='trusted_server_aggregate_only',
        sampling_amplification=False, noise_rng='private OS seed; torch research RNG'), indent=2), encoding='utf-8')
    rows, layers = [], []
    for t in range(1, cfg['train']['rounds']+1):
        active, radii, denominator = round_inputs(cfg, registry.names, clipping, events, t)
        plan = allocator.plan(t, active, ledger.remaining, radii, denominator, ledger.adjacency_factor)
        # Debit the ACTUAL joint Gaussian mechanism before any private query.
        ledger.record(t, plan.sensitivity, plan.noise_std)
        private_sum = {g: torch.zeros_like(v) for g, v in registry.pack(model).items()}
        selected = np.flatnonzero(sample_rng.random(n) < cfg['privacy']['sample_rate'])
        for i in selected:
            update = local_update(model, registry, train_data, parts[int(i)], active, cfg['train'],
                                  cfg['experiment']['seed'] + t*n + int(i), device, cfg['train']['lr'])
            clipped = clip_blocks(update, active, plan.clip_radius)
            for g in registry.names:
                private_sum[g].add_(clipped[g])
            del update, clipped
        # Always add noise, including rounds with zero selected clients.
        released = release_sum(private_sum, active, plan, denominator, noise_rng)
        del private_sum, selected
        allocator.observe(t, released, {g: plan.noise_std[g]**2 for g in registry.names})
        registry.apply(model, released)
        loss, accuracy = evaluate(model, test_data, device)
        rows.append(dict(round=t, public_test_loss=loss, public_test_accuracy=accuracy,
                         rho_round=gaussian_rho(plan.sensitivity, plan.noise_std),
                         epsilon_bound=ledger.epsilon_bound,
                         epsilon_accounting_value=ledger.epsilon_accounting_value,
                         expected_noise_energy=sum(registry.dimensions[g]*plan.noise_std[g]**2 for g in registry.names)))
        layers.extend(layer_rows(t, registry, active, plan, denominator, cfg))
        write_csv(out/'rounds.csv', rows)
        write_csv(out/'layers.csv', layers)
        write_csv(out/'privacy.csv', ledger.history)
        print(f'ST round={t} public_accuracy={accuracy:.2f}% epsilon_bound={ledger.epsilon_bound:.6f}', flush=True)
    torch.save(model.state_dict(), out/'model.pt')
    np.savez(out/'accountant.npz', alpha=ledger.alpha, rho_cap=ledger.rho_cap,
             rho_spent=ledger.spent, rdp_spent=ledger.alpha*ledger.spent,
             rdp_filter_cap=ledger.alpha*ledger.rho_cap,
             epsilon_target=ledger.target, delta=ledger.delta)
    summary = dict(status='completed', mechanism='time_layer_budget_gaussian',
                   scope='synthetic engineering smoke' if cfg['data']['dataset']=='synthetic' else 'public dataset research experiment',
                   epsilon_target=ledger.target, epsilon_bound=ledger.epsilon_bound,
                   epsilon_accounting_value=ledger.epsilon_accounting_value,
                   delta=ledger.delta, alpha=ledger.alpha, rho_cap=ledger.rho_cap,
                   rho_spent=ledger.spent, sampling_amplification=False,
                   privacy_unit='one fixed client slot', trust_model='trusted_server_aggregate_only',
                   adjacency=ledger.adjacency, partition='fixed public index modulo client count',
                   final_test_accuracy=rows[-1]['public_test_accuracy'],
                   best_test_accuracy=max(row['public_test_accuracy'] for row in rows),
                   rounds=len(rows), personalized_budgets=False)
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Three rounds of public synthetic CPU data')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.smoke:
        cfg['data'].update(dataset='synthetic', clients=4, limit_train=32, limit_test=16)
        cfg['model']['name'] = 'prototype_cnn'
        cfg['train'].update(rounds=3, batch_size=8, local_epochs_or_iterations=1, device='cpu')
        cfg['budget']['time_weights'] = [1., 2., 3.]
        cfg['freezing']['freeze_rounds'] = {'conv1': 2, 'conv2': 3}
    if args.preview:
        result = preview(cfg)
        if args.output:
            with Path(args.output).open('x', encoding='utf-8') as handle:
                json.dump(result, handle, indent=2)
        print(json.dumps({k: v for k, v in result.items() if k not in {'layers', 'privacy'}}, indent=2))
    else:
        if not args.output:
            parser.error('training requires --output <new-directory>')
        print(json.dumps(run(cfg, args.output), indent=2))


if __name__ == '__main__':
    main()
