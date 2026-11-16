"""Author spend-as-you-go scheduling in the existing ST training/accounting setup.

This is an explicitly adapted prior-method baseline, not original Table 2
reproduction. Author source is read unchanged; privacy schedule shape is scaled
once, publicly, to match the existing conservative no-amplification rho budget.
"""
import ast
import copy
import csv
import functools
import hashlib
import json
import math
from pathlib import Path
import secrets
import types

import numpy as np
from opacus.accountants.analysis import rdp
from opacus.accountants.utils import get_noise_multiplier
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from .clients import local_update
from .metrics import evaluate
from .st_accountant import GaussianRDPFilter, gaussian_rho
from .st_audit import close
from .st_budget import ReleasePlan
from .st_mechanism import clip_blocks, release_sum
from .st_runner import _datasets, layer_rows, make_components, round_inputs, validate_config, write_csv


AUTHOR_ROOT = Path(__file__).resolve().parents[2]/'fl-experiments-code-submission'
AUTHOR_CLIENT = 'src/clients/client_scheme2_time_adaptive_dpfl.py'


@functools.lru_cache(maxsize=1024)
def _noise(items):
    return get_noise_multiplier(**dict(items))


@functools.lru_cache(maxsize=1024)
def _rdp(q, sigma, orders):
    return rdp.compute_rdp(q=q, noise_multiplier=sigma, steps=1., orders=orders)


def author_client():
    """Execute the supplied Client class, with no import/module side effects.

    Its privacy methods run verbatim. A dummy one-weight model is sufficient
    because this object is used ONLY for public scheduling, never for training.
    """
    path = AUTHOR_ROOT/AUTHOR_CLIENT
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Client']
    if len(tree.body) != 1:
        raise ValueError('Expected the supplied time-adaptive Client class')
    analysis = types.SimpleNamespace(get_privacy_spent=rdp.get_privacy_spent,
        compute_rdp=lambda **kw: _rdp(float(kw['q']), float(kw['noise_multiplier']), tuple(kw['orders'])))
    namespace = dict(np=np, copy=copy, torch=torch, nn=nn, optim=optim,
                     CosineAnnealingLR=CosineAnnealingLR, rdp=analysis,
                     get_model=lambda *args: nn.Linear(1, 1),
                     get_noise_multiplier=lambda **kw: _noise(tuple(sorted(kw.items()))))
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['Client']


def source_schedule(common_config, q_save=.5, transition_round=13):
    """Raw source recurrence + a single data-independent common-accountant scale."""
    cfg = validate_config(common_config)
    q_spend = cfg['privacy']['sample_rate']
    rounds = cfg['train']['rounds']
    if not 0 < q_save <= q_spend <= 1:
        raise ValueError('Require 0 < q_save <= q_spend <= 1')
    if type(transition_round) is not int or not 1 <= transition_round <= rounds:
        raise ValueError('Invalid saving-to-spending transition')
    cls = author_client()
    client = cls(0, None, None, cfg['train']['local_epochs_or_iterations'],
        cfg['privacy']['epsilon'], cfg['privacy']['delta'], rounds, cfg['data']['clients'],
        cfg['train']['lr'], cfg['train']['batch_size'], 1., 'fmnist', 1000000,
        ['save' if t < transition_round else 'spend' for t in range(1, rounds+1)], q_save, 'cpu')
    rows = []
    for t in range(1, rounds+1):
        sigma = float(client.set_local_sigma_and_sample_rate(q_spend, t-1))
        q = q_save if t < transition_round else q_spend
        epsilon, _, _, _ = client.rdp_accountant(sample_rate=q)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError('Source returned invalid sigma')
        rows.append(dict(round=t, q=q, source_sigma=sigma,
                         source_epsilon_reported=float(epsilon),
                         raw_no_amplification_rho=1/(2*sigma*sigma)))
    ledger = GaussianRDPFilter(cfg['privacy']['epsilon'], cfg['privacy']['delta'], cfg['privacy']['adjacency'])
    raw_total = math.fsum(row['raw_no_amplification_rho'] for row in rows)
    # One public scale preserves the relative spend-as-you-go profile and uses
    # exactly the same conservative budget as the already completed ST runs.
    scale = math.sqrt(raw_total/(ledger.rho_cap*(1-1e-10)))
    for row in rows:
        row['common_accountant_sigma'] = row['source_sigma']*scale
        row['rho_planned'] = 1/(2*row['common_accountant_sigma']**2)
    return dict(rows=rows, sigma_scale=scale, rho_cap=ledger.rho_cap,
                q_save=q_save, transition_round=transition_round,
                source_file=AUTHOR_CLIENT,
                source_sha256=hashlib.sha256((AUTHOR_ROOT/AUTHOR_CLIENT).read_bytes()).hexdigest(),
                provenance='author Client.set_local_sigma_and_sample_rate and rdp_accountant executed verbatim',
                adaptation='homogeneous epsilon; source sigma shape scaled to common no-amplification accountant')


def release_plan(cfg, registry, clipping, events, schedule_row, ledger):
    t = schedule_row['round']
    active, radii, _ = round_inputs(cfg, registry.names, clipping, events, t)
    denominator = cfg['data']['clients']*schedule_row['q']
    # All active coordinates have the same noise, as in the prior method.
    std = (ledger.adjacency_factor*cfg['privacy']['clip_norm']/denominator
           *schedule_row['common_accountant_sigma'])
    sensitivity = {g: ledger.adjacency_factor*radii[g]/denominator if active[g] else 0.
                   for g in registry.names}
    noise = {g: std if active[g] else 0. for g in registry.names}
    rho = {g: .5*(sensitivity[g]/std)**2 if active[g] else 0. for g in registry.names}
    plan = ReleasePlan(t, rho, {g: radii[g] if active[g] else 0. for g in registry.names}, sensitivity, noise)
    return active, denominator, plan


def run_baseline(common_config, schedule, output, variant):
    cfg = validate_config(common_config)
    if variant not in {'TAP_NF', 'TAP_SAME_FREEZE'}:
        raise ValueError('Unknown baseline variant')
    if variant == 'TAP_NF':
        cfg['freezing']['freeze_rounds'] = {}
    torch.set_num_threads(1)
    # Source scheduling may have consumed Torch state: reset BEFORE model init.
    torch.manual_seed(cfg['experiment']['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, registry, _, clipping, events, ledger = make_components(cfg)
    device = cfg['train']['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    train_data, test_data = _datasets(cfg)
    n = cfg['data']['clients']
    if len(train_data) < n or len(schedule['rows']) != cfg['train']['rounds']:
        raise ValueError('Data size or schedule horizon mismatch')
    parts = [list(range(i, len(train_data), n)) for i in range(n)]
    sampling_rng = np.random.default_rng(secrets.randbits(128))
    noise_rng = torch.Generator(device=device).manual_seed(secrets.randbits(63))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    import yaml
    (out/'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    (out/'source_schedule.json').write_text(json.dumps(schedule, indent=2), encoding='utf-8')
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    (out/'implementation.json').write_text(json.dumps(dict(source_sha256=manifest,
        author_client_sha256=schedule['source_sha256'], variant=variant,
        scope='adapted prior baseline, not original-paper table reproduction'), indent=2), encoding='utf-8')
    rounds, layers = [], []
    for source_row in schedule['rows']:
        t = source_row['round']
        active, denominator, plan = release_plan(cfg, registry, clipping, events, source_row, ledger)
        ledger.record(t, plan.sensitivity, plan.noise_std)
        private_sum = {g: torch.zeros_like(v) for g, v in registry.pack(model).items()}
        selected = np.flatnonzero(sampling_rng.random(n) < source_row['q'])
        for i in selected:
            update = local_update(model, registry, train_data, parts[int(i)], active, cfg['train'],
                cfg['experiment']['seed']+t*n+int(i), device, cfg['train']['lr'])
            clipped = clip_blocks(update, active, plan.clip_radius)
            for g in registry.names:
                private_sum[g].add_(clipped[g])
            del update, clipped
        released = release_sum(private_sum, active, plan, denominator, noise_rng)
        del private_sum, selected
        registry.apply(model, released)
        loss, accuracy = evaluate(model, test_data, device)
        if not math.isfinite(loss):
            raise RuntimeError('Non-finite public evaluation; record failure, do not substitute a run')
        rounds.append(dict(round=t, public_test_loss=loss, public_test_accuracy=accuracy,
            sample_rate=source_row['q'], rho_round=gaussian_rho(plan.sensitivity, plan.noise_std),
            epsilon_bound=ledger.epsilon_bound, epsilon_accounting_value=ledger.epsilon_accounting_value,
            expected_noise_energy=sum(registry.dimensions[g]*plan.noise_std[g]**2 for g in registry.names)))
        log_cfg = copy.deepcopy(cfg)
        log_cfg['privacy']['sample_rate'] = source_row['q']
        layers.extend(layer_rows(t, registry, active, plan, denominator, log_cfg))
        write_csv(out/'rounds.csv', rounds)
        write_csv(out/'layers.csv', layers)
        write_csv(out/'privacy.csv', ledger.history)
        print(f'{variant} seed{cfg["experiment"]["seed"]} round={t} accuracy={accuracy:.2f}% rho={ledger.spent:.9f}', flush=True)
    torch.save(model.state_dict(), out/'model.pt')
    np.savez(out/'accountant.npz', alpha=ledger.alpha, rho_cap=ledger.rho_cap, rho_spent=ledger.spent,
        rdp_spent=ledger.alpha*ledger.spent, rdp_filter_cap=ledger.alpha*ledger.rho_cap,
        epsilon_target=ledger.target, delta=ledger.delta)
    summary = dict(status='completed', method=variant, seed=cfg['experiment']['seed'],
        best_test_accuracy=max(row['public_test_accuracy'] for row in rounds),
        final_test_accuracy=rounds[-1]['public_test_accuracy'], rounds=len(rounds),
        epsilon_target=ledger.target, epsilon_bound=ledger.epsilon_bound,
        epsilon_accounting_value=ledger.epsilon_accounting_value, delta=ledger.delta,
        rho_cap=ledger.rho_cap, rho_spent=ledger.spent, alpha=ledger.alpha,
        sampling_amplification=False, expected_client_rounds=n*sum(r['q'] for r in schedule['rows']),
        scope='source-schedule adapted TAP baseline; not original paper Table 2')
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def audit_baseline(directory):
    """Reconstruct geometry, actual q and source-profile noise independently."""
    import yaml
    root = Path(directory)
    cfg = validate_config(yaml.safe_load((root/'config.yaml').read_text(encoding='utf-8')))
    schedule = json.loads((root/'source_schedule.json').read_text(encoding='utf-8'))
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    torch.set_num_threads(1)
    _, registry, _, clipping, events, ledger = make_components(cfg)
    if hashlib.sha256((AUTHOR_ROOT/AUTHOR_CLIENT).read_bytes()).hexdigest() != schedule['source_sha256']:
        raise ValueError('Author source changed')
    raw_total = math.fsum(.5/row['source_sigma']**2 for row in schedule['rows'])
    expected_scale = math.sqrt(raw_total/(ledger.rho_cap*(1-1e-10)))
    close(schedule['sigma_scale'], expected_scale, 'public sigma scale')
    def read(name):
        with (root/name).open(encoding='utf-8', newline='') as handle:
            return list(csv.DictReader(handle))
    layers, privacy, rounds = read('layers.csv'), read('privacy.csv'), read('rounds.csv')
    T = cfg['train']['rounds']
    if len(layers) != T*len(registry.names) or len(privacy) != T or len(rounds) != T or len(schedule['rows']) != T:
        raise ValueError('Incomplete baseline logs')
    for source_row in schedule['rows']:
        t = source_row['round']
        expected_q = schedule['q_save'] if t < schedule['transition_round'] else cfg['privacy']['sample_rate']
        close(source_row['q'], expected_q, 'scheduled sample rate')
        close(source_row['common_accountant_sigma'], source_row['source_sigma']*expected_scale, 'scaled sigma')
        active, denominator, plan = release_plan(cfg, registry, clipping, events, source_row, ledger)
        block_rows = [row for row in layers if int(row['round']) == t]
        if len(block_rows) != len(registry.names) or {r['layer'] for r in block_rows} != set(registry.names):
            raise ValueError('Duplicate or missing layers')
        sensitivity, noise = {}, {}
        for row in block_rows:
            g = row['layer']
            expected = dict(active=int(active[g]), dimensions=registry.dimensions[g],
                denominator=denominator, sample_rate=source_row['q'], clip_radius=plan.clip_radius[g],
                sensitivity=plan.sensitivity[g], noise_std=plan.noise_std[g],
                noise_variance=plan.noise_std[g]**2, rho_allocated=plan.rho[g],
                expected_noise_energy=registry.dimensions[g]*plan.noise_std[g]**2)
            for key, value in expected.items():
                close(float(row[key]), value, f'{g}.{key}')
            sensitivity[g], noise[g] = float(row['sensitivity']), float(row['noise_std'])
        recorded = ledger.record(t, sensitivity, noise)
        for key, value in recorded.items():
            close(float(privacy[t-1][key]), value, f'privacy.{key}')
        close(float(rounds[t-1]['rho_round']), recorded['rho_increment'], 'round rho')
    for key, value in dict(rho_spent=ledger.spent, rho_cap=ledger.rho_cap, alpha=ledger.alpha,
        epsilon_bound=ledger.epsilon_bound, epsilon_accounting_value=ledger.epsilon_accounting_value,
        epsilon_target=ledger.target, delta=ledger.delta).items():
        close(summary[key], value, key)
    close(summary['best_test_accuracy'], max(float(row['public_test_accuracy']) for row in rounds), 'best accuracy')
    close(summary['final_test_accuracy'], float(rounds[-1]['public_test_accuracy']), 'final accuracy')
    with np.load(root/'accountant.npz') as account:
        for key, value in dict(alpha=ledger.alpha, rho_cap=ledger.rho_cap, rho_spent=ledger.spent,
            rdp_spent=ledger.alpha*ledger.spent, rdp_filter_cap=ledger.alpha*ledger.rho_cap,
            epsilon_target=ledger.target, delta=ledger.delta).items():
            close(float(account[key]), value, key)
    return dict(status='passed', rounds=T, epsilon_bound=ledger.epsilon_bound,
                rho_spent=ledger.spent, scope='actual-noise ledger consistency audit')
