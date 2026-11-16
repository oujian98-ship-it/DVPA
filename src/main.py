import argparse
import csv
import importlib.metadata
import hashlib
import json
import math
from pathlib import Path
import random
import secrets
import subprocess
import time
import numpy as np
import torch
import yaml
from .accountant import Ledger, ORDERS
from .original_scheduler import PrivacyScheduler
from .data import load_data, dirichlet_split, split_checksum
from .models import build_model
from .layer_registry import LayerRegistry
from .layer_clip import clip, validate
from .prior_scheduler import weights
from .static_freezer import StaticFreezer
from .dp_feedback_freezer import FeedbackFreezer
from .clients import local_update
from .server import add_client_noise, aggregate
from .metrics import evaluate

def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

def run(config, output=None, resume=False):
    cfg = config
    seed = cfg['experiment']['seed']
    if cfg['method'].get('layer_freeze')=='dp_feedback' and cfg['method'].get('temporal')=='fedavg':
        raise ValueError('Feedback freezing requires DP aggregate releases')
    if (cfg['method'].get('layer_freeze', 'none') != 'none' or cfg['method'].get('layer_clip', 'original') != 'original') and not cfg['experiment'].get('engineering_only', False):
        gate = Path('reports/p0_gate.json')
        if not gate.exists() or json.loads(gate.read_text()).get('status') != 'passed':
            raise RuntimeError('P0 gate not passed. Later-stage configurations require --smoke for engineering checks.')
    if cfg['experiment'].get('oracle_debug') or cfg.get('logging', {}).get('publish_private_oracle_diagnostics'):
        raise ValueError('Oracle diagnostics require a separate offline process')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # The local Windows OpenMP runtime also corrupts large multithreaded
    # reductions. Use a single CPU thread; CUDA training remains parallel.
    torch.set_num_threads(cfg['train'].get('cpu_threads', 1))
    probe = torch.ones(2_000_000)
    if not torch.isclose(probe.mean(), torch.tensor(1.0)):
        raise RuntimeError('CPU parallel runtime failed reduction self-check; use cpu_threads=1')
    device = cfg['train'].get('device', 'auto')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    out = Path(output or f"results/{cfg['experiment']['name']}_seed{seed}_{time.time_ns()}")
    if resume:
        if not output or not out.is_dir() or cfg['method']['layer_freeze'] != 'none':
            raise ValueError('Resume supports existing P0 output directories only')
        if yaml.safe_load((out/'config.yaml').read_text(encoding='utf-8')) != cfg:
            raise ValueError('Resume configuration must match saved configuration exactly')
        if (out/'summary.json').exists():
            raise ValueError('Run already completed')
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out/'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    source_root = Path(__file__).resolve().parent
    (out/('resume_implementation.json' if resume else 'implementation.json')).write_text(json.dumps({
        'source_sha256': {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_root.glob('*.py')},
        'cpu_threads': torch.get_num_threads(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'normalization': 'numpy_float32_v2',
        'scope': 'independent prototype; paper equivalence unverified',
        'sampling_rng': 'private_os_seed_v2', 'noise_rng': 'private_os_seed_v1'
    }, indent=2), encoding='utf-8')
    scheduler = PrivacyScheduler(cfg)
    ledger = Ledger(scheduler.targets, cfg['privacy']['delta'], cfg['experiment']['privacy_claim_mode'], cfg['privacy']['adjacency'])
    train_data, test_data = load_data(cfg['data'])
    parts = dirichlet_split(train_data.tensors[1], cfg['data']['clients'], cfg['data']['dirichlet_alpha'], seed)
    model = build_model(cfg).to(device)
    registry = LayerRegistry(model)
    klass = FeedbackFreezer if cfg['method']['layer_freeze'] == 'dp_feedback' else StaticFreezer
    freezer = klass(registry.names, cfg['method'], seed)
    # A public seed reconstructing participation defeats hidden-sampling
    # amplification, even when the roster itself is omitted from logs.
    sample_rng = np.random.default_rng(secrets.randbits(128))
    # DP noise must not be reconstructible from the published experiment seed.
    # This simulation uses a private OS-derived seed; torch RNG is not a CSPRNG.
    noise_rng = torch.Generator(device=device).manual_seed(secrets.randbits(63))
    private = cfg['method']['temporal'] != 'fedavg'
    rounds, layers, privacy_rows = [], [], []
    first_round = 1
    if resume:
        def read_rows(file):
            with (out/file).open(encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                for key, value in row.items():
                    if value == '':
                        row[key] = None
                    elif key != 'layer':
                        try:
                            row[key] = int(value) if key in {'round','client_slot','mask','active_params','active_groups'} else float(value)
                        except ValueError:
                            pass
            return rows
        rounds, layers, privacy_rows = [read_rows(f) for f in ['rounds.csv','layers.csv','privacy.csv']]
        completed = int(rounds[-1]['round'])
        marker = out/'checkpoint.json'
        if marker.exists() and json.loads(marker.read_text())['round'] != completed:
            raise ValueError('Incomplete checkpoint: CSV/model round mismatch')
        saved = np.load(out/'accountant.npz')
        np.testing.assert_array_equal(saved['orders'], ORDERS)
        np.testing.assert_allclose(saved['targets'], scheduler.targets)
        # Recompute the prefix instead of trusting an arbitrary accountant file.
        for t in range(1, completed+1):
            if private:
                ledger.record(scheduler.params(t))
        np.testing.assert_allclose(saved['rdp'], ledger.rdp)
        model.load_state_dict(torch.load(out/'model.pt', map_location=device, weights_only=True))
        if (out/'failure.json').exists():
            (out/'failure.json').rename(out/f'failure_before_resume_{time.time_ns()}.json')
        first_round = completed+1
        print(f'Resuming P0 from round {first_round} with fresh private sampling/noise randomness', flush=True)
    started = time.perf_counter()
    for t in range(first_round, cfg['train']['rounds']+1):
        tick = time.perf_counter()
        rp = scheduler.params(t)
        active = freezer.decide(t)
        w = weights(registry.names, registry.dimensions, active, t, cfg['method'])
        validate(active, w)
        if private:
            ledger.record(rp)  # abort before querying or releasing if over budget
        selected = np.flatnonzero(sample_rng.random(len(parts)) < rp.q_n)
        template = registry.pack(model)
        messages = []
        lr = cfg['train']['lr']
        if cfg['train'].get('scheduler') == 'cosine':
            lr *= .5*(1+math.cos(math.pi*(t-1)/cfg['train']['rounds']))
        for n in selected:
            delta = local_update(model, registry, train_data, parts[n], active, cfg['train'], seed+t*10000+int(n), device, lr)
            if private:
                delta = clip(delta, float(rp.clip_n[n]), active, w, cfg['method']['layer_clip'])
                delta = add_client_noise(delta, float(rp.clip_n[n]*rp.sigma_n[n]), len(parts), active, noise_rng)
            messages.append(delta)
        scale = cfg['privacy']['average_clip_c']*rp.sigma_global
        released, variance = aggregate(messages, template, active, len(parts), rp.q_bar, scale, noise_rng)
        registry.apply(model, released)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError('Non-finite model after release; see noise scale and learning rate')
        if isinstance(freezer, FeedbackFreezer):
            freezer.observe(t, released, variance)
        loss, acc = evaluate(model, test_data, device)
        eps = ledger.epsilons() if private else np.zeros(len(parts))
        dimensions = sum(registry.dimensions[g] for g in active if active[g])
        # num_selected and local losses intentionally excluded from public outputs.
        rounds.append(dict(round=t, test_loss=loss, test_acc=acc, q_bar=rp.q_bar, epsilon_min=float(eps.min()), epsilon_median=float(np.median(eps)), epsilon_max=float(eps.max()), active_groups=sum(active.values()), active_params=dimensions, active_param_fraction=dimensions/sum(registry.dimensions.values()), noise_var_per_coordinate=variance, noise_energy_expected=dimensions*variance))
        for g in active:
            energy = float(released[g].double().square().sum())
            layers.append(dict(round=t, layer=g, mask=int(active[g]), weight=w[g], active_params=registry.dimensions[g]*int(active[g]), noise_tax=registry.dimensions[g]*variance*int(active[g]), released_dp_update_energy=energy, debiased_energy_score=energy-registry.dimensions[g]*variance if active[g] else None))
        for n in range(len(parts)):
            privacy_rows.append(dict(round=t, client_slot=n, target=float(scheduler.targets[n]), q=float(rp.q_n[n]), sigma=float(rp.sigma_n[n]), clip=float(rp.clip_n[n]), epsilon=float(eps[n])))
        write_csv(out/'rounds.csv', rounds)
        write_csv(out/'layers.csv', layers)
        write_csv(out/'privacy.csv', privacy_rows)
        # Recoverable round checkpoints; never contain individual updates,
        # sample lists or the private noise RNG state.
        torch.save(model.state_dict(), out/'model.pt')
        np.savez(out/'accountant.npz', orders=ORDERS, rdp=ledger.rdp, targets=scheduler.targets)
        (out/'checkpoint.json').write_text(json.dumps({'round':t}), encoding='utf-8')
        print(f'round={t} accuracy={acc:.3f}% epsilon_max={eps.max():.4f} active={dimensions} seconds={time.perf_counter()-tick:.1f}', flush=True)
    np.savez(out/'accountant.npz', orders=ORDERS, rdp=ledger.rdp, targets=scheduler.targets)
    torch.save(model.state_dict(), out/'model.pt')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    summary = dict(final_test_acc=rounds[-1]['test_acc'], best_test_acc=max(r['test_acc'] for r in rounds), test_loss=rounds[-1]['test_loss'], epsilon_n_final=eps.tolist() if private else None, budget_violation_rate=float((eps > scheduler.targets+1e-6).mean()) if private else None, privacy_enabled=private, split_checksum=split_checksum(parts), seed=seed, commit=commit, device=device, versions={p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'opacus', 'numpy', 'PyYAML']}, validation_status='prototype_not_paper_reproduction', data_normalization_version='numpy_float32_v2')
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if (out/'failure.json').exists():
        (out/'failure.json').rename(out/f'failure_before_resume_{time.time_ns()}.json')
    # Runtime, actual communication, selected-client counts and split indices are
    # intentionally not public artifacts: they can disclose participation/data.
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output')
    parser.add_argument('--resume', action='store_true', help='Resume P0 from validated model/accountant checkpoints')
    parser.add_argument('--smoke', action='store_true', help='Real-data engineering check, not reproduction')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.seed is not None:
        cfg['experiment']['seed'] = args.seed
    if args.smoke:
        cfg['experiment']['name'] += '_smoke'
        cfg['experiment']['engineering_only'] = True
        cfg['data'].update(clients=6, limit_train=600, limit_test=200)
        cfg['train'].update(rounds=2, local_epochs_or_iterations=1, batch_size=32)
        cfg['privacy']['transition_round'] = 2
    try:
        print(run(cfg, args.output, resume=args.resume))
    except Exception as exc:
        if args.output and Path(args.output).is_dir():
            (Path(args.output)/'failure.json').write_text(json.dumps({
                'exception':type(exc).__name__, 'message':str(exc), 'status':'failed'
            }, indent=2), encoding='utf-8')
        raise

if __name__ == '__main__':
    main()
