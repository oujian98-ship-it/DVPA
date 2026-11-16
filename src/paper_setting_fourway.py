"""Four time/layer privacy allocations on the actual author training pipeline.

Layer noise is diagonal, with a JOINT whitened sensitivity accounted once
using the client's Poisson probability. Never sum amplified per-layer costs.
"""
import contextlib
import csv
import functools
import hashlib
import io
import json
import math
from pathlib import Path
import secrets

import numpy as np
from opacus.accountants.analysis import rdp
import torch
from torch.utils.data import Subset, TensorDataset
import yaml

from . import upstream_runner as upstream
from .data import load_data, split_checksum
from .metrics import evaluate
from .original_scheduler import groups
from .p1_freezing import StaticMask, joint_group_clip
from .st_audit import close
from .st_runner import write_csv


METHODS = ('UNIFORM', 'TIME_ONLY', 'LAYER_ONLY', 'JOINT')
AUTHOR_ROOT = Path(__file__).resolve().parents[2]/'fl-experiments-code-submission'


def source_classes():
    # Only this adapter's module reference changes, never the original files.
    upstream.SOURCE = AUTHOR_ROOT
    return upstream.build_source_classes('time_adaptive')


@functools.lru_cache(maxsize=8192)
def increment(q, sigma, orders):
    value = np.asarray(rdp.compute_rdp(q=float(q), noise_multiplier=float(sigma), steps=1, orders=orders))
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError('Invalid sampled-Gaussian RDP increment')
    return value


def eps_from_rdp(vector, orders, delta):
    if not np.any(vector):
        return 0.
    return float(rdp.get_privacy_spent(orders=orders, rdp=vector, delta=delta)[0])


def calibrate_uniform(actual_target, q, rounds, orders, delta, initial_sigma):
    """Constant q/sigma => exactly equal RDP increments each round, all orders."""
    lo, hi = 0., float(initial_sigma)
    while eps_from_rdp(rounds*increment(q, hi, orders), orders, delta) > actual_target:
        hi *= 2
    for _ in range(34):
        mid = (lo+hi)/2
        if eps_from_rdp(rounds*increment(q, mid, orders), orders, delta) > actual_target:
            lo = mid
        else:
            hi = mid
    return hi


def validate(cfg):
    if cfg['seed'] != 42:
        raise ValueError('This experiment authorizes seed42 only')
    if cfg['dataset'] != 'fmnist' or cfg['freeze_rounds']:
        raise ValueError('FMNIST without freezing is required for this budget-only comparison')
    if cfg['partition'] != 'source_dirichlet_alpha_0.1' or cfg['normalization'] != 'minus_one_to_one':
        raise ValueError('Require author partition and input normalization')
    for name in ('clients', 'rounds', 'local_epochs', 'batch_size'):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError(f'Invalid {name}')
    if not 1 <= cfg['transition_round'] <= cfg['rounds']:
        raise ValueError('Invalid transition')
    if not 0 < cfg['delta'] < 1 or len(cfg['epsilons']) != 3 or any(e <= 0 for e in cfg['epsilons']):
        raise ValueError('Invalid privacy targets')
    if not all(0 < q <= cfg['q_spend'] <= 1 for q in cfg['q_save']):
        raise ValueError('Invalid sampling schedule')
    groups(cfg['clients'], cfg['group_proportions'])


def preflight(cfg):
    """Compute the author's public time schedule and matched uniform controls."""
    validate(cfg)
    torch.set_num_threads(1)
    Client, _, _ = source_classes()
    clients = [Client(g, None, None, cfg['local_epochs'], e, cfg['delta'], cfg['rounds'], cfg['clients'],
        cfg['lr'], cfg['batch_size'], cfg['rate'], 'fmnist', 1000000, [], cfg['q_save'][g], 'cpu')
        for g,e in enumerate(cfg['epsilons'])]
    orders = tuple(clients[0].alphas)
    Client.set_local_sigma_and_sample_rate.__globals__['get_noise_multiplier'] = (
        lambda **kw: upstream.checked_source_noise(tuple(sorted(kw.items())), orders))
    time_q, time_sigma = [], []
    for t in range(1, cfg['rounds']+1):
        sigmas = [float(c.set_local_sigma_and_sample_rate(cfg['q_spend'], t-1)) for c in clients]
        qs = list(cfg['q_save']) if t < cfg['transition_round'] else [cfg['q_spend']]*3
        for c,q in zip(clients, qs):
            c.rdp_accountant(sample_rate=q)
        time_q.append(qs)
        time_sigma.append(sigmas)
    actual = [float(c.dp_eps_spent) for c in clients]
    if any(used > cap+1e-6 for used,cap in zip(actual, cfg['epsilons'])):
        raise RuntimeError('Source schedule exceeds the registered group caps')
    print(f'Public source schedule ready; actual group epsilons={actual}', flush=True)
    uniform_sigma = []
    for g,e in enumerate(actual):
        s = calibrate_uniform(e, cfg['q_spend'], cfg['rounds'], orders, cfg['delta'],
                              max(row[g] for row in time_sigma))
        uniform_sigma.append(s)
        print(f'Uniform group {g}: sigma={s:.9f}; matched actual epsilon={e:.9f}', flush=True)
    return dict(orders=list(orders), time_q=time_q, time_sigma=time_sigma,
                uniform_q=[[cfg['q_spend']]*3 for _ in range(cfg['rounds'])],
                uniform_sigma=[uniform_sigma.copy() for _ in range(cfg['rounds'])],
                matched_actual_epsilon=actual,
                rule='retain source time trajectory; uniform controls match its actual final group epsilon')


def layer_allocation(names, t, method, cfg):
    if method not in METHODS:
        raise ValueError('Unknown factorial method')
    if set(names) != set(cfg['layer_decay']):
        raise ValueError('Layer prior must match actual author CNN')
    omega = dict.fromkeys(names, 1/len(names))
    if method in {'LAYER_ONLY', 'JOINT'}:
        logs = np.array([-cfg['layer_decay'][g]*(t-1) for g in names], dtype=float)
        if not np.isfinite(logs).all():
            raise ValueError('Invalid public layer prior')
        scores = np.exp(np.maximum(logs-logs.max(), -60.))
        p = dict(zip(names, (scores/scores.sum()).tolist()))
    else:
        p = omega.copy()
    return omega, p


def round_parameters(cfg, plan, method, t, group_counts, names):
    temporal = method in {'TIME_ONLY', 'JOINT'}
    q = np.asarray(plan['time_q' if temporal else 'uniform_q'][t-1])
    sigma = np.asarray(plan['time_sigma' if temporal else 'uniform_sigma'][t-1])
    counts = np.asarray(group_counts, dtype=float)
    global_sigma = float(counts.sum()/np.sum(counts/sigma))
    radius = cfg['clip']*global_sigma/sigma
    denominator = float(np.dot(counts, q))
    omega, p = layer_allocation(names, t, method, cfg)
    physical_scale = cfg['clip']*global_sigma*cfg['lr']*cfg['rate']
    variance = {k: (physical_scale/denominator)**2*omega[k]/p[k] for k in names}
    # Recompute the actual JOINT whitened sensitivity for each privacy group.
    effective = []
    for c in radius:
        squared = math.fsum((c*math.sqrt(omega[k])*cfg['lr']*cfg['rate']/denominator)**2
                            /variance[k] for k in names)
        effective.append(1/math.sqrt(squared))
    np.testing.assert_allclose(effective, sigma, rtol=1e-12, atol=1e-12)
    return dict(q=q, sigma=sigma, effective_sigma=np.asarray(effective), radii=radius,
                global_sigma=global_sigma, denominator=denominator, omega=omega, p=p,
                variance=variance, physical_scale=physical_scale)


def add_diagonal_noise(updates, scales, generator):
    for name,value in updates.items():
        value.add_(torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
                   *float(scales[name.split('.')[0]]))


def run(cfg, plan, method, output):
    validate(cfg)
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = cfg['device']
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train,test = load_data(dict(dataset='fmnist', root=cfg['data_root'],
        limit_train=cfg.get('limit_train'), limit_test=cfg.get('limit_test')))
    train = TensorDataset(train.tensors[0]*2-1, train.tensors[1])
    test = TensorDataset(test.tensors[0]*2-1, test.tensors[1])
    Client, _, model_factory = source_classes()
    parts = upstream.source_partition(train.tensors[1], cfg['clients'])
    ids = groups(cfg['clients'], cfg['group_proportions'])
    np.random.shuffle(ids)
    counts = np.bincount(ids, minlength=3)
    Client.test_metrics = lambda self: (0,1)
    Client.train_metrics = lambda self: (0.,1)
    Client.evaluate = lambda self: 0.
    clients = [Client(i, Subset(train,parts[i]), test, cfg['local_epochs'], cfg['epsilons'][ids[i]],
        cfg['delta'], cfg['rounds'], cfg['clients'], cfg['lr'], cfg['batch_size'], cfg['rate'],
        'fmnist', 1000000, [], cfg['q_save'][ids[i]], device) for i in range(cfg['clients'])]
    model = model_factory('fmnist', cfg['batch_size']).to(device)
    if list(model.named_buffers()):
        raise ValueError('Private model buffers are unsupported')
    mask = StaticMask(model, 'NF', [])
    masks = mask.decide(1)
    names = mask.names
    if sum(mask.dims.values()) != 582026:
        raise ValueError('Expected the actual author CNN with 582026 parameters')
    orders = tuple(plan['orders'])
    np.testing.assert_array_equal(clients[0].alphas, orders)
    sample_rng = np.random.default_rng(secrets.randbits(128))
    noise_rng = torch.Generator(device=device).manual_seed(secrets.randbits(63))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'config.yaml').write_text(yaml.safe_dump({**cfg, 'method':method}), encoding='utf-8')
    state_hash = hashlib.sha256()
    for value in model.state_dict().values():
        state_hash.update(value.detach().cpu().numpy().tobytes())
    metadata = dict(method=method, seed=cfg['seed'], parameters=sum(mask.dims.values()),
        initial_model_sha256=state_hash.hexdigest(), split_checksum=split_checksum(parts),
        group_assignment=ids.tolist(), group_counts=counts.tolist(), dimensions=mask.dims,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        scope='public FMNIST benchmark; privacy accounting conditional on fixed client partitions')
    (out/'implementation.json').write_text(json.dumps(metadata,indent=2), encoding='utf-8')
    ledger = np.zeros((3,len(orders)))
    rounds,layers,privacy = [],[],[]
    for t in range(1,cfg['rounds']+1):
        rp = round_parameters(cfg,plan,method,t,counts,names)
        candidate = ledger+np.array([increment(q,s,orders) for q,s in zip(rp['q'],rp['effective_sigma'])])
        eps = [eps_from_rdp(v,orders,cfg['delta']) for v in candidate]
        if any(e>target+1e-6 for e,target in zip(eps,cfg['epsilons'])):
            raise RuntimeError('Sampled-Gaussian ledger rejected release before private training')
        ledger = candidate
        for i,c in enumerate(clients):
            g = ids[i]
            c.local_noise_multiplier = float(rp['sigma'][g])
            c.local_clip_value = float(rp['radii'][g])
            c.sample_rate = float(rp['q'][g])
            c.dp_eps_spent = eps[g]
        def clip(client, updates, num_layers):
            joint_group_clip(updates,client.local_clip_value,masks,rp['omega'])
        def noise(client, updates):
            base = client.local_clip_value*client.local_noise_multiplier/math.sqrt(cfg['clients'])
            add_diagonal_noise(updates,{k:base*math.sqrt(rp['omega'][k]/rp['p'][k]) for k in names},noise_rng)
        Client.clip_layer_parameters = clip
        Client.add_noise = noise
        selected = set(np.flatnonzero(sample_rng.random(cfg['clients']) < rp['q'][ids]).tolist())
        total = {name:torch.zeros_like(value) for name,value in model.state_dict().items()}
        for i,c in enumerate(clients):
            # Native training executes for ALL clients: retain momentum/cosine
            # state even when a client is not selected for aggregation.
            with contextlib.redirect_stdout(io.StringIO()):
                message = c.train(model,t-1)[0]
            if i in selected:
                for name in total:
                    total[name].add_(message[name])
            del message
        correction = math.sqrt((cfg['clients']-len(selected))/cfg['clients'])
        compensation_scales = {k:rp['physical_scale']*correction*math.sqrt(rp['omega'][k]/rp['p'][k]) for k in names}
        add_diagonal_noise(total,compensation_scales,noise_rng)
        delta = {name:value/rp['denominator'] for name,value in total.items()}
        with torch.no_grad():
            for name,param in model.named_parameters():
                param.add_(delta[name])
        loss,acc = evaluate(model,test,device)
        if not math.isfinite(loss):
            raise RuntimeError('Non-finite public evaluation; retain failed run')
        rounds.append(dict(round=t,test_acc=acc,test_loss=loss,
            epsilon_10=eps[0],epsilon_20=eps[1],epsilon_30=eps[2],
            q_bar=rp['denominator']/cfg['clients'],
            expected_noise_energy=sum(mask.dims[k]*rp['variance'][k] for k in names)))
        for k in names:
            layers.append(dict(round=t,layer=k,dimensions=mask.dims[k],clip_weight=rp['omega'][k],
                budget_fraction=rp['p'][k],noise_variance=rp['variance'][k],
                noise_std=math.sqrt(rp['variance'][k]),expected_noise_energy=mask.dims[k]*rp['variance'][k]))
        for i in range(cfg['clients']):
            g = ids[i]
            privacy.append(dict(round=t,client_slot=i,group=int(g),target=cfg['epsilons'][g],
                q=rp['q'][g],sigma_effective=rp['effective_sigma'][g],
                clip_radius=rp['radii'][g],denominator=rp['denominator'],epsilon=eps[g]))
        write_csv(out/'rounds.csv',rounds)
        write_csv(out/'layers.csv',layers)
        write_csv(out/'privacy.csv',privacy)
        np.savez(out/'accountant.npz',orders=np.asarray(orders),group_rdp=ledger,
            client_rdp=ledger[ids],targets=np.asarray(cfg['epsilons'])[ids],group_assignment=ids)
        torch.save(model.state_dict(),out/'model.pt')
        print(f'PAPER {method} seed42 round={t} accuracy={acc:.2f}% actual_epsilon={eps}',flush=True)
    summary = dict(status='completed',method=method,seed=cfg['seed'],rounds=cfg['rounds'],
        best_test_accuracy=max(row['test_acc'] for row in rounds),final_test_accuracy=rounds[-1]['test_acc'],
        group_epsilon_final=eps,group_epsilon_targets=cfg['epsilons'],parameters=sum(mask.dims.values()),
        privacy_accounting='joint whitened sampled Gaussian RDP, extended source orders',
        sampling_amplification=True,freezing=False,source_model_and_optimizer=True,
        expected_client_rounds=sum(row['q_bar']*cfg['clients'] for row in rounds),
        data_scope='official public FMNIST split; conditional-on-partition privacy accounting')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def audit(directory,plan):
    root = Path(directory)
    cfg = yaml.safe_load((root/'config.yaml').read_text(encoding='utf-8'))
    method = cfg.pop('method')
    validate(cfg)
    meta = json.loads((root/'implementation.json').read_text(encoding='utf-8'))
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    def read(name):
        with (root/name).open(encoding='utf-8',newline='') as f:
            return list(csv.DictReader(f))
    layer_rows,priv,rounds = read('layers.csv'),read('privacy.csv'),read('rounds.csv')
    names = list(meta['dimensions'])
    counts = meta['group_counts']
    ids = np.asarray(meta['group_assignment'])
    if len(priv)!=cfg['rounds']*cfg['clients'] or len(layer_rows)!=cfg['rounds']*len(names) or len(rounds)!=cfg['rounds']:
        raise ValueError('Incomplete audit logs')
    orders = tuple(plan['orders'])
    ledger = np.zeros((3,len(orders)))
    for t in range(1,cfg['rounds']+1):
        rp = round_parameters(cfg,plan,method,t,counts,names)
        lrows = [r for r in layer_rows if int(r['round'])==t]
        if len(lrows)!=len(names) or {r['layer'] for r in lrows}!=set(names):
            raise ValueError('Missing/duplicate layers')
        variance = {}
        for row in lrows:
            k = row['layer']
            for key,value in dict(dimensions=meta['dimensions'][k],clip_weight=rp['omega'][k],
                budget_fraction=rp['p'][k],noise_variance=rp['variance'][k],
                noise_std=math.sqrt(rp['variance'][k]),
                expected_noise_energy=meta['dimensions'][k]*rp['variance'][k]).items():
                close(float(row[key]),value,f'{k}.{key}')
            variance[k] = float(row['noise_variance'])
        effective = [1/math.sqrt(math.fsum((c*math.sqrt(rp['omega'][k])*cfg['lr']*cfg['rate']/rp['denominator'])**2
                                          /variance[k] for k in names)) for c in rp['radii']]
        ledger += np.array([increment(q,s,orders) for q,s in zip(rp['q'],effective)])
        eps = [eps_from_rdp(v,orders,cfg['delta']) for v in ledger]
        prows = [r for r in priv if int(r['round'])==t]
        if len(prows)!=cfg['clients'] or {int(r['client_slot']) for r in prows}!=set(range(cfg['clients'])):
            raise ValueError('Missing/duplicate client slots')
        for row in prows:
            g = int(ids[int(row['client_slot'])])
            for key,value in dict(group=g,target=cfg['epsilons'][g],q=rp['q'][g],
                sigma_effective=effective[g],clip_radius=rp['radii'][g],denominator=rp['denominator'],epsilon=eps[g]).items():
                close(float(row[key]),value,key)
        if any(e>target+1e-6 for e,target in zip(eps,cfg['epsilons'])):
            raise RuntimeError('Audit found over-budget release')
        for g,key in enumerate(('epsilon_10','epsilon_20','epsilon_30')):
            close(float(rounds[t-1][key]),eps[g],key)
    np.testing.assert_allclose(summary['group_epsilon_final'],eps,rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose(eps,plan['matched_actual_epsilon'],rtol=1e-7,atol=1e-6)
    close(summary['best_test_accuracy'],max(float(r['test_acc']) for r in rounds),'best accuracy')
    close(summary['final_test_accuracy'],float(rounds[-1]['test_acc']),'final accuracy')
    with np.load(root/'accountant.npz') as saved:
        np.testing.assert_array_equal(saved['orders'],orders)
        np.testing.assert_allclose(saved['group_rdp'],ledger,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(saved['client_rdp'],ledger[ids],rtol=1e-10,atol=1e-10)
    return dict(status='passed',rounds=cfg['rounds'],group_epsilon_actual=eps,
                group_epsilon_caps=cfg['epsilons'],all_client_slots=cfg['clients'],
                scope='actual diagonal noise, joint sensitivity and complete sampled RDP ledger audited')
