"""Run user-supplied source classes without altering their files/import tree.

Only private local diagnostic evaluations are suppressed. The upstream training,
optimizer, clipping, scaling, compensation and schedule methods execute verbatim.
Data use the clean official train/test split; upstream merged-split leakage is
explicitly not reproduced. This is a source-behavior audit, not DP certification.
"""
import ast
import argparse
import collections
import copy
import contextlib
import io
import secrets
import functools
import hashlib
import json
from pathlib import Path
import types
import numpy as np
import torch
import torchvision
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants.analysis import rdp
from .data import load_data, dirichlet_split, split_checksum
from .metrics import evaluate
from .main import write_csv

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'fl-experiments-code-submission'

def load_classes(relative, namespace):
    path=SOURCE/relative
    tree=ast.parse(path.read_text(encoding='utf-8'))
    tree.body=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef))]
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace

@functools.lru_cache(maxsize=4096)
def cached_noise(items):
    return get_noise_multiplier(**dict(items))

@functools.lru_cache(maxsize=4096)
def cached_rdp(q,noise_multiplier,steps,orders):
    return rdp.compute_rdp(q=q,noise_multiplier=noise_multiplier,steps=steps,orders=orders)

@functools.lru_cache(maxsize=4096)
def checked_source_noise(items,orders):
    """Validate utility calibration on the SAME orders as the source ledger."""
    kw=dict(items);sigma=cached_noise(items)
    steps=int(kw['epochs']/kw['sample_rate']) if 'epochs' in kw else kw['steps']
    for _ in range(10000):
        rho=cached_rdp(kw['sample_rate'],sigma,steps,orders)
        spent=rdp.get_privacy_spent(orders=orders,rdp=rho,delta=kw['target_delta'])[0]
        if spent<=kw['target_epsilon']:
            return sigma
        sigma*=1.001
    raise ValueError('Cannot match source accountant orders to target budget')

def source_model():
    ns=load_classes('src/models/model.py',dict(torch=torch,nn=nn,F=torch.nn.functional))
    return ns['get_model']

def source_partition(labels, clients):
    """Use author's minimum-size/balancing Dirichlet loop on clean IDs."""
    path=SOURCE/'src/datasets/dataset_utils.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='separate_data']
    ns=dict(np=np,least_samples=40.)
    exec(compile(tree,str(path),'exec'),ns)
    if len(labels)<40*clients:
        raise ValueError('Author partition requires at least 40 samples per client')
    with contextlib.redirect_stdout(io.StringIO()):
        indices,_,_=ns['separate_data']((np.arange(len(labels)),np.asarray(labels)),clients,10,True,True,'dir')
    return [np.asarray(v,dtype=int).tolist() for v in indices]

def build_source_classes(scheme='time_adaptive'):
    model=source_model()
    analysis=types.SimpleNamespace(get_privacy_spent=rdp.get_privacy_spent,
        compute_rdp=lambda **kw: cached_rdp(kw['q'],float(kw['noise_multiplier']),kw['steps'],tuple(kw['orders'])))
    ns=dict(np=np,copy=copy,torch=torch,nn=nn,optim=optim,DataLoader=DataLoader,
        torchvision=torchvision,transforms=transforms,get_model=model,
        CosineAnnealingLR=CosineAnnealingLR,Counter=collections.Counter,
        get_noise_multiplier=lambda **kw:cached_noise(tuple(sorted(kw.items()))),rdp=analysis)
    suffix='scheme2_time_adaptive_dpfl' if scheme=='time_adaptive' else 'scheme1_baseline'
    client=load_classes(f'src/clients/client_{suffix}.py',ns.copy())['Client']
    server=load_classes(f'src/server/server_{suffix}.py',ns.copy())['Server']
    return client,server,model

def joint_tensor_clip(client, parameters, num_layers):
    """Correct the source K-modules/M-tensors mismatch; unit scale unchanged."""
    radius=client.local_clip_value/np.sqrt(len(parameters))
    for vector in parameters.values():
        if not torch.isfinite(vector).all():
            vector.zero_()
        norm=vector.double().norm().item()
        vector.mul_(min(1.,radius/(norm+1e-12)))

def run(config, output):
    torch.set_num_threads(1)
    torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'config.yaml').write_text(yaml.safe_dump(config))
    manifest={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.rglob('*') if p.is_file()}
    (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'adapter_snapshot.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    train,test=load_data(dict(dataset='fmnist',root=config['data_root'],limit_train=config.get('limit_train'),limit_test=config.get('limit_test')))
    # Match upstream Normalize([0.5],[0.5]) without unsafe CPU in-place ops.
    train=torch.utils.data.TensorDataset(train.tensors[0]*2-1,train.tensors[1])
    test=torch.utils.data.TensorDataset(test.tensors[0]*2-1,test.tensors[1])
    partition=config.get('partition','legacy_dirichlet')
    if partition not in {'legacy_dirichlet','source_dirichlet'}:
        raise ValueError('Unknown data partition')
    parts=source_partition(train.tensors[1],config['clients']) if partition=='source_dirichlet' else dirichlet_split(train.tensors[1],config['clients'],.1,config['seed'])
    C,S,model=build_source_classes(config.get('scheme','time_adaptive'))
    profile=config.get('profile','source_compat')
    if profile not in {'source_compat','joint_corrected'}:
        raise ValueError('Unknown upstream profile')
    if profile=='joint_corrected':
        C.clip_layer_parameters=joint_tensor_clip
        sample_rng=np.random.default_rng(secrets.randbits(128))
        S.sample_clients=lambda self:np.flatnonzero(sample_rng.random(self.num_clients)<np.asarray(self.sample_rates)).tolist()
        noise_rng=torch.Generator().manual_seed(secrets.randbits(63))
        def private_noise(client,parameters):
            scale=client.local_noise_multiplier*client.local_clip_value/np.sqrt(client.num_clients)
            for value in parameters.values():
                value.add_(torch.randn(value.shape,generator=noise_rng).to(value.device)*float(scale))
        C.add_noise=private_noise
        # Server compensation also draws from an unpublished independent RNG:
        # source uses torch.normal, so isolate all source global torch randomness
        # from the published experiment seed after model construction below.
    # Disable only redundant local diagnostics; never publish their raw values.
    C.test_metrics=lambda self:(0,1)
    C.train_metrics=lambda self:(0.,1)
    C.evaluate=lambda self:0.
    from .original_scheduler import groups
    ids=groups(config['clients'],[.34,.43,.23]);np.random.shuffle(ids)
    budgets=np.asarray(config['epsilons'])[ids]
    rates=np.asarray(config['q_save'])[ids]
    states=[['save' if t<config['transition_round']-1 else 'spend' for t in range(config['rounds'])] for _ in ids]
    clients=[]
    adaptive=config.get('scheme','time_adaptive')=='time_adaptive'
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    for n in range(config['clients']):
        args=[n,Subset(train,parts[n]),test,config['local_epochs'],budgets[n],1e-5,config['rounds'],config['clients'],config['lr'],config['batch_size'],config['rate'],'fmnist',1000000]
        if adaptive:
            args.extend([states[n],rates[n],device])
        else:
            args.append(device)
        clients.append(C(*args))
    if profile=='joint_corrected':
        orders=tuple(clients[0].alphas)
        C.set_local_sigma_and_sample_rate.__globals__['get_noise_multiplier']=lambda **kw:checked_source_noise(tuple(sorted(kw.items())),orders)
    old=torchvision.datasets.FashionMNIST
    torchvision.datasets.FashionMNIST=lambda **kwargs:test
    try:
        args=[clients,config['clients'],config['rounds'],config['q_spend'],config['clip'],1e-5,config['lr'],config['batch_size'],config['rate'],0,'fmnist',1]
        if adaptive:args.extend([states,rates,device])
        else:args.append(device)
        server=S(*args)
    finally:
        torchvision.datasets.FashionMNIST=old
    if profile=='joint_corrected':
        torch.manual_seed(secrets.randbits(63))
    rows=[];privacy=[]
    for t in range(config['rounds']):
        server.sample_rates=[float(config['q_spend'])]*server.num_clients
        if adaptive:server.receive_local_sigmas(t)
        elif t==0:server.receive_local_sigmas()
        server.global_sigma=server.compute_global_sigma()
        server.send_global_privacy_parameters()
        # Source trains every client, including subsequently unsampled clients;
        # this preserves client momentum and the per-local-epoch scheduler.
        with contextlib.redirect_stdout(io.StringIO()):
            server.client_operations(t)
        if adaptive:server.update_sample_rate_local_accounting(t)
        server.sampled_clients=server.sample_clients()
        if server.sampled_clients:
            server.aggregate_sampled_clients();server.update_global_model()
        loss,acc=evaluate(server.global_model.to(device),test,device)
        eps=[float(c.dp_eps_spent) for c in clients]
        variance=(server.global_sigma*config['clip']*config['lr']*config['rate']/sum(server.sample_rates))**2
        dimensions=sum(p.numel() for p in server.global_model.parameters())
        rows.append(dict(round=t+1,test_acc=acc,test_loss=loss,epsilon_max_reported=max(eps),noise_var_per_coordinate=variance,noise_energy_expected=dimensions*variance))
        for n,c in enumerate(clients):
            privacy.append(dict(round=t+1,client_slot=n,target=float(budgets[n]),q=float(server.sample_rates[n]),sigma=float(c.local_noise_multiplier),clip=float(c.local_clip_value),epsilon_reported=eps[n]))
        write_csv(out/'rounds.csv',rows);write_csv(out/'privacy.csv',privacy)
        torch.save(server.global_model.state_dict(),out/'model.pt')
        print(f'upstream round={t+1} accuracy={acc:.2f}% epsilon_reported_max={max(eps):.4f}',flush=True)
    (out/'summary.json').write_text(json.dumps(dict(final_test_acc=rows[-1]['test_acc'],best_test_acc=max(r['test_acc'] for r in rows),seed=config['seed'],split_checksum=split_checksum(parts),parameters=sum(p.numel() for p in server.global_model.parameters()),profile=profile,partition=partition,status='source_training_semantics_clean_split_not_dp_certified',scope='public FMNIST benchmark',differences=['clean official train/test split',partition,'local diagnostic evaluation suppressed','single CPU thread; local data only'],privacy_warning='source_compat has a joint-sensitivity mismatch and public randomness; joint_corrected fixes tensor clipping and hides sampling/noise seeds but remains a research implementation, not production certification. See source audit.'),indent=2))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    if args.smoke:cfg.update(rounds=3,clients=6,local_epochs=1,limit_train=600,limit_test=200,transition_round=2)
    print(run(cfg,args.output))

if __name__=='__main__':main()
