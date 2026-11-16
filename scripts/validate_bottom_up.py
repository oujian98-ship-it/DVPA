"""Pre-register and execute full, equal-RDP P1 comparison on public FMNIST."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from src.upstream_runner import build_source_classes, checked_source_noise, SOURCE
from src.original_scheduler import groups
from src.p1_runner import run

def preflight(cfg,path):
    torch.set_num_threads(1)
    C,_,_=build_source_classes('time_adaptive')
    data=torch.utils.data.TensorDataset(torch.zeros(1,1,28,28),torch.zeros(1,dtype=torch.long))
    budgets=np.asarray(cfg['epsilons'],dtype=float)
    clients=[C(i,data,data,0,e,1e-5,cfg['rounds'],cfg['clients'],cfg['lr'],cfg['batch_size'],cfg['rate'],'fmnist',1000000,[],cfg['q_save'][i],'cpu') for i,e in enumerate(budgets)]
    orders=tuple(clients[0].alphas)
    C.set_local_sigma_and_sample_rate.__globals__['get_noise_multiplier']=lambda **kw:checked_source_noise(tuple(sorted(kw.items())),orders)
    ids=groups(cfg['clients'],[.34,.43,.23]);rho=[];params=[];eps=[]
    for t in range(1,cfg['rounds']+1):
        sigmas=np.array([c.set_local_sigma_and_sample_rate(cfg['q_spend'],t-1) for c in clients])
        sg=float(1/np.mean(1/sigmas[ids]))
        radii=np.array([c.set_local_clip_value(cfg['clip'],sg) for c in clients])
        q=np.asarray(cfg['q_save']) if t<cfg['transition_round'] else np.full(3,cfg['q_spend'])
        for i,c in enumerate(clients):
            c.rdp_accountant(sample_rate=float(q[i]))
            if c.dp_eps_spent>budgets[i]+1e-6:raise RuntimeError('Preflight exceeds budget')
        rho.append(np.array([c.rdp_eps_spent for c in clients]))
        params.append(np.column_stack([q,sigmas,radii]))
        eps.append([c.dp_eps_spent for c in clients])
    np.savez(path,orders=np.asarray(orders),targets=budgets,rdp=np.asarray(rho),params=np.asarray(params),epsilon=np.asarray(eps))

def main():
    p=argparse.ArgumentParser();p.add_argument('--results-dir',required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
    root=Path(args.results_dir).resolve()
    if not args.resume:
        root.mkdir(parents=True,exist_ok=False)
        base=yaml.safe_load(Path('configs/p1_nf.yaml').read_text())
        fixed={k:v for k,v in base.items() if k not in {'seed','mask_seed','mode'}}
        preflight(fixed,root/'privacy_preflight.npz')
        sources=[p for p in SOURCE.rglob('*') if p.is_file()]
        sources.extend(Path('src').glob('*.py'))
        sources.append(Path(__file__))
        protocol=dict(kind='controlled_p1_comparison_v1',authorization='User explicitly requested controlled BU vs NF/RAND/TD validation; P0 gate is not marked passed',fixed_config=fixed,training_seeds=[42,43,44],random_mask_seeds=[11,23,37],privacy_preflight='privacy_preflight.npz',privacy_preflight_sha256=hashlib.sha256((root/'privacy_preflight.npz').read_bytes()).hexdigest(),source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},primary_metric='final_test_acc',freeze_rounds=[9,17],delta=1e-5,selection_policy='all registered runs; no tuning or seed selection',publication_scope='public FMNIST research experiment; per-run privacy accounting, not combined privacy of releasing all experiments')
        (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    jobs=[]
    # Complete one whole training-seed block before moving to another.
    for seed in protocol['training_seeds']:
        for mode in ['NF','BU','TD','RAND']:
            for ms in protocol['random_mask_seeds'] if mode=='RAND' else [seed]:
                jobs.append((seed,mode,ms))
    status=[]
    for seed,mode,ms in jobs:
        directory=root/f'{mode}_seed{seed}_mask{ms}'
        if args.resume and (directory/'summary.json').exists():
            status.append(dict(run=directory.name,status='completed'));continue
        if directory.exists():
            raise RuntimeError(f'Partial run preserved at {directory}; do not silently replace or select another seed')
        cfg=copy.deepcopy(protocol['fixed_config']);cfg.update(seed=seed,mode=mode,mask_seed=ms,validation_protocol=str(root/'protocol.json'))
        (root/'status.json').write_text(json.dumps(dict(state='running',current=directory.name,completed=status,total=len(jobs)),indent=2))
        try:
            run(cfg,directory)
        except Exception as exc:
            (root/'status.json').write_text(json.dumps(dict(state='failed',current=directory.name,error=str(exc),completed=status,total=len(jobs)),indent=2));raise
        status.append(dict(run=directory.name,status='completed'))
    (root/'status.json').write_text(json.dumps(dict(state='completed',completed=status,total=len(jobs)),indent=2))
    print(root,flush=True)

if __name__=='__main__':main()
