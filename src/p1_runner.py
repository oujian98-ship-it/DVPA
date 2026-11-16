"""P1 static freezing on the author's persistent SGD/local-epoch pipeline."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import secrets
import time
import numpy as np
import torch
import yaml
from torch.utils.data import Subset, TensorDataset
from .data import load_data, split_checksum
from .main import write_csv
from .metrics import evaluate
from .original_scheduler import groups
from .upstream_runner import SOURCE, build_source_classes, source_partition, checked_source_noise
from .p1_freezing import StaticMask, group_counts, joint_group_clip, masked_noise, install_local_freeze, aggregate_active
from .p1_validation import check_protocol, check_round
from .p2_allocation import weights as allocation_weights, validate_allocation
from .p3_freezing import AdaptiveMask, validate_feedback_run

def run(config, output):
    cfg=copy.deepcopy(config)
    validate_feedback_run(cfg)
    allocation=cfg.get('allocation',{'layer_clip':'equal_active'})
    if 'allocation' in cfg:validate_allocation(allocation)
    preflight=check_protocol(cfg) if cfg.get('validation_protocol') else None
    if cfg.get('diagnostic_rng_path') and preflight is None:raise ValueError('Private RNG replay requires a locked diagnostic protocol')
    if preflight is None and not cfg.get('engineering_only',False):
        path=Path('reports/p0_gate.json')
        if not path.exists() or json.loads(path.read_text()).get('status')!='passed':
            raise RuntimeError('P0 gate not passed: P1/P2/P3 engineering checks require --smoke')
    if cfg.get('scheme','time_adaptive')!='time_adaptive':
        raise ValueError('P1 preserves the source time-adaptive scheduler')
    seed=int(cfg['seed']);torch.set_num_threads(1);torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=cfg.get('device','auto')
    if device=='auto':device='cuda:0' if torch.cuda.is_available() else 'cpu'
    train,test=load_data(dict(dataset='fmnist',root=cfg['data_root'],limit_train=cfg.get('limit_train'),limit_test=cfg.get('limit_test')))
    train=TensorDataset(train.tensors[0]*2-1,train.tensors[1]);test=TensorDataset(test.tensors[0]*2-1,test.tensors[1])
    parts=source_partition(train.tensors[1],cfg['clients'])
    Client,_,model_factory=build_source_classes('time_adaptive')
    Client.test_metrics=lambda self:(0,1);Client.train_metrics=lambda self:(0.,1);Client.evaluate=lambda self:0.
    ids=groups(cfg['clients'],[.34,.43,.23]);np.random.shuffle(ids)
    targets=np.asarray(cfg['epsilons'])[ids].astype(float)
    save_rates=np.asarray(cfg['q_save'])[ids].astype(float)
    n=cfg['clients'];T=cfg['rounds']
    clients=[Client(i,Subset(train,parts[i]),test,cfg['local_epochs'],targets[i],1e-5,T,n,cfg['lr'],cfg['batch_size'],cfg['rate'],'fmnist',1000000,[],save_rates[i],device) for i in range(n)]
    model=model_factory('fmnist',cfg['batch_size']).to(device)
    if list(model.named_buffers()):
        raise ValueError('P1 source runner currently supports the buffer-free author CNN only')
    if cfg['mode']=='ADAPTIVE':
        scheduler=(AdaptiveMask(model,cfg['feedback'],seed,cfg['diagnostic_freezing'],T) if 'diagnostic_freezing' in cfg else AdaptiveMask(model,cfg['feedback'],seed))
    else:
        scheduler=(StaticMask(model,cfg['mode'],cfg['freeze_rounds'],cfg.get('mask_seed',seed),cfg['static_order'])
                   if 'static_order' in cfg else StaticMask(model,cfg['mode'],cfg['freeze_rounds'],cfg.get('mask_seed',seed)))
    allocation_weights(scheduler.names,scheduler.dims,dict.fromkeys(scheduler.names,True),1,allocation)
    private=bool(cfg.get('privacy_enabled',True)) and cfg['mode']!='NO-DP'
    orders=tuple(clients[0].alphas)
    Client.set_local_sigma_and_sample_rate.__globals__['get_noise_multiplier']=lambda **kw:checked_source_noise(tuple(sorted(kw.items())),orders)
    if cfg.get('diagnostic_rng_path'):
        # D1 pair only: local secret entropy, never copy values to result artifacts.
        # Different policies beyond identical D1 replay use fresh independent noise.
        secret_state=json.loads(Path(cfg['diagnostic_rng_path']).read_text())
        sample_seed=secret_state['sampling'];noise_seed=secret_state['noise']
    else:sample_seed=secrets.randbits(128);noise_seed=secrets.randbits(63)
    sample_rng=np.random.default_rng(sample_seed)
    noise_rng=torch.Generator().manual_seed(noise_seed)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'config.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
    manifest={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.rglob('*') if p.is_file()}
    (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'implementation.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},indent=2))
    for p in [Path(__file__),Path(__file__).with_name('p1_freezing.py')]:
        (out/p.name).write_text(p.read_text(encoding='utf-8'),encoding='utf-8')
    (out/'layer_registry.json').write_text(json.dumps(dict(dimensions=scheduler.dims,order=scheduler.order,coordinate_control=cfg['mode']=='SAME-DIM'),indent=2))
    rows=[];layers=[];privacy=[];feedback_rows=[];freeze_first={};total_dims=sum(scheduler.dims.values())
    started=time.perf_counter()
    for t in range(1,T+1):
        masks=scheduler.decide(t)
        counts=group_counts(masks);active_dims=sum(counts.values())
        K=sum(v>0 for v in counts.values())
        omega=allocation_weights(scheduler.names,counts,{g:bool(d) for g,d in counts.items()},t,allocation)
        q=save_rates.copy() if t<cfg['transition_round'] else np.full(n,cfg['q_spend'])
        if not np.isfinite(q).all() or (q<=0).any() or (q>1).any():
            raise ValueError('Poisson rates must be in (0,1]')
        if private:
            sigmas=np.array([c.set_local_sigma_and_sample_rate(cfg['q_spend'],t-1) for c in clients])
            sigma_global=float(1/np.mean(1/sigmas))
            radii=np.array([c.set_local_clip_value(cfg['clip'],sigma_global) for c in clients])
            # Account before any query or release; do not depend on realized sampling.
            for i,c in enumerate(clients):
                c.rdp_accountant(sample_rate=float(q[i]))
                if c.dp_eps_spent>targets[i]+1e-6:
                    raise RuntimeError('Privacy target exceeded; release aborted')
        else:
            sigmas=np.zeros(n);radii=np.full(n,cfg['clip']);sigma_global=0.
            for c in clients:c.local_clip_value=cfg['clip'];c.local_noise_multiplier=0.;c.sample_rate=cfg['q_spend']
        if preflight is not None:
            check_round(preflight,t,clients,targets,q,sigmas,radii)
        selected=set(np.flatnonzero(sample_rng.random(n)<q).tolist())
        def clip(client,updates,num_layers):
            if private:joint_group_clip(updates,client.local_clip_value,masks,omega)
            else:
                for name,value in updates.items():value.masked_fill_(~masks[name].to(value.device),0)
        def noise(client,updates):
            if private:masked_noise(updates,masks,client.local_clip_value*client.local_noise_multiplier/np.sqrt(n),noise_rng)
        Client.clip_layer_parameters=clip;Client.add_noise=noise
        messages=[]
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        for i,c in enumerate(clients):
            cleanup=install_local_freeze(c,masks)
            try:
                with contextlib.redirect_stdout(io.StringIO()):message=c.train(model,t-1)[0]
            finally:cleanup()
            for name,value in c.local_model.state_dict().items():
                frozen=~masks[name].to(value.device)
                if not torch.equal(value[frozen],before[name][frozen]):
                    raise RuntimeError('Frozen local coordinates changed during training')
            if i in selected:messages.append(message)
        physical_scale=cfg['clip']*sigma_global*cfg['lr']*cfg['rate']
        delta,variance=aggregate_active(messages,before,masks,n,float(q.mean()),physical_scale,noise_rng)
        # Only the final compensated, normalized DP aggregate reaches feedback.
        # Its pending decisions cannot alter the mask used for this round.
        if isinstance(scheduler,AdaptiveMask):
            feedback_rows.extend(scheduler.observe(t,delta,variance))
        with torch.no_grad():
            for name,param in model.named_parameters():param.add_(delta[name])
        for name,value in model.state_dict().items():
            frozen=~masks[name].to(value.device)
            if not torch.equal(value[frozen],before[name][frozen]):raise RuntimeError('Frozen global coordinates changed')
        loss,acc=evaluate(model,test,device)
        eps=np.array([c.dp_eps_spent for c in clients]) if private else None
        rows.append(dict(round=t,test_acc=acc,test_loss=loss,privacy_enabled=private,q_bar=float(q.mean()),epsilon_min=float(eps.min()) if private else None,epsilon_max=float(eps.max()) if private else None,active_groups=K,active_params=active_dims,active_param_fraction=active_dims/total_dims,noise_var_per_coordinate=variance,noise_energy_expected=active_dims*variance,communication_bytes_expected=float(q.sum())*(2*active_dims*4+len(masks)),trainable_parameters_per_client=active_dims))
        for g in scheduler.names:
            if counts[g]<scheduler.dims[g]:freeze_first.setdefault(g,t)
            energy=sum(v.double().square().sum().item() for name,v in delta.items() if name.split('.')[0]==g)
            layers.append(dict(round=t,layer=g,mask=int(counts[g]>0),fully_active=int(counts[g]==scheduler.dims[g]),weight=omega[g],active_params=counts[g],active_fraction=counts[g]/scheduler.dims[g],clip_radius_mean=float(radii.mean())*np.sqrt(omega[g]) if private else None,clip_radius_min=float(radii.min())*np.sqrt(omega[g]) if private else None,clip_radius_max=float(radii.max())*np.sqrt(omega[g]) if private else None,noise_tax=counts[g]*variance,released_dp_update_energy=energy if private else None,freeze_round=freeze_first.get(g)))
        for i in range(n):
            privacy.append(dict(round=t,client_slot=i,q=float(q[i]),sigma=float(sigmas[i]) if private else None,clip=float(radii[i]) if private else None,target=float(targets[i]) if private else None,epsilon=float(eps[i]) if private else None))
        write_csv(out/'rounds.csv',rows);write_csv(out/'layers.csv',layers);write_csv(out/'privacy.csv',privacy)
        if isinstance(scheduler,AdaptiveMask):
            write_csv(out/'feedback.csv',feedback_rows)
            (out/'feedback_state.json').write_text(json.dumps(scheduler.controller.state_dict(),indent=2))
        torch.save(model.state_dict(),out/'model.pt')
        np.savez(out/'accountant.npz',orders=np.asarray(orders),rdp=np.asarray([c.rdp_eps_spent for c in clients]),targets=targets,privacy_enabled=private)
        stage='P3' if 'feedback' in cfg else ('P2' if 'allocation' in cfg else 'P1')
        print(f'{stage} {cfg["mode"]} round={t} accuracy={acc:.2f}% active={active_dims}/{total_dims}',flush=True)
    summary=dict(mode=cfg['mode'],seed=seed,mask_seed=cfg.get('mask_seed',seed),final_test_acc=rows[-1]['test_acc'],best_test_acc=max(r['test_acc'] for r in rows),privacy_enabled=private,epsilon_n_final=eps.tolist() if private else None,budget_violation_rate=float((eps>targets+1e-6).mean()) if private else None,split_checksum=split_checksum(parts),parameters=total_dims,freeze_first_round=freeze_first,status='controlled_validation' if preflight is not None else ('engineering_only' if cfg.get('engineering_only') else 'p1_experiment'),exact_privacy_preflight_matched=preflight is not None,train_flops=None,same_dim_scope='exact coordinates, not an alternative whole-block subset' if cfg['mode']=='SAME-DIM' else None)
    summary.update(stage='P3' if 'feedback' in cfg else ('P2' if 'allocation' in cfg else 'P1'),allocation=allocation)
    if 'feedback' in cfg:
        summary.update(feedback=scheduler.controller.config,feedback_scope='historical DP aggregate heuristic; no safety guarantee',
                       pending_after_final_round=sorted(scheduler.controller.pending))
    if summary['status']=='p1_experiment':summary['status']=summary['stage'].lower()+'_experiment'
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'runtime_local.json').write_text(json.dumps(dict(wall_time_sec=time.perf_counter()-started,scope='trusted local benchmark diagnostics; exclude from private transcript')))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.smoke:cfg.update(engineering_only=True,clients=6,rounds=4,local_epochs=1,limit_train=600,limit_test=200,transition_round=3,freeze_rounds=[] if cfg.get('mode')=='ADAPTIVE' else [2,3])
    try:print(run(cfg,args.output))
    except Exception as exc:
        if Path(args.output).is_dir():(Path(args.output)/'failure.json').write_text(json.dumps(dict(type=type(exc).__name__,message=str(exc))))
        raise

if __name__=='__main__':main()
