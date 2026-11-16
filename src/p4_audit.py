"""Read-only consistency audit for the author-runner P1/P2/P3 artifact schema."""
import csv
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
import yaml
from opacus.accountants.analysis import rdp


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def number(value, name, minimum=None):
    value=float(value)
    if not math.isfinite(value) or (minimum is not None and value<minimum):raise ValueError(f'Invalid {name}')
    return value


def boolean(value):
    if value is True or value=='True':return True
    if value is False or value=='False':return False
    raise ValueError('Expected explicit boolean')


def close(actual, expected, name, *, atol=1e-9):
    try:np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=atol)
    except AssertionError as exc:raise ValueError(f'{name} mismatch') from exc


def spent(orders, vector, delta):
    return 0. if not np.any(vector) else float(rdp.get_privacy_spent(orders=orders,rdp=vector,delta=delta)[0])


@lru_cache(maxsize=4096)
def increment(orders, q, sigma):
    a=np.asarray(rdp.compute_rdp(q=q,noise_multiplier=sigma,steps=1,orders=np.asarray(orders)))
    a.setflags(write=False);return a


def audit_feedback(folder, cfg, registry, grouped, rounds):
    if 'diagnostic_freezing' in cfg:
        from .diagnostic_freezing import audit_diagnostic_feedback
        return audit_diagnostic_feedback(folder,cfg,registry,grouped,rounds)
    from .dp_feedback_freezer import validate_feedback
    policy=validate_feedback(cfg['feedback']);T=cfg['rounds'];names=list(registry)
    if policy['min_active_groups']>len(names):raise ValueError('Invalid protected group count')
    protected=set(names[-policy['min_active_groups']:]);rows=read_csv(folder/'feedback.csv')
    indexed={}
    for row in rows:
        key=(int(row['round']),row['layer'])
        if key in indexed or not 1<=key[0]<=T or key[1] not in registry:raise ValueError('Duplicate/invalid feedback row')
        indexed[key]=row
    if len(indexed)!=T*len(names):raise ValueError('Incomplete feedback trace')
    active=dict.fromkeys(names,True);ema={};low=dict.fromkeys(names,0);pending=set();last_freeze=None;frozen={};safe=[]
    for t in range(1,T+1):
        for g in pending:active[g]=False;low[g]=0;frozen[g]=t;last_freeze=t
        cooling=last_freeze is not None and t+1-last_freeze<=policy['cooldown_rounds']
        candidates=[];scores={};taxes={};next_flags=set()
        for g in names:
            row=indexed[(t,g)];observed=boolean(row['observed']);flag=boolean(row['freeze_next_round'])
            d=int(grouped[t][g]['active_params'])
            if observed!=active[g] or d!=(registry[g] if active[g] else 0):raise ValueError('Feedback must use previous pending mask, never same-round evidence')
            if boolean(row['protected'])!=(g in protected) or boolean(row['cooldown'])!=cooling:raise ValueError('Feedback policy flag mismatch')
            clean=dict(round=t,layer=g,observed=observed,freeze_next_round=flag)
            fields=['dimension','released_energy','noise_variance','noise_tax','debiased_energy','nonnegative_score','score','threshold','low_count']
            if not observed:
                if flag or any(row[k] not in {'',None} for k in fields):raise ValueError('Frozen zeros cannot become observations')
                clean.update({k:None for k in fields});safe.append(clean);continue
            variance=float(rounds[t-1]['noise_var_per_coordinate']);tax=d*variance
            energy=number(grouped[t][g]['released_dp_update_energy'],'DP energy',0.)
            score=energy if policy['feedback_score']=='raw' else max(0.,energy-tax)
            decay=policy['ema_decay'] if policy['feedback_score']=='ema' else 0.
            smooth=decay*ema.get(g,score)+(1-decay)*score;threshold=policy['threshold_ratio']*tax
            low[g]=low[g]+1 if t>=policy['warmup_rounds'] and smooth<threshold else 0;ema[g]=smooth
            expected=[d,energy,variance,tax,energy-tax,score,smooth,threshold,low[g]]
            for key,value in zip(fields,expected):close(number(row[key],key),value,'feedback '+key)
            clean.update(dict(zip(fields,expected)));safe.append(clean);scores[g]=smooth;taxes[g]=tax
            if flag:next_flags.add(g)
            if low[g]>=policy['patience'] and g not in protected and not cooling:candidates.append(g)
        candidates.sort(key=lambda g:(scores[g]/taxes[g],names.index(g)))
        pending=set(candidates[:min(policy['max_freeze_per_round'],sum(active.values())-policy['min_active_groups'])])
        if pending!=next_flags:raise ValueError('Feedback pending decision mismatch')
    state=json.loads((folder/'feedback_state.json').read_text())
    if (state['last_observed']!=T or state['last_decided']!=T or state['active']!=active or
        set(state['pending'])!=pending or state['freeze_round']!=frozen or state['names']!=names or
        state['config']!=policy or state['low']!=low or state['last_freeze_round']!=last_freeze):
        raise ValueError('Feedback final state mismatch')
    if set(state['ema'])!=set(ema):raise ValueError('Feedback EMA registry mismatch')
    for g in ema:close(state['ema'][g],ema[g],'feedback final EMA')
    return safe


def audit_run(folder, *, accuracy_target=None, allow_engineering=False, include_local_cost=False):
    folder=Path(folder)
    artifact_names={'config.yaml','summary.json','layer_registry.json','source_manifest.json',
                    'rounds.csv','layers.csv','privacy.csv','accountant.npz','feedback.csv','feedback_state.json','implementation.json'}
    if include_local_cost:artifact_names.add('runtime_local.json')
    snapshot=lambda:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.name in artifact_names}
    initial_hashes=snapshot()
    cfg=yaml.safe_load((folder/'config.yaml').read_text(encoding='utf-8'))
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    registry=json.loads((folder/'layer_registry.json').read_text())['dimensions']
    source=json.loads((folder/'source_manifest.json').read_text())
    from .p3_freezing import validate_feedback_run
    validate_feedback_run(cfg)
    if summary.get('mode')!=cfg.get('mode') or summary.get('seed')!=cfg.get('seed'):raise ValueError('Summary method/seed mismatch')
    if cfg.get('profile')!='joint_corrected':raise ValueError('P4 requires joint-corrected author artifacts; legacy/source-compatible profiles are not DP comparison evidence')
    if not isinstance(registry,dict) or not registry or any(type(d) is not int or d<1 for d in registry.values()):
        raise ValueError('Invalid layer registry')
    model_hash={k.replace('\\','/'):v for k,v in source.items()}.get('src/models/model.py')
    if not model_hash:raise ValueError('Missing source model provenance')
    if summary.get('privacy_enabled') is not True or cfg.get('privacy_enabled',True) is not True:
        raise ValueError('P4 DP comparison cannot include non-DP runs')
    if summary.get('status')=='engineering_only' or cfg.get('engineering_only',False):
        if not allow_engineering:raise ValueError('Engineering runs cannot enter formal tables')
    if summary.get('status') not in {'controlled_validation','p1_experiment','p2_experiment','p3_experiment','engineering_only'}:
        raise ValueError('Unverified or unsupported run status')
    if (folder/'failure.json').exists():raise ValueError('Failed run cannot enter tables')
    T=cfg['rounds'];n=cfg['clients'];delta=number(cfg.get('delta',1e-5),'delta',0.)
    if type(T) is not int or T<1 or type(n) is not int or n<1 or not 0<delta<1:raise ValueError('Invalid training/privacy dimensions')
    with np.load(folder/'accountant.npz',allow_pickle=False) as archive:
        orders=archive['orders'].copy();rho=archive['rdp'].copy();targets=archive['targets'].copy()
    if orders.ndim!=1 or len(orders)==0 or not np.isfinite(orders).all() or (orders<=1).any() or len(set(orders))!=len(orders):
        raise ValueError('Invalid RDP orders')
    if rho.shape!=(n,len(orders)) or targets.shape!=(n,) or not np.isfinite(rho).all() or (rho<0).any() or not np.isfinite(targets).all() or (targets<=0).any():
        raise ValueError('Invalid accountant dimensions/values')
    rounds=read_csv(folder/'rounds.csv');layers=read_csv(folder/'layers.csv');privacy=read_csv(folder/'privacy.csv')
    if [int(r['round']) for r in rounds]!=list(range(1,T+1)):raise ValueError('Incomplete or duplicate round curve')
    ledger=np.zeros_like(rho,dtype=float);seen=set();by_round={t:[] for t in range(1,T+1)}
    for row in privacy:
        t=int(row['round']);i=int(row['client_slot'])
        if not 1<=t<=T or not 0<=i<n or (t,i) in seen:raise ValueError('Invalid/duplicate privacy row')
        seen.add((t,i));by_round[t].append((i,row))
    if len(seen)!=T*n:raise ValueError('Incomplete privacy ledger')
    group_curves=[];final_eps=None
    for t in range(1,T+1):
        values=[];products=[];qs=[];sigmas=[];radii=[]
        for i,row in sorted(by_round[t]):
            q=number(row['q'],'q',0.);sigma=number(row['sigma'],'sigma',0.);clip=number(row['clip'],'clip',0.)
            if not 0<q<=1 or sigma<=0:raise ValueError('Invalid sampled Gaussian parameters')
            budget_index=np.flatnonzero(np.asarray(cfg['epsilons'])==targets[i])
            if not len(budget_index):raise ValueError('Unknown configured budget group')
            configured_qs=[cfg['q_save'][int(j)] for j in budget_index] if t<cfg['transition_round'] else [cfg['q_spend']]
            if not any(math.isclose(q,v,rel_tol=0,abs_tol=1e-12) for v in configured_qs):raise ValueError('Declared sampling rate mismatch')
            close(number(row['target'],'target'),targets[i],'target assignment')
            ledger[i]+=increment(tuple(orders),q,sigma)
            epsilon=spent(orders,ledger[i],delta)
            close(number(row['epsilon'],'epsilon',0.),epsilon,'per-round epsilon')
            if epsilon>targets[i]+1e-6:raise ValueError('Privacy budget exceeded')
            values.append((targets[i],epsilon));products.append(clip*sigma);qs.append(q);sigmas.append(sigma);radii.append(clip)
        close(products,np.full(n,products[0]),'c_n sigma_n invariant')
        close(products[0],cfg['clip']/np.mean(1/np.asarray(sigmas)),'configured total clip/noise product')
        curve=rounds[t-1]
        if not boolean(curve['privacy_enabled']):raise ValueError('Non-DP curve row')
        close(number(curve['q_bar'],'q_bar',0.),np.mean(qs),'q_bar')
        close(number(curve['noise_var_per_coordinate'],'noise variance',0.),
              (products[0]*cfg['lr']*cfg['rate']/(n*np.mean(qs)))**2,'final physical noise variance',atol=0.)
        es=np.array([e for _,e in values]);final_eps=es
        close(number(curve['epsilon_min'],'epsilon_min',0.),es.min(),'epsilon_min')
        close(number(curve['epsilon_max'],'epsilon_max',0.),es.max(),'epsilon_max')
        for target in sorted(set(targets)):
            v=np.array([e for budget,e in values if budget==target])
            group_curves.append(dict(round=t,target=float(target),clients=len(v),epsilon_min=float(v.min()),
                                     epsilon_mean=float(v.mean()),epsilon_median=float(np.median(v)),
                                     epsilon_max=float(v.max()),utilization_mean=float(v.mean()/target)))
    close(ledger,rho,'recomputed final RDP')
    close(summary['epsilon_n_final'],final_eps,'summary epsilon')
    close(number(summary['budget_violation_rate'],'violation_rate',0.),0.,'violation_rate')
    seen=set();grouped={t:{} for t in range(1,T+1)};previous=dict(registry)
    for row in layers:
        t=int(row['round']);g=row['layer']
        if t not in grouped or g not in registry or (t,g) in seen:raise ValueError('Invalid/duplicate layer row')
        seen.add((t,g));grouped[t][g]=row
    if len(seen)!=T*len(registry):raise ValueError('Incomplete layer curve')
    public_layers=[];freeze_events=set();freeze_first={}
    for t in range(1,T+1):
        counts={};weights={};curve=rounds[t-1]
        variance=number(curve['noise_var_per_coordinate'],'noise variance',0.)
        for g,row in grouped[t].items():
            d=int(row['active_params']);w=number(row['weight'],'weight',0.)
            if not 0<=d<=previous[g] or (d==0 and w!=0):raise ValueError('Reactivated layer or invalid frozen weight')
            if d<previous[g]:freeze_events.add(t);freeze_first.setdefault(g,t)
            if int(row['mask'])!=int(d>0):raise ValueError('Layer mask mismatch')
            counts[g]=d;weights[g]=w
            close(number(row['noise_tax'],'noise_tax',0.),d*variance,'layer noise tax',atol=0.)
            radii=[float(row['clip']) for _,row in by_round[t]]
            for key,stat in [('clip_radius_mean',np.mean),('clip_radius_min',np.min),('clip_radius_max',np.max)]:
                if row.get(key) not in {None,''}:
                    close(number(row[key],key,0.),stat(radii)*math.sqrt(w),key)
            public_layers.append(dict(round=t,layer=g,mask=int(d>0),weight=w,active_params=d,noise_tax=d*variance))
        if not any(counts.values()):raise ValueError('Empty active set')
        close(sum(weights.values()),1.,'active weight sum')
        close(int(curve['active_params']),sum(counts.values()),'active parameter count')
        close(int(curve['active_groups']),sum(d>0 for d in counts.values()),'active group count')
        close(number(curve['active_param_fraction'],'active fraction',0.),sum(counts.values())/sum(registry.values()),'active fraction')
        close(number(curve['noise_energy_expected'],'noise energy',0.),sum(counts.values())*variance,'noise energy',atol=0.)
        previous=counts
    acc=[number(r['test_acc'],'accuracy',0.) for r in rounds]
    if any(v>100 for v in acc):raise ValueError('Accuracy must be a percentage')
    close(summary['final_test_acc'],acc[-1],'final accuracy')
    close(summary['best_test_acc'],max(acc),'best accuracy')
    if 'freeze_first_round' in summary and summary['freeze_first_round']!=freeze_first:raise ValueError('Summary freeze events mismatch')
    feedback=audit_feedback(folder,cfg,registry,grouped,rounds) if 'feedback' in cfg else []
    losses=[number(r['test_loss'],'test loss',0.) for r in rounds];loss=losses[-1]
    if accuracy_target is not None:
        accuracy_target=number(accuracy_target,'accuracy target',0.)
        if accuracy_target>100:raise ValueError('Invalid accuracy target')
    first=next((i+1 for i,a in enumerate(acc) if a>=accuracy_target),None) if accuracy_target is not None else None
    total=lambda key:sum(number(r[key],key,0.) for r in rounds) if all(r.get(key) not in {None,''} for r in rounds) else None
    expected=total('communication_bytes_expected');actual=total('communication_bytes') if include_local_cost else None
    wall=None
    if include_local_cost and (folder/'runtime_local.json').exists():
        wall=number(json.loads((folder/'runtime_local.json').read_text())['wall_time_sec'],'wall time',0.)
    if snapshot()!=initial_hashes:raise ValueError('Run artifacts changed during audit')
    # Never replace actual traffic/FLOPs or private client metrics with estimates.
    safe_columns={'round','test_acc','test_loss','q_bar','epsilon_min','epsilon_max','active_groups','active_params',
                  'active_param_fraction','noise_var_per_coordinate','noise_energy_expected','communication_bytes_expected'}
    if include_local_cost:safe_columns.add('communication_bytes')
    record=dict(path=str(folder.resolve()),seed=summary['seed'],mode=summary['mode'],
                final_test_acc=acc[-1],best_test_acc=max(acc),test_loss=loss,rounds_to_target=first,
                rounds_to_target_status='not_requested' if accuracy_target is None else 'reached' if first is not None else 'not_reached_censored',
                accuracy_target=accuracy_target,communication_expected_MB=None if expected is None else expected/1e6,
                communication_MB=None if actual is None else actual/1e6,wall_time_sec=wall,
                train_FLOPs=None if summary.get('train_flops') is None else number(summary['train_flops'],'FLOPs',0.),
                epsilon_max=float(final_eps.max()),epsilon_min=float(final_eps.min()),delta=delta,
                budget_violation_rate=0.,active_fraction_last=float(rounds[-1]['active_param_fraction']),
                noise_energy_total=sum(float(r['noise_energy_expected']) for r in rounds),
                freeze_event_rounds=sorted(freeze_events),
                audit_scope='saved-artifact consistency, not an independent proof of DP',
                artifacts_sha256=initial_hashes,
                _orders=orders,_rdp=rho,_targets=targets,_eps=final_eps,_config=cfg,
                _split=summary['split_checksum'],_model=(registry,model_hash),
                _rounds=[{k:v for k,v in r.items() if k in safe_columns} for r in rounds],_layers=public_layers,_privacy_groups=group_curves,_feedback=feedback)
    return record


def match_runs(left, right, mode='same_rdp'):
    if mode not in {'same_rdp','same_epsilon','same_cap'}:raise ValueError('Unknown budget matching mode')
    ignored={'mode','allocation','feedback','freeze_rounds','mask_seed','validation_protocol','diagnostic_freezing','diagnostic_rng_path','static_order'}
    a={k:v for k,v in left['_config'].items() if k not in ignored}
    b={k:v for k,v in right['_config'].items() if k not in ignored}
    if a!=b or left['_model']!=right['_model'] or left['_split']!=right['_split'] or left['seed']!=right['seed']:
        raise ValueError('Unmatched data/model/partition/training budget')
    if left['delta']!=right['delta']:raise ValueError('Unmatched delta')
    close(left['_targets'],right['_targets'],'client budget caps')
    if mode=='same_rdp':
        if not np.array_equal(left['_orders'],right['_orders']):raise ValueError('Unmatched RDP orders')
        try:np.testing.assert_allclose(left['_rdp'],right['_rdp'],rtol=0,atol=1e-12)
        except AssertionError as exc:raise ValueError('Unmatched actual RDP') from exc
    elif mode=='same_epsilon':close(left['_eps'],right['_eps'],'actual epsilon')
