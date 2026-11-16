"""Isolated offline analysis; never import this module from training controllers."""
import math
import numpy as np
from .p4_metrics import decision_metrics, estimator_metrics


def conditional_decomposition(reference, masked_local, clipped, q, active, noise_variance, *, offline_only=False):
    if not offline_only:raise ValueError('Raw oracle vectors require offline_only=True')
    u,h,v=[np.asarray(a,dtype=float) for a in [reference,masked_local,clipped]]
    q=np.asarray(q,dtype=float);mask=np.asarray(active)
    if u.ndim!=2 or min(u.shape)==0 or u.shape!=h.shape or u.shape!=v.shape:
        raise ValueError('Oracle update matrices must match N x D')
    if q.shape!=(len(u),) or mask.shape!=(u.shape[1],) or mask.dtype!=bool:
        raise ValueError('Invalid probability or coordinate mask shape')
    if not all(np.isfinite(a).all() for a in [u,h,v,q]) or (q<0).any() or (q>1).any() or q.sum()<=0:
        raise ValueError('Invalid finite updates/probabilities')
    if not math.isfinite(noise_variance) or noise_variance<0 or not mask.any():raise ValueError('Invalid noise or active set')
    if np.any(h[:,~mask]!=0) or np.any(v[:,~mask]!=0):raise ValueError('Frozen coordinates must be zero')
    # These bias vectors telescope. Their squared norms alone do NOT add up.
    freeze=u.mean(0)-h.mean(0);clip=h.mean(0)-v.mean(0)
    sampling_mean=v.mean(0)-(q[:,None]*v).sum(0)/q.sum()
    bias=freeze+clip+sampling_mean
    norm=lambda x:float(x@x)
    cross=2*float(freeze@clip+freeze@sampling_mean+clip@sampling_mean)
    sampling=float(np.sum(q*(1-q)*np.sum(v*v,axis=1))/q.sum()**2)
    noise=float(mask.sum()*noise_variance)
    return dict(scope='private offline conditional single-round analysis',
                freeze_and_local_dynamics_bias_sq=norm(freeze),clipping_bias_sq=norm(clip),
                sampling_mean_bias_sq=norm(sampling_mean),bias_cross_terms=cross,
                combined_bias_sq=norm(bias),sampling_variance=sampling,dp_noise_variance=noise,
                conditional_mse=norm(bias)+sampling+noise)


def analyze_oracle(feedback, oracle, horizon, *, offline_only=False):
    """Oracle rows use EFFECTIVE decision_round, never observation round t.

    Required oracle: decision_round, layer, mse_active, mse_freeze. Optional:
    eligible, estimated_energy, noiseless_energy. Null MSE has no truth label.
    """
    if not offline_only:raise ValueError('Oracle analysis requires offline_only=True')
    if type(horizon) is not int or horizon<1:raise ValueError('Invalid horizon')
    history={}
    for r in feedback:
        key=(r['round'],r['layer'])
        if type(key[0]) is not int or not 1<=key[0]<=horizon or key in history:raise ValueError('Invalid feedback round/key')
        if type(r['observed']) is not bool or type(r['freeze_next_round']) is not bool:raise ValueError('Invalid feedback flags')
        if not r['observed'] and r['freeze_next_round']:raise ValueError('Frozen groups cannot request new freezing')
        history[key]=r
    for (t,g),row in history.items():
        following=history.get((t+1,g))
        if following and ((row['freeze_next_round'] and following['observed']) or (not row['observed'] and following['observed'])):
            raise ValueError('Feedback history violates irreversible next-round freezing')
    seen=set();joined=[];pred=[];truth=[];eligible=[];est=[];energy=[];strata={};rank_rounds={}
    for row in oracle:
        t=row['decision_round'];g=row['layer'];key=(t,g)
        if type(t) is not int or not 1<=t<=horizon or key in seen:raise ValueError('Invalid oracle round/key')
        seen.add(key);a=row['mse_active'];b=row['mse_freeze']
        if (a is None)!=(b is None):raise ValueError('Both MSE values must be present or null')
        if a is not None and (not all(math.isfinite(v) and v>=0 for v in [a,b])):raise ValueError('Invalid oracle MSE')
        y=None if a is None else a>b
        previous=history.get((t-1,g))
        p=previous['freeze_next_round'] if previous and previous['observed'] else None
        allowed=row.get('eligible',True)
        if type(allowed) is not bool:raise ValueError('Eligibility must be boolean')
        pred.append(p);truth.append(y);eligible.append(allowed)
        joined.append(dict(decision_round=t,layer=g,truth=y,predicted=p,eligible=allowed))
        if row.get('estimated_energy') is not None or row.get('noiseless_energy') is not None:
            if row.get('estimator_target')!='sampled_clipped_aggregate_energy':raise ValueError('Wrong oracle estimator target')
            est.append(row['estimated_energy']);energy.append(row['noiseless_energy'])
            epsilon=row.get('epsilon');dimension=row.get('dimension')
            if epsilon is not None and (not math.isfinite(epsilon) or epsilon<0):raise ValueError('Invalid estimator epsilon metadata')
            if dimension is not None and (type(dimension) is not int or dimension<1):raise ValueError('Invalid estimator dimension metadata')
            pairs=strata.setdefault((epsilon,dimension),([],[]));pairs[0].append(est[-1]);pairs[1].append(energy[-1])
            pairs=rank_rounds.setdefault(t,([],[]));pairs[0].append(est[-1]);pairs[1].append(energy[-1])
    delays=[]
    for g in sorted({r['layer'] for r in joined}):
        labels={r['decision_round']:r['truth'] for r in joined if r['layer']==g and r['eligible']}
        first=min((t for t,y in labels.items() if y is True),default=None)
        applied=min((t+1 for (t,layer),r in history.items() if layer==g and r['freeze_next_round'] and
                     t+1<=horizon and (t+1,g) in history and not history[(t+1,g)]['observed']),default=None)
        delay=applied-first if first is not None and applied is not None and applied>=first and labels.get(applied) is True else None
        status=('no_oracle_event' if first is None else 'observed' if delay is not None else
                'not_frozen_by_horizon' if applied is None else 'premature_or_unverified_freeze')
        delays.append(dict(layer=g,oracle_first_round=first,applied_freeze_round=applied,
                           delay=delay,censored=first is not None and delay is None,status=status))
    changes=[]
    for (t,g),r in history.items():
        prev=history.get((t-1,g))
        if r['observed'] and prev and prev['observed'] and r.get('score') is not None and prev.get('score') is not None:
            pair=[r['score'],prev['score']]
            if not all(math.isfinite(v) for v in pair):raise ValueError('Invalid DP scores')
            changes.append(abs(pair[0]-pair[1]))
    return dict(scope='private offline oracle analysis; not a DP method/output',
                decisions=decision_metrics(pred,truth,eligible),delays=delays,
                estimator=estimator_metrics(est,energy,target='sampled_clipped_aggregate_energy'),
                estimator_strata=[dict(epsilon=k[0],dimension=k[1],**estimator_metrics(v[0],v[1],target='sampled_clipped_aggregate_energy')) for k,v in strata.items()],
                layer_rank_by_round=[dict(decision_round=t,spearman=estimator_metrics(v[0],v[1],target='sampled_clipped_aggregate_energy')['spearman']) for t,v in sorted(rank_rounds.items())],
                mean_abs_consecutive_score_change=float(np.mean(changes)) if changes else None,
                oracle_gap=None,oracle_gap_status='requires a separately trained resource-matched oracle baseline',
                joined=joined)


def oracle_gap(oracle, method, *, offline_only=False):
    if not offline_only:raise ValueError('Oracle baseline comparison belongs offline')
    required={'dataset','model','split_checksum','seed','rounds','clients','local_epochs','sampling_policy'}
    for item in [oracle,method]:
        if not required<=set(item.get('resources',{})):raise ValueError('Missing oracle resource metadata')
        if not math.isfinite(item['final_test_acc']) or not 0<=item['final_test_acc']<=100:raise ValueError('Invalid oracle/method accuracy')
    if oracle['resources']!=method['resources']:raise ValueError('Oracle training resources must match')
    loss=None
    if oracle.get('test_loss') is not None and method.get('test_loss') is not None:
        if not all(math.isfinite(x['test_loss']) and x['test_loss']>=0 for x in [oracle,method]):raise ValueError('Invalid test loss')
        loss=oracle['test_loss']-method['test_loss']
    return dict(accuracy_gap_pp=oracle['final_test_acc']-method['final_test_acc'],oracle_minus_method_test_loss=loss,
                scope='isolated oracle comparison; resource equality is based on supplied metadata, not independently certified')
