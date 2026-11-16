"""Validated statistical and decision metrics. No training or data loading."""
import itertools
import math
import numpy as np
from scipy.stats import t as student_t, spearmanr


def finite_vector(values, name):
    a=np.asarray(values,dtype=float)
    if a.ndim!=1 or not np.isfinite(a).all():raise ValueError(f'Invalid {name}')
    return a


def mean_interval(values):
    a=finite_vector(values,'observations');n=len(a)
    if not n:return dict(n=0,mean=None,std=None,ci95=None)
    mean=float(a.mean());std=float(a.std(ddof=1)) if n>1 else None
    half=float(student_t.ppf(.975,n-1))*std/math.sqrt(n) if n>1 else None
    return dict(n=n,mean=mean,std=std,ci95=None if half is None else [mean-half,mean+half])


def paired_comparison(left, right):
    a=finite_vector(left,'left');b=finite_vector(right,'right')
    if a.shape!=b.shape:raise ValueError('Paired samples must match')
    gaps=a-b;result=mean_interval(gaps)
    p=None
    if 0<len(gaps)<=16:
        observed=abs(float(gaps.mean()))
        p=sum(abs(float(np.mean(gaps*np.asarray(s))))>=observed-1e-12
              for s in itertools.product([-1,1],repeat=len(gaps)))/2**len(gaps)
    return dict(**result,gaps=gaps.tolist(),exact_two_sided_sign_flip_p=p,
                p_status='exact' if p is not None else 'unavailable_empty_or_more_than_16_pairs')


def holm_adjust(pvalues):
    if any(p is None for p in pvalues):return [None]*len(pvalues)
    p=finite_vector(pvalues,'p-values')
    if ((p<0)|(p>1)).any():raise ValueError('Invalid p-values')
    result=[None]*len(p);previous=0.
    for rank,index in enumerate(np.argsort(p)):
        previous=max(previous,min(1.,float(p[index])*(len(p)-rank)));result[int(index)]=previous
    return result


def decision_metrics(predicted, truth, eligible=None):
    p=list(predicted);y=list(truth)
    e=[True]*len(y) if eligible is None else list(eligible)
    if len(p)!=len(y) or len(e)!=len(y):raise ValueError('Decision vectors must match')
    counts=dict(tp=0,fp=0,fn=0,tn=0);excluded=0
    for pred,label,allowed in zip(p,y,e):
        if not isinstance(allowed,(bool,np.bool_)):raise ValueError('Eligibility must be boolean')
        if not allowed or label is None or pred is None:excluded+=1;continue
        if not isinstance(pred,(bool,np.bool_)) or not isinstance(label,(bool,np.bool_)):raise ValueError('Decision labels must be boolean or null')
        counts['tp' if pred and label else 'fp' if pred else 'fn' if label else 'tn']+=1
    tp,fp,fn,tn=[counts[k] for k in ['tp','fp','fn','tn']]
    ratio=lambda a,b:a/b if b else None
    return dict(**counts,evaluable=tp+fp+fn+tn,excluded=excluded,
                false_freeze_rate=ratio(fp,fp+tn),missed_freeze_rate=ratio(fn,fn+tp),
                precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),f1=ratio(2*tp,2*tp+fp+fn))


def estimator_metrics(estimated, truth, *, target):
    if target!='sampled_clipped_aggregate_energy':raise ValueError('Wrong estimator target')
    a=finite_vector(estimated,'estimated');b=finite_vector(truth,'truth')
    if a.shape!=b.shape:raise ValueError('Estimator samples must match')
    if (b<0).any():raise ValueError('True energy must be nonnegative')
    if not len(a):return dict(n=0,bias=None,rmse=None,spearman=None)
    diff=a-b
    rank=None if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0 else float(spearmanr(a,b).statistic)
    return dict(n=len(a),bias=float(diff.mean()),rmse=float(np.sqrt(np.mean(diff**2))),spearman=rank)


def client_utility(rows, *, offline_only=False):
    if not offline_only:raise ValueError('Local utility requires isolated offline diagnostics')
    grouped={};seen=set()
    for row in rows:
        slot=row['client_slot'];target=float(row['target']);acc=float(row['accuracy'])
        if slot in seen or not math.isfinite(target) or target<=0 or not math.isfinite(acc) or not 0<=acc<=100:
            raise ValueError('Invalid or duplicate client utility')
        seen.add(slot);grouped.setdefault(target,[]).append(acc)
    return [dict(target=t,clients=len(v),mean_accuracy=float(np.mean(v)),worst_accuracy=min(v),
                 p10=float(np.quantile(v,.1)),p50=float(np.quantile(v,.5)),p90=float(np.quantile(v,.9)))
            for t,v in sorted(grouped.items())]
