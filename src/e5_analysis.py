"""Complete-block factorial contrasts; no missing-cell or best-seed selection."""
import math
from .p4_metrics import mean_interval,paired_comparison,holm_adjust

METHODS=('STATIC_EQUAL','STATIC_4A','DYNAMIC_EQUAL','DYNAMIC_4A')
CONTRASTS={
    '4A_under_static':(-1,1,0,0),
    'dynamic_under_equal':(-1,0,1,0),
    'dynamic_under_4A':(0,-1,0,1),
    'joint_vs_static_equal':(-1,0,0,1),
    'interaction':(1,-1,-1,1)}


def factorial_summary(records,seeds):
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Invalid registered seed block')
    indexed={}
    for r in records:
        key=(r['seed'],r['method'])
        if key in indexed or key[0] not in seeds or key[1] not in METHODS:raise ValueError('Duplicate/unregistered factorial cell')
        for metric in ['best_test_acc','final_test_acc']:
            v=r[metric]
            if not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=100:raise ValueError('Invalid accuracy')
        indexed[key]=r
    if set(indexed)!={(s,m) for s in seeds for m in METHODS}:raise ValueError('Incomplete factorial block')
    result=dict(training_seeds=seeds,per_run=records,main_table=[],contrasts={})
    for method in METHODS:
        result['main_table'].append(dict(method=method,
            best=mean_interval([indexed[(s,method)]['best_test_acc'] for s in seeds]),
            final=mean_interval([indexed[(s,method)]['final_test_acc'] for s in seeds])))
    for metric in ['best_test_acc','final_test_acc']:
        rows=[]
        for name,coefficients in CONTRASTS.items():
            gaps=[sum(c*indexed[(s,m)][metric] for c,m in zip(coefficients,METHODS)) for s in seeds]
            rows.append(dict(contrast=name,metric=metric,unit='percentage points',
                             **paired_comparison(gaps,[0.]*len(gaps))))
        adjusted=holm_adjust([r['exact_two_sided_sign_flip_p'] for r in rows])
        for row,p in zip(rows,adjusted):row['holm_adjusted_p_within_metric']=p
        result['contrasts'][metric]=rows
    result['scope']='Descriptive three-seed factorial comparison; peak is test-curve maximum. Five contrasts per metric Holm-adjusted separately. No future-utility guarantee or joint-release privacy certification.'
    return result
