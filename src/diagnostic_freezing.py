"""D1/D2/D3 public-benchmark diagnostics; no future-utility safety certificate."""
import copy
import math
import torch
from scipy.optimize import brentq
from scipy.stats import ncx2
from .dp_feedback_freezer import validate_feedback


def energy_ucb(energy, dimension, variance, alpha):
    """Invert F_ncx2(E/v; d, lambda_U)=alpha, then return v*lambda_U.

    Conditional Gaussian model only: U is the clipped, sampled aggregate, not
    the full-client/unclipped gradient. Boundary truncation at lambda=0 is valid.
    alpha is LOWER-tail probability, not 1-alpha. Never use interval() on Y.
    """
    if type(dimension) is not int or dimension<1:raise ValueError('Invalid dimension')
    if not all(math.isfinite(x) for x in (energy,variance,alpha)) or energy<0 or variance<0 or not 0<alpha<1:
        raise ValueError('Invalid UCB input')
    if variance==0:return float(energy)
    x=energy/variance
    if not math.isfinite(x):raise ValueError('Unrepresentable normalized energy')
    if ncx2.cdf(x,dimension,0.)<=alpha:return 0.
    upper=max(1.,x-dimension,math.sqrt(2.*dimension))
    for _ in range(100):
        if ncx2.cdf(x,dimension,upper)<=alpha:break
        upper*=2
    else:raise RuntimeError('Could not bracket noncentrality upper bound')
    bound=brentq(lambda noncentrality:ncx2.cdf(x,dimension,noncentrality)-alpha,
                 0.,upper,xtol=1e-10,rtol=1e-12)
    return float(bound*variance)


def validate_diagnostic(policy, feedback, rounds):
    p=copy.deepcopy(policy)
    if not isinstance(p,dict) or p.get('kind') not in {'replay','delayed','confidence'}:raise ValueError('Unknown diagnostic policy')
    if type(rounds) is not int or rounds<1:raise ValueError('Invalid diagnostic horizon')
    required={'kind','min_freeze_round','min_active_fraction','freeze_order','protected_groups'}
    if p['kind']=='replay':required={'kind','replay_rounds'}
    if p['kind']=='confidence':required|={'alpha_total','drift_window','max_ratio_range','drift_margin_ratio'}
    if set(p)!=required:raise ValueError('Unknown/missing diagnostic fields')
    validate_feedback(feedback)
    if p['kind']=='replay':
        events=p['replay_rounds']
        if not isinstance(events,list) or any(type(t) is not int or not 1<=t<=rounds for t in events) or sorted(set(events))!=events:
            raise ValueError('Invalid replay rounds')
    else:
        if type(p['min_freeze_round']) is not int or not 1<=p['min_freeze_round']<=rounds:raise ValueError('Invalid minimum freeze round')
        if isinstance(p['min_active_fraction'],bool) or not math.isfinite(p['min_active_fraction']) or not 0<p['min_active_fraction']<=1:raise ValueError('Invalid active fraction')
        if p['freeze_order'] not in {'any','bottom_up'}:raise ValueError('Invalid diagnostic order')
        if not isinstance(p['protected_groups'],list) or len(set(p['protected_groups']))!=len(p['protected_groups']) or any(not isinstance(g,str) for g in p['protected_groups']):raise ValueError('Invalid protected groups')
        if p['kind']=='confidence':
            if not 0<p['alpha_total']<1 or not math.isfinite(p['alpha_total']):raise ValueError('Invalid alpha')
            if type(p['drift_window']) is not int or p['drift_window']<2:raise ValueError('Invalid drift window')
            for key in ['max_ratio_range','drift_margin_ratio']:
                if isinstance(p[key],bool) or not math.isfinite(p[key]) or p[key]<0:raise ValueError('Invalid drift guard')
    return p


class DiagnosticFreezer:
    def __init__(self,names,dimensions,feedback,policy,rounds):
        self.names=list(names);self.dimensions=dict(dimensions);self.config=validate_feedback(feedback)
        self.policy=validate_diagnostic(policy,feedback,rounds);self.rounds=rounds
        if set(names)!=set(dimensions) or any(type(d) is not int or d<=0 for d in dimensions.values()):raise ValueError('Invalid layer registry')
        if self.config['min_active_groups']>len(names):raise ValueError('Invalid protected group count')
        self.protected=set(names[-self.config['min_active_groups']:])|set(policy.get('protected_groups',[]))
        if not self.protected<=set(names):raise ValueError('Unknown protected group')
        if policy['kind']=='replay' and len(policy['replay_rounds'])>=len(names):raise ValueError('Replay freezes all layers')
        self.active=dict.fromkeys(names,True);self.low=dict.fromkeys(names,0)
        self.ema={};self.history={g:[] for g in names};self.pending=set();self.freeze_round={}
        self.last_decided=0;self.last_observed=0;self.last_freeze_round=None
        self.alpha=policy.get('alpha_total',.05)/(rounds*len(names))

    def decide(self,t):
        if t==self.last_decided:return self.active.copy()
        if type(t) is not int or t!=self.last_decided+1 or self.last_observed!=t-1 or t>self.rounds:raise ValueError('Sequential historical rounds required')
        if self.policy['kind']=='replay':
            expected={self.names[i] for i,event in enumerate(self.policy['replay_rounds']) if t>=event}
            self.pending={g for g in expected if self.active[g]}
        for g in self.pending:
            self.active[g]=False;self.low[g]=0;self.freeze_round[g]=t;self.last_freeze_round=t
        self.pending.clear();self.last_decided=t
        return self.active.copy()

    def observe(self,t,delta,variance):
        enabled={g for g in self.names if self.active[g]}
        if set(delta)!=enabled:raise ValueError('Active DP layer registry mismatch')
        stats={}
        for g,tensors in delta.items():
            if any(not isinstance(v,torch.Tensor) or not torch.isfinite(v).all() for v in tensors):raise ValueError('Invalid DP update')
            stats[g]=(sum(v.numel() for v in tensors),math.fsum(v.detach().double().square().sum().item() for v in tensors))
        return self.observe_statistics(t,stats,variance)

    def observe_statistics(self,t,stats,variance):
        """Sufficient statistics entry also used for read-only transcript replay."""
        if type(t) is not int or t!=self.last_decided or self.last_observed!=t-1:raise ValueError('Observe once after deciding')
        if not math.isfinite(variance) or variance<0:raise ValueError('Invalid variance')
        enabled={g for g in self.names if self.active[g]}
        if set(stats)!=enabled or any(d!=self.dimensions[g] or not math.isfinite(e) or e<0 for g,(d,e) in stats.items()):raise ValueError('Invalid DP energy registry')
        cfg=self.config;p=self.policy;rows=[];candidates=[]
        cooling=self.last_freeze_round is not None and t+1-self.last_freeze_round<=cfg['cooldown_rounds']
        next_bottom=next((g for g in self.names if self.active[g] and g not in self.protected),None)
        active_dims=sum(self.dimensions[g] for g in enabled);total_dims=sum(self.dimensions.values())
        for g in self.names:
            row=dict(round=t,layer=g,observed=self.active[g],dimension=None,released_energy=None,
                     noise_variance=None,noise_tax=None,debiased_energy=None,nonnegative_score=None,
                     score=None,threshold=None,low_count=None,protected=g in self.protected,
                     cooldown=cooling,freeze_next_round=False,ucb_energy=None,alpha_per_test=None,
                     drift_stable=None,capacity_allowed=None,order_allowed=None,earliest_allowed=None)
            rows.append(row)
            if not self.active[g]:continue
            d,energy=stats[g];tax=d*variance;point=max(0.,energy-tax)
            base=energy if cfg['feedback_score']=='raw' else point
            decay=cfg['ema_decay'] if cfg['feedback_score']=='ema' else 0.
            smooth=decay*self.ema.get(g,base)+(1-decay)*base
            self.ema[g]=smooth;ucb=None
            if p['kind']=='confidence':
                ucb=energy_ucb(energy,d,variance,self.alpha)
                smooth=ucb+p['drift_margin_ratio']*tax
            threshold=cfg['threshold_ratio']*tax
            self.low[g]=self.low[g]+1 if t>=cfg['warmup_rounds'] and smooth<threshold else 0
            ratio=smooth/tax if tax>0 else float('inf')
            self.history[g].append(ratio if math.isfinite(ratio) else None)
            stable=True
            if p['kind']=='confidence':
                recent=self.history[g][-p['drift_window']:]
                stable=len(recent)==p['drift_window'] and None not in recent and max(recent)-min(recent)<=p['max_ratio_range']
            capacity=p['kind']=='replay' or active_dims-d>=p['min_active_fraction']*total_dims
            ordered=p.get('freeze_order','any')=='any' or g==next_bottom
            earliest=p['kind']!='replay' and p['min_freeze_round']<=t+1<=self.rounds
            row.update(dimension=d,released_energy=energy,noise_variance=variance,noise_tax=tax,
                       debiased_energy=energy-tax,nonnegative_score=base,score=smooth,threshold=threshold,
                       low_count=self.low[g],ucb_energy=ucb,alpha_per_test=self.alpha if ucb is not None else None,
                       drift_stable=stable,capacity_allowed=capacity,order_allowed=ordered,earliest_allowed=earliest)
            if p['kind']!='replay' and self.low[g]>=cfg['patience'] and g not in self.protected and not cooling and stable and capacity and ordered and earliest:
                candidates.append(g)
        if p['kind']=='replay':
            self.pending={self.names[i] for i,event in enumerate(p['replay_rounds']) if event==t+1}
        else:
            indexed={r['layer']:r for r in rows}
            candidates.sort(key=lambda g:(indexed[g]['score']/indexed[g]['noise_tax'],self.names.index(g)))
            self.pending=set(candidates[:min(cfg['max_freeze_per_round'],sum(self.active.values())-cfg['min_active_groups'])])
        for row in rows:row['freeze_next_round']=row['layer'] in self.pending
        self.last_observed=t
        return rows

    def state_dict(self):
        return copy.deepcopy(dict(names=self.names,config=self.config,policy=self.policy,rounds=self.rounds,
                                  active=self.active,low=self.low,ema=self.ema,history=self.history,pending=sorted(self.pending),
                                  freeze_round=self.freeze_round,last_decided=self.last_decided,last_observed=self.last_observed,
                                  last_freeze_round=self.last_freeze_round))


def audit_diagnostic_feedback(folder,cfg,registry,grouped,rounds):
    """Recompute published energies and replay policy; accountants audited separately."""
    import csv,json
    from .p4_audit import close,boolean
    with (folder/'feedback.csv').open() as f:saved=list(csv.DictReader(f))
    indexed={(int(r['round']),r['layer']):r for r in saved}
    if len(indexed)!=len(saved) or len(saved)!=len(registry)*cfg['rounds']:raise ValueError('Incomplete diagnostic feedback')
    c=DiagnosticFreezer(list(registry),registry,cfg['feedback'],cfg['diagnostic_freezing'],cfg['rounds']);safe=[]
    for t in range(1,cfg['rounds']+1):
        active=c.decide(t)
        for g in registry:
            if int(grouped[t][g]['active_params'])!=(registry[g] if active[g] else 0):raise ValueError('Diagnostic mask mismatch')
        stats={g:(registry[g],float(grouped[t][g]['released_dp_update_energy'])) for g in registry if active[g]}
        expected=c.observe_statistics(t,stats,float(rounds[t-1]['noise_var_per_coordinate']))
        for row in expected:
            stored=indexed[(t,row['layer'])]
            for key,value in row.items():
                if value is None:
                    if stored[key]!='':raise ValueError('Missing diagnostic observation must be empty')
                elif isinstance(value,bool):
                    if boolean(stored[key])!=value:raise ValueError('Diagnostic decision flag mismatch')
                elif isinstance(value,(int,float)):close(float(stored[key]),value,'diagnostic '+key)
                elif stored[key]!=value:raise ValueError('Diagnostic registry mismatch')
            safe.append(row)
    if json.loads((folder/'feedback_state.json').read_text())!=c.state_dict():raise ValueError('Diagnostic final state mismatch')
    return safe
