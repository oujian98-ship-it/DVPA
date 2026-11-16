"""Irreversible heuristic freezing from completed, released DP rounds only."""
import copy
import math
from numbers import Real
import torch


DEFAULTS = dict(feedback_score='ema', ema_decay=.9, patience=3, warmup_rounds=3,
                threshold_ratio=1., min_active_groups=1, cooldown_rounds=1,
                max_freeze_per_round=1)


def validate_feedback(config):
    cfg={**DEFAULTS, **config}
    if cfg['feedback_score'] not in {'raw','debiased','ema'}:
        raise ValueError('Unsupported feedback score; conservative/oracle are not DP heuristics here')
    for key, minimum in [('patience',1),('warmup_rounds',1),('min_active_groups',1),
                         ('cooldown_rounds',0),('max_freeze_per_round',1)]:
        if type(cfg[key]) is not int or cfg[key]<minimum:raise ValueError(f'Invalid {key}')
    decay=cfg['ema_decay'];ratio=cfg['threshold_ratio']
    if isinstance(decay,bool) or not isinstance(decay,Real) or not math.isfinite(decay) or not 0<=decay<1:raise ValueError('Invalid ema_decay')
    if isinstance(ratio,bool) or not isinstance(ratio,Real) or not math.isfinite(ratio) or ratio<0:raise ValueError('Invalid threshold_ratio')
    return cfg


class FeedbackFreezer:
    """decide(t) -> observe(t, released_update, final_coordinate_variance).

    Nonnegative debiasing/EMA is heuristic, not an unbiased estimator or safety
    certificate. The deepest min_active_groups blocks are protected throughout.
    """
    def __init__(self, names, config, seed=42):
        self.names=list(names)
        if not self.names or len(set(self.names))!=len(self.names):raise ValueError('Invalid layer names')
        self.config=validate_feedback(copy.deepcopy(config))
        if self.config['min_active_groups']>len(self.names):raise ValueError('Too many protected groups')
        self.order=list(self.names)
        self.protected=set(self.names[-self.config['min_active_groups']:])
        self.active=dict.fromkeys(self.names,True)
        self.ema={};self.low=dict.fromkeys(self.names,0);self.dimensions={}
        self.pending=set();self.freeze_round={}
        self.last_decided=0;self.last_observed=0;self.last_freeze_round=None

    def decide(self, t):
        if type(t) is not int or t<1:raise ValueError('Invalid decision round')
        if t==self.last_decided:return self.active.copy()
        if t!=self.last_decided+1 or self.last_observed!=t-1:
            raise ValueError('Decisions require sequential completed historical observations')
        for g in self.names:
            if g in self.pending:
                self.active[g]=False;self.low[g]=0;self.freeze_round[g]=t
                self.last_freeze_round=t
        self.pending.clear();self.last_decided=t
        return self.active.copy()

    def observe(self, t, delta, variance):
        if type(t) is not int or t!=self.last_decided or t<=self.last_observed:
            raise ValueError('Observe exactly once, after deciding the same round')
        if isinstance(variance,bool) or not isinstance(variance,Real) or not math.isfinite(variance) or variance<0:raise ValueError('Invalid released noise variance')
        enabled={g for g in self.names if self.active[g]}
        if set(delta)-set(self.names) or not enabled<=set(delta):
            raise ValueError('Missing active or unknown released layer')
        # Validate everything before mutating counters or consuming the round.
        statistics={}
        for g in enabled:
            tensors=delta[g] if isinstance(delta[g],(list,tuple)) else (delta[g],)
            if not tensors or any(not isinstance(v,torch.Tensor) or not torch.isfinite(v).all() for v in tensors):
                raise ValueError('Released updates must contain finite tensors')
            d=sum(v.numel() for v in tensors)
            energy=math.fsum(v.detach().double().square().sum().item() for v in tensors)
            if d<=0 or not math.isfinite(energy) or not math.isfinite(d*variance):
                raise ValueError('Invalid released energy or dimensions')
            if not math.isfinite(self.config['threshold_ratio']*(d*variance)):
                raise ValueError('Threshold overflowed')
            if g in self.dimensions and self.dimensions[g]!=d:raise ValueError('Active layer dimensions changed')
            statistics[g]=(d,energy)
        cfg=self.config;pending=[];rows=[]
        cooling=self.last_freeze_round is not None and t+1-self.last_freeze_round<=cfg['cooldown_rounds']
        for g in self.names:
            if not self.active[g]:
                rows.append(dict(round=t,layer=g,observed=False,dimension=None,released_energy=None,
                                 noise_variance=None,noise_tax=None,debiased_energy=None,
                                 nonnegative_score=None,score=None,threshold=None,low_count=None,
                                 protected=g in self.protected,cooldown=cooling,freeze_next_round=False))
                continue  # No inference from frozen zeros; EMA/counters remain untouched.
            d,energy=statistics[g];tax=d*variance;debiased=energy-tax
            score=energy if cfg['feedback_score']=='raw' else max(0.,debiased)
            decay=cfg['ema_decay'] if cfg['feedback_score']=='ema' else 0.
            smooth=decay*self.ema.get(g,score)+(1-decay)*score
            threshold=cfg['threshold_ratio']*tax
            self.ema[g]=smooth;self.dimensions[g]=d
            self.low[g]=self.low[g]+1 if t>=cfg['warmup_rounds'] and smooth<threshold else 0
            if self.low[g]>=cfg['patience'] and g not in self.protected and not cooling:
                pending.append(g)
            rows.append(dict(round=t,layer=g,observed=True,dimension=d,released_energy=energy,
                             noise_variance=variance,noise_tax=tax,debiased_energy=debiased,
                             nonnegative_score=score,score=smooth,threshold=threshold,low_count=self.low[g],
                             protected=g in self.protected,cooldown=cooling,freeze_next_round=False))
        # Noise-normalized scores; deterministic registry order breaks ties.
        by_name={r['layer']:r for r in rows}
        pending.sort(key=lambda g:(by_name[g]['score']/by_name[g]['noise_tax'],self.names.index(g)))
        self.pending=set(pending[:min(cfg['max_freeze_per_round'],sum(self.active.values())-cfg['min_active_groups'])])
        for row in rows:row['freeze_next_round']=row['layer'] in self.pending
        self.last_observed=t
        return rows

    def state_dict(self):
        """JSON-safe DP postprocessing state; no raw client values/RNG seeds."""
        return copy.deepcopy(dict(names=self.names,config=self.config,active=self.active,ema=self.ema,
                                  low=self.low,dimensions=self.dimensions,pending=sorted(self.pending),
                                  freeze_round=self.freeze_round,last_decided=self.last_decided,
                                  last_observed=self.last_observed,last_freeze_round=self.last_freeze_round))
