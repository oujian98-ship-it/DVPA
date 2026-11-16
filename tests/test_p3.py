import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from src.dp_feedback_freezer import FeedbackFreezer, validate_feedback
from src.p3_freezing import AdaptiveMask, validate_feedback_run, feedback_preview


def controller(names=('a','b','c'), **config):
    return FeedbackFreezer(names,dict(feedback_score='debiased',warmup_rounds=1,
                                   patience=1,cooldown_rounds=0,**config))


def zeros(f):
    return {g:torch.zeros(2) for g in f.names if f.active[g]}


def test_feedback_applies_next_round_only_and_deepest_groups_are_protected():
    f=controller();assert all(f.decide(1).values())
    rows=f.observe(1,zeros(f),1.)
    assert f.active['a'] and f.pending=={'a'}
    assert all(f.decide(1).values())  # Calling again cannot use current-round evidence.
    assert rows[0]['freeze_next_round'] and not rows[-1]['freeze_next_round']
    assert f.decide(2)==dict(a=False,b=True,c=True)
    frozen_ema=f.ema['a'];rows=f.observe(2,zeros(f),1.)
    assert rows[0]['observed'] is False and rows[0]['score'] is None
    assert f.ema['a']==frozen_ema
    assert f.decide(3)==dict(a=False,b=False,c=True)
    for t in range(3,7):
        f.observe(t,zeros(f),1.);assert f.decide(t+1)==dict(a=False,b=False,c=True)


@pytest.mark.parametrize('action',['observe_first','skip_decision','unobserved_advance','duplicate_observe','stale_observe','rewind'])
def test_round_order_is_enforced(action):
    f=controller()
    with pytest.raises(ValueError):
        if action=='observe_first':f.observe(1,zeros(f),1.)
        elif action=='skip_decision':f.decide(2)
        else:
            f.decide(1)
            if action=='unobserved_advance':f.decide(2)
            else:
                f.observe(1,zeros(f),1.)
                if action=='duplicate_observe':f.observe(1,zeros(f),1.)
                else:
                    f.decide(2)
                    if action=='stale_observe':f.observe(1,zeros(f),1.)
                    else:f.decide(1)


@pytest.mark.parametrize('mode,score,pending',[('raw',2.,False),('debiased',0.,True),('ema',0.,True)])
def test_scores_use_final_coordinate_variance(mode,score,pending):
    f=FeedbackFreezer(['a','b'],dict(feedback_score=mode,warmup_rounds=1,patience=1))
    f.decide(1);rows=f.observe(1,dict(a=torch.ones(2),b=torch.ones(3)),1.)
    assert rows[0]['noise_tax']==2.
    assert rows[0]['debiased_energy']==0.
    assert rows[0]['score']==score
    assert ('a' in f.pending)==pending


def test_ema_starts_at_first_observed_score_not_zero():
    f=FeedbackFreezer(['a','b'],dict(feedback_score='ema',ema_decay=.5,patience=1,warmup_rounds=1))
    f.decide(1);f.observe(1,dict(a=torch.tensor([2.,2.]),b=torch.ones(2)),1.)
    assert f.ema['a']==6. and not f.pending
    f.decide(2);f.observe(2,zeros(f),1.);assert f.ema['a']==3. and not f.pending
    f.decide(3);f.observe(3,zeros(f),1.);assert f.ema['a']==1.5 and f.pending=={'a'}


def test_warmup_then_consecutive_patience_resets_on_high_score():
    f=FeedbackFreezer(['a','b'],dict(feedback_score='debiased',warmup_rounds=3,patience=2))
    for t in range(1,4):
        f.decide(t);f.observe(t,zeros(f),1.);assert not f.pending
    assert f.low['a']==1
    f.decide(4);f.observe(4,dict(a=torch.full((2,),10.),b=torch.zeros(2)),1.)
    assert f.low['a']==0
    for t in [5,6]:f.decide(t);f.observe(t,zeros(f),1.)
    assert f.active['a'] and not f.decide(7)['a']


def test_cooldown_leaves_complete_intervening_rounds():
    f=FeedbackFreezer(['a','b','c','d'],dict(warmup_rounds=1,patience=1,cooldown_rounds=2))
    f.decide(1);f.observe(1,zeros(f),1.);f.decide(2)
    for t in [2,3]:
        f.observe(t,zeros(f),1.);assert not f.pending;f.decide(t+1)
    assert f.freeze_round=={'a':2}
    f.observe(4,zeros(f),1.);f.decide(5)
    assert f.freeze_round=={'a':2,'b':5}


def test_minimum_deep_blocks_and_maximum_simultaneous_freezes():
    f=controller(names=('a','b','c','d','e'),min_active_groups=2,max_freeze_per_round=9)
    f.decide(1);f.observe(1,zeros(f),1.)
    assert f.pending=={'a','b','c'}
    assert f.decide(2)==dict(a=False,b=False,c=False,d=True,e=True)


def test_candidate_order_uses_noise_normalized_score_and_stable_ties():
    f=FeedbackFreezer(['a','b','c'],dict(feedback_score='raw',warmup_rounds=1,patience=1))
    f.decide(1);f.observe(1,dict(a=torch.tensor([.5]),b=torch.full((4,),.25),c=torch.zeros(1)),1.)
    assert f.pending=={'b'}  # Both energies .25, but b has larger noise tax.


@pytest.mark.parametrize('variance',[float('nan'),float('inf'),-1.,True])
def test_bad_variance_is_transactionally_rejected(variance):
    f=controller();f.decide(1);state=f.state_dict()
    with pytest.raises(ValueError):f.observe(1,zeros(f),variance)
    assert f.state_dict()==state


@pytest.mark.parametrize('problem',['nan','missing','unknown','empty','dimension_change','threshold_overflow'])
def test_bad_release_does_not_advance_or_partially_mutate_state(problem):
    f=controller(threshold_ratio=1e308 if problem=='threshold_overflow' else 1.)
    f.decide(1);release=zeros(f);variance=1.
    if problem=='nan':release['b'][0]=float('nan')
    elif problem=='missing':release.pop('b')
    elif problem=='unknown':release['x']=torch.ones(1)
    elif problem=='empty':release['b']=torch.empty(0)
    elif problem=='threshold_overflow':variance=10.
    else:
        f.observe(1,release,1.);f.decide(2);release=zeros(f);release['b']=torch.zeros(3)
    state=f.state_dict()
    with pytest.raises(ValueError):f.observe(f.last_decided,release,variance)
    assert f.state_dict()==state


def test_zero_noise_does_not_freeze_on_zero_threshold_and_frozen_values_are_ignored():
    f=controller();f.decide(1);f.observe(1,zeros(f),0.);assert not f.pending
    f.decide(2);f.observe(2,zeros(f),1.);f.decide(3)
    release=zeros(f);release['a']=torch.tensor([float('nan')])
    rows=f.observe(3,release,1.);assert rows[0]['score'] is None


@pytest.mark.parametrize('config',[dict(feedback_score='conservative'),dict(feedback_score='oracle'),
    dict(ema_decay=1.),dict(ema_decay=float('nan')),dict(ema_decay='invalid'),
    dict(patience=0),dict(patience=True),dict(warmup_rounds=0),dict(cooldown_rounds=-1),
    dict(max_freeze_per_round=0),dict(min_active_groups=0),dict(threshold_ratio=-1.)])
def test_invalid_feedback_config_rejected(config):
    with pytest.raises(ValueError):validate_feedback(config)


def test_adapter_combines_weight_and_bias_and_never_reactivates():
    model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Linear(2,1))
    mask=AdaptiveMask(model,dict(warmup_rounds=1,patience=1))
    before=mask.decide(1);delta={n:torch.zeros_like(v) for n,v in model.named_parameters()}
    rows=mask.observe(1,delta,.5)
    assert rows[0]['dimension']==6 and rows[0]['noise_tax']==3.
    assert all(m.all() for m in before.values())
    after=mask.decide(2)
    assert not after['0.weight'].any() and after['1.weight'].all()
    bad=dict(delta);bad['1.weight']=torch.zeros(3)
    with pytest.raises(ValueError):mask.observe(2,bad,.5)
    json.dumps(mask.controller.state_dict(),allow_nan=False)


@pytest.mark.parametrize('change',[dict(privacy_enabled=False),dict(freeze_rounds=[2]),
    dict(mode='BU'),dict(validation_protocol='old.json'),dict(feedback={'oracle':True})])
def test_invalid_p3_run_rejected_before_data_access(change,tmp_path):
    from src.p1_runner import run
    cfg=dict(mode='ADAPTIVE',feedback={},freeze_rounds=[],privacy_enabled=True);cfg.update(change)
    with pytest.raises(ValueError):run(cfg,tmp_path/'must_not_exist')
    assert not (tmp_path/'must_not_exist').exists()


def test_shipped_configs_preview_without_data_or_fabricated_history(monkeypatch):
    import src.data
    monkeypatch.setattr(src.data,'load_data',lambda *_:pytest.fail('No dataset access allowed'))
    files=[Path('configs')/f'p3_{name}.yaml' for name in ['raw','debiased','ema','ema_4a']]
    for path in files:
        preview=feedback_preview(yaml.safe_load(path.read_text()))
        assert preview['adaptive_freeze_rounds'] is None
        assert preview['protected_groups']==['fc']
        assert sum(preview['initial_weights'].values())==pytest.approx(1.)


def test_author_runner_feedback_only_consumes_compensated_release(monkeypatch,tmp_path):
    """Mock contract test: no local training, optimizer steps or real dataset."""
    import src.p1_runner as runner
    from torch.utils.data import TensorDataset
    prototype=torch.nn.Sequential(*[torch.nn.Linear(1,1) for _ in range(3)])
    trace=[];decisions=[];observations=[]
    class Client:
        def __init__(self,*args):
            self.local_model=copy.deepcopy(prototype)
            self.optimizer=torch.optim.SGD(self.local_model.parameters(),lr=.1)
            self.alphas=[2.,3.];self.rdp_eps_spent=np.zeros(2);self.dp_eps_spent=0.
        def set_local_sigma_and_sample_rate(self,*args):
            self.local_noise_multiplier=1.;return 1.
        def set_local_clip_value(self,radius,*args):
            self.local_clip_value=radius;return radius
        def rdp_accountant(self,**kwargs):
            trace.append('account');self.rdp_eps_spent+=.1;self.dp_eps_spent+=.1
        def train(self,model,t):
            trace.append('mock_message')
            assert trace[-2]=='account'
            self.local_model.load_state_dict(model.state_dict())
            updates={n:torch.full_like(p,9.) for n,p in model.named_parameters()}
            self.clip_layer_parameters(updates,3)
            return (updates,)
    class SpyMask(AdaptiveMask):
        def decide(self,t):
            masks=super().decide(t)
            decisions.append({g:bool(masks[g+'.weight'].any()) for g in self.names})
            return masks
        def observe(self,t,released,variance):
            assert variance==.25
            assert all(torch.count_nonzero(v)==0 for v in released.values())
            observations.append(t)
            return super().observe(t,released,variance)
    def compensate(messages,template,masks,*args):
        assert messages and any(torch.count_nonzero(v)>0 for v in messages[0].values())
        return {n:torch.zeros_like(v) for n,v in template.items()},.25
    data=TensorDataset(torch.zeros(2,1),torch.zeros(2,dtype=torch.long))
    monkeypatch.setattr(runner,'load_data',lambda *_:(data,data))
    monkeypatch.setattr(runner,'source_partition',lambda *_:[[0,1]])
    monkeypatch.setattr(runner,'groups',lambda *_:np.array([0]))
    monkeypatch.setattr(runner,'build_source_classes',lambda *_:(Client,None,lambda *_:copy.deepcopy(prototype)))
    monkeypatch.setattr(runner,'AdaptiveMask',SpyMask)
    monkeypatch.setattr(runner,'aggregate_active',compensate)
    monkeypatch.setattr(runner,'evaluate',lambda *_:(0.,0.))
    cfg=dict(mode='ADAPTIVE',feedback=dict(patience=1,warmup_rounds=1,cooldown_rounds=0),
             engineering_only=True,privacy_enabled=True,freeze_rounds=[],rounds=3,
             seed=42,data_root='must_not_read',clients=1,local_epochs=0,batch_size=1,
             epsilons=[10.,20.,30.],q_save=[1.,1.,1.],q_spend=1.,transition_round=2,
             lr=.001,rate=.9,clip=2.,device='cpu')
    out=runner.run(cfg,tmp_path/'mock_contract')
    assert decisions==[{'0':True,'1':True,'2':True},{'0':False,'1':True,'2':True},
                       {'0':False,'1':False,'2':True}]
    assert observations==[1,2,3] and trace==['account','mock_message']*3
    summary=json.loads((out/'summary.json').read_text())
    assert summary['stage']=='P3' and summary['freeze_first_round']=={'0':2,'1':3}
    assert summary['pending_after_final_round']==[]
    import csv
    rows=list(csv.DictReader((out/'feedback.csv').open()))
    assert len(rows)==9 and rows[-3]['observed']=='False' and rows[-3]['score']==''
