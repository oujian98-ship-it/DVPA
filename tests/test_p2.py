import copy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from src.p2_allocation import weights, validate_allocation, allocation_plan
from src.p1_freezing import joint_group_clip, masked_noise
from src.p2_diagnostics import oracle_moment_weights, clipping_diagnostics


@pytest.mark.parametrize('method',['equal_active','param_count','exponential_depth_time','prior_decay_4A'])
def test_allocations_match_closed_form_after_freezing(method):
    names=['a','b','c'];dims=dict(a=0,b=4,c=12);active=dict(a=False,b=True,c=True)
    cfg=dict(layer_clip=method,prior_amplitude=[3.,2.,1.],prior_decay=[.5,.2,.1],rho=2.,depth_rate=2.,time_scale=10.)
    w=weights(names,dims,active,5,cfg)
    expected={'equal_active':[1.,1.], 'param_count':[4.,12.],
              'exponential_depth_time':np.exp([1.,2.]),
              'prior_decay_4A':(np.array([2.,1.])*np.exp(-np.array([.2,.1])*5))**(2/3)}[method]
    np.testing.assert_allclose([w['b'],w['c']],np.array(expected)/sum(expected))
    assert w['a']==0 and sum(w.values())==pytest.approx(1.)


def test_log_domain_preserves_ratios_when_old_log_clipping_would_flatten():
    cfg=dict(layer_clip='prior_decay_4A',prior_amplitude={'a':1.,'b':1.},prior_decay={'a':.1,'b':.10001},rho=2.)
    w=weights(['a','b'],dict(a=1,b=1),dict(a=True,b=True),100000,cfg)
    assert w['a']/w['b']==pytest.approx(np.exp(2/3))


@pytest.mark.parametrize('change',[
    {'rho':float('nan')},{'rho':0},{'prior_decay':[.1,-.2]},
    {'prior_amplitude':[1.]},{'prior_amplitude':[1.,float('inf')]},
    {'prior_decay':{'a':.1,'wrong':.2}}, {'prior_amplitude':[1.,0.]},
])
def test_invalid_priors_rejected_even_for_frozen_layers(change):
    cfg=dict(layer_clip='prior_decay_4A',prior_amplitude=[1.,1.],prior_decay=[.1,.2]);cfg.update(change)
    with pytest.raises(ValueError):weights(['a','b'],dict(a=0,b=1),dict(a=False,b=True),1,cfg)


def test_weighted_clipping_joint_bound_and_noise_independence():
    masks={'a.weight':torch.tensor([True,False]),'a.bias':torch.tensor([True]),'b.weight':torch.tensor([True]),'c.weight':torch.tensor([False])}
    before={n:torch.full(m.shape,100.) for n,m in masks.items()}
    outputs=[]
    for w in [dict(a=.25,b=.75,c=0.),dict(a=.9,b=.1,c=0.)]:
        out=copy.deepcopy(before);joint_group_clip(out,2.,masks,w)
        assert sum(v.double().square().sum() for v in out.values())<=4.000001
        a_norm=(out['a.weight'].double().square().sum()+out['a.bias'].double().square().sum()).sqrt().item()
        assert a_norm==pytest.approx(2*np.sqrt(w['a']),rel=1e-6)
        assert out['a.weight'][1]==0 and out['c.weight'][0]==0
        noise={n:torch.zeros_like(v) for n,v in out.items()}
        masked_noise(noise,masks,.7,torch.Generator().manual_seed(6));outputs.append(noise)
    assert all(torch.equal(outputs[0][n],outputs[1][n]) for n in masks)
    with pytest.raises(ValueError):joint_group_clip(copy.deepcopy(before),2.,masks,dict(a=.5,b=.5,c=.1))


def test_equal_allocation_regresses_to_p1_and_zero_weight_is_valid():
    masks={'a.weight':torch.ones(4,dtype=torch.bool),'b.weight':torch.ones(3,dtype=torch.bool)}
    delta={n:torch.full(m.shape,2.) for n,m in masks.items()}
    a=copy.deepcopy(delta);b=copy.deepcopy(delta)
    joint_group_clip(a,1.,masks);joint_group_clip(b,1.,masks,dict(a=.5,b=.5))
    assert all(torch.equal(a[n],b[n]) for n in a)
    joint_group_clip(b,1.,masks,dict(a=0.,b=1.));assert b['a.weight'].count_nonzero()==0


def test_offline_diagnostics_are_explicit_and_measure_clipping_not_mean_bias():
    with pytest.raises(ValueError):oracle_moment_weights(dict(a=1.),dict(a=True))
    w=oracle_moment_weights(dict(a=64.,b=1.,c=9.),dict(a=True,b=True,c=False),offline_only=True)
    assert w==pytest.approx(dict(a=.8,b=.2,c=0.))
    updates={'a.weight':torch.tensor([3.,4.]),'b.weight':torch.tensor([2.])}
    masks={'a.weight':torch.tensor([True,True]),'b.weight':torch.tensor([False])}
    with pytest.raises(ValueError):clipping_diagnostics(updates,masks,2.,dict(a=1.,b=0.))
    result=clipping_diagnostics(updates,masks,2.,dict(a=1.,b=0.),offline_only=True)
    assert result['layers']['a']['clip_factor']==pytest.approx(.4)
    assert result['layers']['a']['clip_error_sq']==pytest.approx(9.,rel=1e-6)
    assert result['layers']['b']['freeze_error_sq']==4.
    assert updates['a.weight'].tolist()==[3.,4.]


def test_public_entry_rejects_oracle_and_undeclared_prior_before_loading_data(tmp_path):
    from src.p1_runner import run
    with pytest.raises(ValueError,match='oracle'):run({'allocation':{'layer_clip':'oracle_moment'}},tmp_path/'blocked')
    with pytest.raises(ValueError,match='prior_source'):validate_allocation({'layer_clip':'prior_decay_4A'})
    assert not (tmp_path/'blocked').exists()


def test_all_shipped_configs_have_same_masks_and_no_data_access(monkeypatch):
    import src.data
    monkeypatch.setattr(src.data,'load_data',lambda *_:pytest.fail('Preview must not load data'))
    plans=[]
    for p in Path('configs').glob('p2_*_bu.yaml'):
        cfg=yaml.safe_load(p.read_text());plan=allocation_plan(cfg);plans.append(plan)
        assert len(plan['rows'])==100
        for t in range(1,26):
            rows=[r for r in plan['rows'] if r['round']==t]
            assert sum(r['weight'] for r in rows)==pytest.approx(1.)
            assert all(r['weight']==0 for r in rows if r['active_params']==0)
    assert len(plans)==4
    assert all([r['active_params'] for r in p['rows']]==[r['active_params'] for r in plans[0]['rows']] for p in plans)
