import copy
import json
import math
import numpy as np
import pytest
import torch
from scipy.stats import ncx2
from src.diagnostic_freezing import DiagnosticFreezer,energy_ucb,validate_diagnostic
from src.dp_feedback_freezer import DEFAULTS
from src.p1_freezing import StaticMask
from src.p3_freezing import AdaptiveMask


def policy(kind='delayed',**changes):
    p=dict(kind=kind,min_freeze_round=6,min_active_fraction=.5,freeze_order='any',protected_groups=['fc1'])
    if kind=='confidence':p.update(alpha_total=.05,drift_window=3,max_ratio_range=.25,drift_margin_ratio=.25)
    return {**p,**changes}


@pytest.mark.parametrize('d,nc',[(2,3.),(832,20.),(51264,1000.),(524800,2000.)])
def test_ucb_inverts_noncentral_lower_tail_not_wrong_quantile(d,nc):
    alpha=.0005;v=.03;x=ncx2.ppf(alpha,d,nc)
    upper=energy_ucb(x*v,d,v,alpha)
    assert upper==pytest.approx(nc*v,rel=1e-7,abs=1e-8)
    assert energy_ucb(ncx2.ppf(.5,d,nc)*v,d,v,alpha)>nc*v


def test_ucb_boundary_zero_noise_and_monotonicity():
    assert energy_ucb(0.,8,1.,.05)==0.
    assert energy_ucb(3.,8,0.,.05)==3.
    assert energy_ucb(8.,8,1.,.01)>energy_ucb(8.,8,1.,.1)
    assert energy_ucb(9.,8,1.,.05)>energy_ucb(8.,8,1.,.05)
    for values in [(-1,8,1,.05),(1,0,1,.05),(1,8,-1,.05),(1,8,1,0),(math.inf,8,1,.05)]:
        with pytest.raises(ValueError):energy_ucb(*values)


def test_ucb_frequentist_coverage_at_known_signal_no_data_training():
    rng=np.random.default_rng(317);alpha=.05
    # Inversion coverage follows the CDF event exactly; verify inversion over
    # independent synthetic noncentral samples, not just implementation values.
    d=16;nc=4.;xs=rng.noncentral_chisquare(d,nc,size=500)
    failures=sum(energy_ucb(float(x),d,1.,alpha)<nc-1e-9 for x in xs)
    assert 8<=failures<=43


def make(p,**feedback):
    return DiagnosticFreezer(['conv1','conv2','fc1','fc'],dict(conv1=8,conv2=16,fc1=64,fc=12),
                            {**DEFAULTS,**feedback},p,12)


def zero_stats(c):return {g:(c.dimensions[g],0.) for g in c.names if c.active[g]}


def test_delayed_freezing_never_before_minimum_and_capacity_preserves_fc1():
    c=make(policy(),warmup_rounds=1,patience=5,cooldown_rounds=1)
    for t in range(1,13):
        c.decide(t);c.observe_statistics(t,zero_stats(c),1.)
        if t<5:assert not c.pending and not c.freeze_round
    assert c.freeze_round==dict(conv1=6,conv2=8)
    assert c.active==dict(conv1=False,conv2=False,fc1=True,fc=True)
    json.dumps(c.state_dict(),allow_nan=False)


def test_capacity_blocks_huge_layer_without_explicit_protection():
    c=make(policy(protected_groups=[]),warmup_rounds=1,patience=1)
    for t in range(1,13):c.decide(t);c.observe_statistics(t,zero_stats(c),1.)
    assert c.active['fc1'] and sum(c.dimensions[g] for g in c.names if c.active[g])>=50


def test_low_counter_resets_and_observation_cannot_change_current_mask():
    c=make(policy(min_freeze_round=2),warmup_rounds=1,patience=2)
    c.decide(1);c.observe_statistics(1,zero_stats(c),1.);assert not c.pending
    c.decide(2);stats=zero_stats(c);stats['conv1']=(8,10000.);c.observe_statistics(2,stats,1.)
    assert c.low['conv1']==0
    before=c.active.copy();assert c.decide(2)==before
    with pytest.raises(ValueError):c.observe_statistics(2,stats,1.)
    with pytest.raises(ValueError):c.decide(4)


def test_confidence_bottom_up_cannot_skip_high_signal_first_layer():
    c=make(policy('confidence',min_freeze_round=2,freeze_order='bottom_up'),warmup_rounds=1,patience=1)
    for t in range(1,7):
        c.decide(t);stats=zero_stats(c);stats['conv1']=(8,10000.)
        c.observe_statistics(t,stats,1.)
        assert not c.pending
    assert c.alpha==pytest.approx(.05/(12*4))


def test_confidence_drift_guard_prevents_unstable_low_score_freeze():
    c=make(policy('confidence',min_freeze_round=2,max_ratio_range=.00001),warmup_rounds=1,patience=1)
    for t in range(1,6):
        c.decide(t);stats={g:(c.dimensions[g],float(c.dimensions[g])*(.2 if t%2 else 1.)) for g in c.names if c.active[g]}
        rows=c.observe_statistics(t,stats,1.)
        assert not c.pending
        if t>=3:assert not rows[0]['drift_stable']


def test_replay_adapter_masks_match_static_without_using_observations():
    model=torch.nn.Sequential(*[torch.nn.Linear(2,2) for _ in range(4)])
    p=dict(kind='replay',replay_rounds=[3,6])
    static=StaticMask(model,'BU',[3,6]);replay=AdaptiveMask(model,DEFAULTS,42,p,8)
    for t in range(1,9):
        a=static.decide(t);b=replay.decide(t)
        assert all(torch.equal(a[n],b[n]) for n in a)
        replay.observe(t,{n:torch.full_like(v,1000.*t) for n,v in model.named_parameters()},1.)


def test_policy_validation_and_observation_failure_leave_state_unchanged():
    c=make(policy());c.decide(1);before=c.state_dict()
    with pytest.raises(ValueError):c.observe_statistics(1,{'conv1':(9,0.)},1.)
    assert c.state_dict()==before
    for p in [policy(extra=True),policy(min_freeze_round=0),policy(min_active_fraction=0),policy('confidence',alpha_total=0)]:
        with pytest.raises(ValueError):validate_diagnostic(p,DEFAULTS,12)


def test_d1_gate_checks_full_trajectory_and_model_even_if_final_score_matches(tmp_path):
    from scripts.validate_freezing_diagnostics import compare_replay
    left=tmp_path/'a';right=tmp_path/'b';left.mkdir();right.mkdir()
    for folder in [left,right]:
        for name in ['rounds.csv','layers.csv','privacy.csv','layer_registry.json','source_manifest.json']:
            (folder/name).write_text('same')
        torch.save({'p':torch.ones(3)},folder/'model.pt')
        np.savez(folder/'accountant.npz',rdp=np.ones((2,3)))
    assert compare_replay(left,right)['passed']
    (right/'rounds.csv').write_text('different earlier rounds')
    assert not compare_replay(left,right)['passed']
    (right/'rounds.csv').write_text('same');torch.save({'p':torch.zeros(3)},right/'model.pt')
    assert not compare_replay(left,right)['passed']
