"""Information preference changes scores, never the release/accountant mapping."""
import copy
import numpy as np
import pytest
import torch

from src.dvpa_config import ROOT, load_config, public_preflight, resolve, smoke_config
from src.dvpa_controller import DecisionValueController
from src.dvpa_feedback import DPHistoryFeedback
from src.dvpa_filter import GroupRDPFilter


@pytest.fixture(scope='module')
def prepared():
    torch.set_num_threads(1)
    p = public_preflight(smoke_config(load_config(ROOT/'configs/dvpa_fmnist.yaml')))
    f, ledger, c = DPHistoryFeedback(p), GroupRDPFilter(p), DecisionValueController(p)
    f.prepare(1)
    a, _ = c.choose(1, f, ledger.remaining())
    cert = ledger.certify(a); ledger.reserve(cert)
    f.observe(1, {n: torch.ones(d)*0.01 for n,d in p.dimensions.items()}, cert)
    f.prepare(2)
    return p,f,ledger


def test_weight_scales_only_information_and_preserves_release_cost(prepared):
    original,f,ledger=prepared
    p=copy.deepcopy(original)
    c=DecisionValueController(p)
    actions,tails=c.candidate_context(2,f,ledger.remaining())
    actions=actions[:4]
    old=c.score_actions(2,f,actions,tails)
    assert any(v['information_value'] > 0 for v in old)
    certificates=[GroupRDPFilter(p).certify(a).as_dict() for a in actions]
    p.config['controller']['information_value_weight']=10.0
    weighted=c.score_actions(2,f,actions,tails)
    for a,b in zip(old,weighted):
        assert a['objective']==a['current_loss']+a['tail_loss']
        assert b['objective']==pytest.approx(a['objective']-9*a['information_value'],rel=1e-13)
        assert b['unweighted_objective']==a['objective']
        assert b['weighted_information_value']==pytest.approx(10*a['information_value'])
        for key in ['current_loss','tail_loss','information_value','tower_correction']:
            assert b[key]==a[key]
    assert certificates==[GroupRDPFilter(p).certify(a).as_dict() for a in actions]
    p.config['controller']['information_value_weight']=0.0
    zero=c.score_actions(2,f,actions,tails)
    for a,b in zip(old,zero):
        assert b['objective']==pytest.approx(a['frozen_objective'],rel=1e-13)


@pytest.mark.parametrize('weight',[-1.0,float('nan'),float('inf'),True,'10'])
def test_bad_weight_rejected(weight):
    cfg=load_config(ROOT/'configs/dvpa_fmnist.yaml')
    cfg['controller']['information_value_weight']=weight
    with pytest.raises(ValueError,match='information value weight'):
        resolve(cfg)


def test_frozen_and_terminal_ignore_weight(prepared):
    original,f,ledger=prepared
    p=copy.deepcopy(original);c=DecisionValueController(p)
    actions,tails=c.candidate_context(2,f,ledger.remaining())
    actions=actions[:2]
    p.config['controller'].update(mode='frozen_info',information_value_weight=1.0)
    first=c.score_actions(2,f,actions,tails)
    p.config['controller']['information_value_weight']=10.0
    second=c.score_actions(2,f,actions,tails)
    assert [r['objective'] for r in first]==[r['objective'] for r in second]
    p.config['controller']['mode']='decision_value'
    last=copy.deepcopy(f);last.prepared_round=p.rounds;last.last_observed=p.rounds-1
    _,log=c.choose(p.rounds,last,ledger.remaining())
    assert log['information_value']==log['weighted_information_value']==0.0
    assert log['objective']==log['unweighted_objective']
