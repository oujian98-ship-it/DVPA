import copy
import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset
import yaml

from src.paper_setting_fourway import (METHODS, audit, eps_from_rdp, increment,
    layer_allocation, round_parameters, run, source_classes)


def base():
    return yaml.safe_load((Path(__file__).parents[1]/'configs/paper_setting_fourway_seed42.yaml').read_text())


def toy_plan():
    return dict(orders=[2.,4.,8.,16.], time_q=[[.5,.6,.7],[.9,.9,.9]],
        time_sigma=[[3.,2.,1.5],[2.9,1.9,1.4]], uniform_q=[[.9]*3]*2,
        uniform_sigma=[[4.,3.,2.]]*2)


def test_diagonal_noise_joint_sensitivity_equals_each_client_group_sigma():
    cfg = base()
    plan = toy_plan()
    names = list(cfg['layer_decay'])
    for method in METHODS:
        for t in (1,2):
            rp = round_parameters(cfg,plan,method,t,[34,43,23],names)
            np.testing.assert_allclose(rp['sigma'],rp['effective_sigma'],rtol=1e-12)
            np.testing.assert_allclose(rp['radii']*rp['sigma'],cfg['clip']*rp['global_sigma'])
            assert sum(rp['p'].values()) == pytest.approx(1.)
    uniform = round_parameters(cfg,plan,'UNIFORM',2,[34,43,23],names)
    layer = round_parameters(cfg,plan,'LAYER_ONLY',2,[34,43,23],names)
    np.testing.assert_array_equal(uniform['q'],layer['q'])
    np.testing.assert_array_equal(uniform['radii'],layer['radii'])
    np.testing.assert_allclose(uniform['effective_sigma'],layer['effective_sigma'])
    assert uniform['variance']['conv1'] != layer['variance']['conv1']
    # Larger fc budget decreases its noise, with exactly unchanged clipping.
    assert layer['p']['fc'] > layer['p']['conv1']
    assert layer['variance']['fc'] < layer['variance']['conv1']


def test_uniform_rdp_is_equal_each_round_and_no_time_means_no_saving():
    cfg = base()
    plan = toy_plan()
    names = list(cfg['layer_decay'])
    a = round_parameters(cfg,plan,'UNIFORM',1,[34,43,23],names)
    b = round_parameters(cfg,plan,'UNIFORM',2,[34,43,23],names)
    c = round_parameters(cfg,plan,'TIME_ONLY',1,[34,43,23],names)
    assert list(a['q']) == [.9,.9,.9]
    assert list(c['q']) == [.5,.6,.7]
    for g in range(3):
        np.testing.assert_array_equal(increment(a['q'][g],a['sigma'][g],tuple(plan['orders'])),
                                      increment(b['q'][g],b['sigma'][g],tuple(plan['orders'])))
    omega,p = layer_allocation(names,25,'JOINT',cfg)
    assert set(omega.values()) == {.25}
    assert p['conv1'] < p['fc']


def test_author_pipeline_retains_optimizer_and_audits_actual_noise(tmp_path,monkeypatch):
    import src.paper_setting_fourway as module
    torch.set_num_threads(1)
    cfg = base()
    cfg.update(clients=3,rounds=2,local_epochs=1,batch_size=4,device='cpu',transition_round=1)
    data = TensorDataset(torch.rand(12,1,28,28),torch.arange(12)%10)
    test = TensorDataset(torch.rand(6,1,28,28),torch.arange(6)%10)
    monkeypatch.setattr(module,'load_data',lambda config:(data,test))
    monkeypatch.setattr(module.upstream,'source_partition',lambda labels,n:[list(range(i,12,n)) for i in range(n)])
    cls,server,factory = source_classes()
    monkeypatch.setattr(module,'source_classes',lambda:(cls,server,factory))
    observed = []
    original_train = cls.train
    def train(self,*args):
        observed.append((self.id,id(self.optimizer),self.scheduler.last_epoch))
        assert self.optimizer.defaults['momentum'] == .9
        assert self.optimizer.defaults['weight_decay'] == 5e-4
        return original_train(self,*args)
    cls.train = train
    plan = toy_plan()
    plan['orders'] = list(cls(0,None,None,1,10.,1e-5,2,3,.001,4,.9,'fmnist',1000000,[],.5,'cpu').alphas)
    plan['time_q'] = copy.deepcopy(plan['uniform_q'])
    plan['time_sigma'] = copy.deepcopy(plan['uniform_sigma'])
    plan['matched_actual_epsilon'] = [eps_from_rdp(2*increment(.9,s,tuple(plan['orders'])),
        tuple(plan['orders']),cfg['delta']) for s in plan['uniform_sigma'][0]]
    out = tmp_path/'run'
    result = run(cfg,plan,'JOINT',out)
    assert result['parameters'] == 582026
    assert result['freezing'] is False
    assert len(observed) == 6  # all clients train, not just selected clients
    for g in range(3):
        slots = [r for r in observed if r[0]==g]
        assert slots[0][1] == slots[1][1]  # exact same persistent optimizer
        assert [r[2] for r in slots] == [0,1]  # per-local-epoch cosine advanced
    assert audit(out,plan)['status'] == 'passed'
    with (out/'layers.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    rows[0]['noise_variance'] = str(float(rows[0]['noise_variance'])/4)
    with (out/'layers.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        audit(out,plan)
