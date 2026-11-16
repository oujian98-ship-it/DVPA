import copy
import numpy as np
import pytest
import torch
import yaml
from src.layer_clip import clip, validate
from src.server import aggregate, add_client_noise
from src.models import PrototypeCNN
from src.layer_registry import LayerRegistry
from src.clients import local_update
from src.data import dirichlet_split
from src.original_scheduler import PrivacyScheduler
from src.accountant import Ledger, increment
from src.static_freezer import StaticFreezer
from src.dp_feedback_freezer import FeedbackFreezer
from src.diagnostics import conditional_mse

@pytest.mark.parametrize('radius',[0., 1., 100.])
def test_joint_clipping(radius):
    d = {'a':torch.full((30,),1e20), 'b':torch.zeros(5), 'c':torch.randn(8)}
    a, w = dict(a=True,b=True,c=False),dict(a=.3,b=.7,c=0.)
    out = clip(d,radius,a,w)
    assert sum(v.double().square().sum() for v in out.values()) <= radius**2+1e-5
    assert torch.count_nonzero(out['c']) == 0
    with pytest.raises(ValueError):validate(dict(a=False),dict(a=0))

@pytest.mark.parametrize('count',[0,2,5])
def test_compensated_noise(count):
    g=torch.Generator().manual_seed(4)
    template=dict(a=torch.zeros(20000),b=torch.zeros(1))
    active=dict(a=True,b=False)
    messages=[add_client_noise(template,3.,5,active,g) for _ in range(count)]
    result,var=aggregate(messages,template,active,5,.6,3.,g)
    assert var == 1.
    assert abs(result['a'].mean()) < .03
    assert abs(result['a'].var()-var) < .04
    assert result['b'].item() == 0

def test_denominator():
    g=torch.Generator().manual_seed(1)
    result,_=aggregate([dict(a=torch.tensor([2.]))],dict(a=torch.zeros(1)),dict(a=True),4,.5,0.,g)
    assert result['a'].item()==1.

def test_frozen_training():
    torch.manual_seed(1)
    model=PrototypeCNN()
    # Add BN to exercise frozen running statistics.
    model.conv1.append(torch.nn.BatchNorm2d(16))
    registry=LayerRegistry(model)
    active=dict.fromkeys(registry.names,True);active['conv1']=False
    before=copy.deepcopy(model.state_dict())
    data=torch.utils.data.TensorDataset(torch.randn(8,1,28,28),torch.arange(8)%10)
    delta=local_update(model,registry,data,list(range(8)),active,dict(momentum=.9,batch_size=4,local_epochs_or_iterations=1),42,'cpu',.1)
    assert torch.count_nonzero(delta['conv1'])==0
    assert delta['fc2'].norm()>0
    local=copy.deepcopy(model);local.train();registry.freeze(local,active)
    local(torch.randn(8,1,28,28))
    for n,v in before.items():
        if n.startswith('conv1.'):
            assert torch.equal(local.state_dict()[n],v)

def test_accounting():
    cfg=yaml.safe_load(open('configs/original_fmnist.yaml'))
    cfg['data']['clients']=6;cfg['train']['rounds']=3;cfg['privacy']['transition_round']=2
    scheduler=PrivacyScheduler(cfg)
    ledger=Ledger(scheduler.targets,1e-5)
    expected=np.zeros_like(ledger.rdp)
    for t in range(1,4):
        rp=scheduler.params(t);ledger.record(rp)
        expected+=np.array([increment(q,s) for q,s in zip(rp.q_n,rp.sigma_n)])
        np.testing.assert_allclose(rp.clip_n*rp.sigma_n,250*rp.sigma_global)
    np.testing.assert_allclose(expected,ledger.rdp)
    assert (ledger.epsilons()<=scheduler.targets+1e-6).all()
    with pytest.raises(ValueError):Ledger([10],1e-5,'local_messages_visible')

def test_uniform_schedule():
    cfg=yaml.safe_load(open('configs/original_fmnist.yaml'))
    cfg['data']['clients']=6;cfg['train']['rounds']=3
    cfg['privacy']['q_save']=[.9]*3
    cfg['privacy']['calibration']='full_rdp_remaining'
    a=PrivacyScheduler(cfg)
    cfg['method']['temporal']='idp_fedavg'
    b=PrivacyScheduler(cfg)
    for t in range(1,4):np.testing.assert_allclose(a.params(t).sigma_n,b.params(t).sigma_n)

def test_monotonic_and_feedback():
    cfg=dict(layer_freeze='static_random',freeze_rounds=[2,3,None])
    a=StaticFreezer(['a','b','c'],cfg,42);b=StaticFreezer(['a','b','c'],cfg,42)
    assert a.order==b.order
    assert sum(a.decide(1).values())==3
    assert sum(a.decide(3).values())==1
    f=FeedbackFreezer(['a','b'],dict(layer_freeze='dp_feedback',patience=1,warmup_rounds=1))
    assert all(f.decide(1).values())
    f.observe(1,dict(a=torch.zeros(2),b=torch.ones(2)*100),1.)
    assert f.active['a']
    assert not f.decide(2)['a']

def test_partition_reproducibility():
    labels=np.arange(100)%10
    a=dirichlet_split(labels,10,.1,42)
    assert a==dirichlet_split(labels,10,.1,42)
    assert sorted(sum(a,[]))==list(range(100))
    assert all(a)

def test_conditional_mse_monte_carlo():
    rng=np.random.default_rng(42)
    u=np.array([[1.,2.],[2.,-1.],[.2,.3]])
    v=u*.7;q=np.array([.3,.6,.9]);scale=.4
    draws=(rng.random((60000,3))<q)@v/q.sum()+rng.normal(size=(60000,2))*scale/q.sum()
    observed=((draws-u.mean(0))**2).sum(1).mean()
    assert observed==pytest.approx(conditional_mse(u,v,q,scale),rel=.025)
