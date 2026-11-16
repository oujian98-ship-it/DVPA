import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.p1_freezing import StaticMask, group_counts, joint_group_clip, masked_noise, install_local_freeze, aggregate_active
from src.upstream_runner import source_model

@pytest.mark.parametrize('mode',['NF','BU','TD','RAND','SAME-DIM','NO-DP'])
def test_masks_are_shared_monotonic_and_exact_dimensions(mode):
    torch.set_num_threads(1)
    model=source_model()('fmnist',125)
    scheduler=StaticMask(model,mode,[2,3],42)
    previous=scheduler.decide(1)
    for t in [2,3,4]:
        masks=scheduler.decide(t)
        assert all(not torch.any(masks[n]&~previous[n]) for n in masks)
        assert sum(group_counts(masks).values())>0
        if mode=='SAME-DIM':
            bu=StaticMask(model,'BU',[2,3],42).decide(t)
            assert sum(group_counts(masks).values())==sum(group_counts(bu).values())
            assert any(not torch.equal(masks[n],bu[n]) for n in masks)
        previous=masks
    with pytest.raises(ValueError):scheduler.decide(1)
    a=StaticMask(model,'RAND',[2,3],7);b=StaticMask(model,'RAND',[2,3],7)
    assert a.order==b.order

def test_group_clipping_combines_weight_and_bias_and_zeroes_frozen():
    masks={'a.weight':torch.tensor([True,False]),'a.bias':torch.tensor([True]),'b.weight':torch.tensor([True]),'c.weight':torch.tensor([False])}
    values={n:torch.full(m.shape,1e10) for n,m in masks.items()}
    joint_group_clip(values,3.,masks)
    assert sum(v.double().square().sum() for v in values.values())<=9.000001
    assert values['a.weight'][1]==0 and values['c.weight'][0]==0
    assert (values['a.weight'].double().square().sum()+values['a.bias'].double().square().sum()).sqrt()==pytest.approx(3/np.sqrt(2),rel=1e-6)

@pytest.mark.parametrize('selected',[0,2,5])
def test_active_noise_compensation_and_expected_energy(selected):
    generator=torch.Generator().manual_seed(31)
    masks={'a.weight':torch.ones(25000,dtype=torch.bool),'b.weight':torch.zeros(1,dtype=torch.bool)}
    template={n:torch.zeros(m.shape) for n,m in masks.items()}
    messages=[]
    for _ in range(selected):
        message={n:v.clone() for n,v in template.items()}
        masked_noise(message,masks,3/np.sqrt(5),generator);messages.append(message)
    release,var=aggregate_active(messages,template,masks,5,.6,3.,generator)
    assert var==1 and release['b.weight'].item()==0
    assert abs(release['a.weight'].mean())<.025
    assert release['a.weight'].var().item()==pytest.approx(var,rel=.04)
    assert release['a.weight'].square().sum().item()==pytest.approx(25000*var,rel=.04)

def test_local_whole_block_freezes_bn_and_existing_momentum():
    torch.manual_seed(1)
    model=torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4)),torch.nn.Linear(4,2))
    opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=.2)
    x=torch.randn(12,4)
    model(x).square().sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    before=copy.deepcopy(model.state_dict())
    momentum={n:opt.state[p]['momentum_buffer'].clone() for n,p in model.named_parameters()}
    masks={n:torch.full_like(p,n.startswith('1.'),dtype=torch.bool) for n,p in model.named_parameters()}
    cleanup=install_local_freeze(SimpleNamespace(local_model=model,optimizer=opt),masks)
    try:
        model.train()  # Source Client.train resets this after freeze is installed.
        model(x).square().sum().backward();opt.step()
    finally:cleanup()
    for n,v in model.state_dict().items():
        if n.startswith('0.'):assert torch.equal(v,before[n])
    for n,p in model.named_parameters():
        if n.startswith('0.'):assert torch.equal(opt.state[p]['momentum_buffer'],momentum[n])
    assert not torch.equal(model.state_dict()['1.weight'],before['1.weight'])

def test_partial_coordinates_survive_decay_and_old_momentum():
    torch.manual_seed(2)
    model=torch.nn.Sequential(torch.nn.Linear(4,2))
    opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=.3)
    x=torch.randn(6,4)
    model(x).square().sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    masks={n:torch.ones_like(p,dtype=torch.bool) for n,p in model.named_parameters()}
    masks['0.weight'].view(-1)[:3]=False
    p=model[0].weight;old=p.detach().clone();old_m=opt.state[p]['momentum_buffer'].clone()
    cleanup=install_local_freeze(SimpleNamespace(local_model=model,optimizer=opt),masks)
    try:
        model(x).square().sum().backward();opt.step()
    finally:cleanup()
    frozen=~masks['0.weight']
    assert torch.equal(p[frozen],old[frozen])
    assert torch.equal(opt.state[p]['momentum_buffer'][frozen],old_m[frozen])
    assert not torch.equal(p[~frozen],old[~frozen])

def test_p1_formal_run_is_gated(tmp_path,monkeypatch):
    from src.p1_runner import run
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError,match='P0 gate'):run({},tmp_path/'blocked')

def test_nf_and_no_freeze_bu_match_deterministically_before_event():
    model=source_model()('fmnist',125)
    a=StaticMask(model,'NF',[10,18],42).decide(1)
    b=StaticMask(model,'BU',[10,18],42).decide(1)
    torch.manual_seed(5)
    delta={n:torch.randn_like(p) for n,p in model.named_parameters()}
    left=copy.deepcopy(delta);right=copy.deepcopy(delta)
    joint_group_clip(left,1.,a);joint_group_clip(right,1.,b)
    masked_noise(left,a,.1,torch.Generator().manual_seed(9));masked_noise(right,b,.1,torch.Generator().manual_seed(9))
    assert all(torch.equal(left[n],right[n]) for n in left)
