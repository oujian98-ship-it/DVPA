import numpy as np
import torch
import pytest
from src.upstream_runner import build_source_classes, source_model, joint_tensor_clip, source_partition

def test_actual_source_model_has_valid_convolutions():
    model=source_model()('fmnist',125)
    assert model.conv1[0].padding==(0,0)
    assert model.fc1[0].in_features==1024
    assert sum(p.numel() for p in model.parameters())==582026
    assert model(torch.randn(2,1,28,28)).shape==(2,10)

def test_source_clips_tensors_not_joint_groups():
    Client,_,_=build_source_classes()
    c=Client.__new__(Client);c.local_clip_value=250.
    vectors={f'g{i}.weight':torch.tensor([1000.]) for i in range(4)}
    vectors.update({f'g{i}.bias':torch.tensor([1000.]) for i in range(4)})
    c.clip_layer_parameters(vectors,4)
    np.testing.assert_allclose([v.norm().item() for v in vectors.values()],[125.]*8)
    # Counterexample to interpreting upstream c as a joint sensitivity bound.
    assert sum(v.square().sum() for v in vectors.values()).sqrt()>250.

@pytest.mark.parametrize('scheme,factor',[('time_adaptive',.9**2),('baseline',1.)])
def test_actual_source_training_return_units(scheme,factor):
    from types import SimpleNamespace
    Client,_,_=build_source_classes(scheme)
    c=Client.__new__(Client)
    c.local_model=torch.nn.Linear(1,2)
    global_model=torch.nn.Linear(1,2)
    before={k:v.clone() for k,v in global_model.state_dict().items()}
    c.train_dataset=torch.utils.data.TensorDataset(torch.ones(2,1),torch.zeros(2,dtype=torch.long))
    c.epochs=1;c.batch_size=2;c.device='cpu';c.learning_rate=.01;c.rate=.9
    c.loss=torch.nn.CrossEntropyLoss()
    c.optimizer=torch.optim.SGD(c.local_model.parameters(),lr=.01)
    c.scheduler=SimpleNamespace(step=lambda:None)
    c.test_metrics=lambda:(0,1);c.train_metrics=lambda:(0.,1);c.evaluate=lambda:0.
    c.clip_layer_parameters=lambda *args:None;c.add_noise=lambda *args:None
    c.rdp_accountant=lambda:None
    c.id=0;c.cprint_rounds=1000000;c.local_clip_value=250.
    c.sample_rate=.9;c.local_noise_multiplier=1.;c.dp_eps_budget=10.
    c.dp_eps_spent=0.;c.previous_dp_eps_spent=0.;c.rdp_eps_sample=0.;c.best_alpha=2.
    result=c.train(global_model,0)[0]
    for name,param in c.local_model.state_dict().items():
        torch.testing.assert_close(result[name],(param-before[name])*factor)
    assert any(v.abs().sum()>0 for v in result.values())

def test_corrected_source_tensor_sensitivity():
    from types import SimpleNamespace
    vectors={str(i):torch.tensor([1000.]) for i in range(8)}
    joint_tensor_clip(SimpleNamespace(local_clip_value=250.),vectors,4)
    assert sum(v.double().square().sum() for v in vectors.values()).sqrt()<=250.00001

def test_source_partition_has_no_overlap_and_meets_minimum():
    labels=np.arange(600)%10
    np.random.seed(42);a=source_partition(labels,6)
    np.random.seed(42);b=source_partition(labels,6)
    assert a==b
    assert min(map(len,a))>=40
    assert sorted(sum(a,[]))==list(range(600))

def test_corrected_noise_uses_actual_ledger_orders():
    from src.upstream_runner import checked_source_noise, cached_rdp
    from opacus.accountants.analysis import rdp
    orders=(2.,3.,4.,8.,16.,32.,64.)
    kw=dict(target_epsilon=10.,target_delta=1e-5,sample_rate=.9,epochs=3)
    sigma=checked_source_noise(tuple(sorted(kw.items())),orders)
    rho=cached_rdp(.9,sigma,int(3/.9),orders)
    assert rdp.get_privacy_spent(orders=orders,rdp=rho,delta=1e-5)[0]<=10.
