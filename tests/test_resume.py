import copy
import csv
import json
import numpy as np
import pytest
import torch
import yaml
import src.main as entry

@pytest.mark.parametrize('temporal',['fedavg','idp_fedavg'])
def test_resume_matches_accounting_prefix_and_preserves_rounds(tmp_path, monkeypatch, temporal):
    c=yaml.safe_load(open('configs/original_fmnist.yaml'))
    c['method']['temporal']=temporal
    c['train'].update(rounds=2,local_epochs_or_iterations=1,batch_size=4,device='cpu')
    c['data']['clients']=1
    c['privacy'].update(privacy_groups=[10],group_proportions=[1.],q_save=[1.],q_spend=1.,calibration='full_rdp_remaining',average_clip_c=.25)
    data=torch.utils.data.TensorDataset(torch.randn(12,4),torch.arange(12)%2)
    monkeypatch.setattr(entry,'load_data',lambda _: (data,data))
    monkeypatch.setattr(entry,'build_model',lambda _: torch.nn.Sequential(torch.nn.Linear(4,2)))
    original=entry.local_update
    calls=[]
    def interrupted(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:
            raise RuntimeError('simulated interruption before second release')
        return original(*args,**kwargs)
    monkeypatch.setattr(entry,'local_update',interrupted)
    out=tmp_path/'interrupted'
    with pytest.raises(RuntimeError,match='simulated interruption'):
        entry.run(c,out)
    prefix=(out/'rounds.csv').read_text().splitlines()[1]
    monkeypatch.setattr(entry,'local_update',original)
    entry.run(c,out,resume=True)
    rows=list(csv.DictReader((out/'rounds.csv').open()))
    assert len(rows)==2 and prefix==(out/'rounds.csv').read_text().splitlines()[1]
    assert json.loads((out/'checkpoint.json').read_text())['round']==2
    summary=json.loads((out/'summary.json').read_text())
    if temporal=='idp_fedavg':
        np.testing.assert_allclose(summary['epsilon_n_final'],[10.],rtol=1e-6)
    modified=copy.deepcopy(c);modified['train']['lr']*=2
    with pytest.raises(ValueError,match='configuration'):
        entry.run(modified,out,resume=True)
