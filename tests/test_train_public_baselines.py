"""Native IDP/SAYG policies, shared 59k training and full 10k evaluation.

The pipeline test uses zero-update clients for engineering coverage only.
"""
import copy
import json
import numpy as np
import pytest
import torch
from src.dvpa_config import ROOT,load_config
from src.brdp_source_protocol import validate_config,public_training_data_config,preflight,PersistentSourceClient
from src.brdp_source_runner import run
from src.brdp_source_audit import audit


@pytest.mark.parametrize('method',['IDP_FEDAVG','SPEND_AS_YOU_GO'])
def test_native_baselines_use_train_public_split_and_replay(tmp_path,monkeypatch,method):
    cfg=copy.deepcopy(load_config(ROOT/'configs/dvpa_fmnist_trainpublic_weight10_seed42.yaml')['base'])
    cfg.update(method=method,clients=6,rounds=4,local_epochs=1,batch_size=16,transition_round=3,warmup_rounds=1,device='cpu')
    cfg=validate_config(cfg)
    assert cfg['method']==method and public_training_data_config(cfg)['method']=='JOINT'
    original=copy.deepcopy(cfg);original.pop('public_train_file');original.pop('public_train_sha256')
    dimensions=dict(conv1=832,conv2=51264,fc1=524800,fc=5130)
    native,old=preflight(cfg,dimensions),preflight(original,dimensions)
    assert [p.as_dict() for p in native['plans']]==[p.as_dict() for p in old['plans']]
    rates=[p.group_rates for p in native['plans']]
    if method=='IDP_FEDAVG':assert all(np.allclose(v,[.9]*3) for v in rates)
    else:assert np.allclose(rates[0],[.5,.6,.7]) and np.allclose(rates[-1],[.9]*3)
    calls=[]
    def zero_update(self,global_model,registry,return_update=True):
        assert len(self.data.dataset)==59000
        calls.append(1)
        for _ in range(self.cfg['local_epochs']):
            self.optimizer.step();self.scheduler.step();self.epochs_completed+=1
        return {n:torch.zeros(d,device=self.device) for n,d in registry.dimensions.items()} if return_update else None
    monkeypatch.setattr(PersistentSourceClient,'train',zero_update)
    out=tmp_path/method
    result=run(cfg,out)
    assert len(calls)==24 and result['evaluation_examples']==10000 and result['client_training_examples']==59000
    assert result['method']==method and audit(out)['status']=='passed'
    parts=json.loads((out/'partition.json').read_text(encoding='utf-8'))['parts']
    assert sorted(v for p in parts for v in p)==list(range(59000))
    path=out/'evaluation_protocol.json';data=json.loads(path.read_text(encoding='utf-8'))
    data['final_evaluation_examples']=9000;path.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='evaluation protocol'):audit(out)
