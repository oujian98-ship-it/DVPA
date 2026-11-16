"""Train/public/test isolation and the newly connected replay path.

The last test substitutes zero-update clients to check plumbing; it is not an
FL accuracy experiment. Real formal execution uses the original clients.
"""
import copy
import hashlib
import json
import pytest
import torch
from src.dvpa_config import ROOT,load_config,resolve,public_preflight
from src.dvpa_train_public import make_package,read_package,PublicCurvatureProxy,partition_record
from src.brdp_source_protocol import load_datasets,partition_and_groups,PersistentSourceClient


def configuration():
    return load_config(ROOT/'configs/dvpa_fmnist_trainpublic_weight10_seed42.yaml')


def test_public_package_reads_only_training_idx_files(tmp_path,monkeypatch):
    from src import data
    original=data.idx;seen=[]
    def guarded(path,offset):
        assert 't10k' not in str(path)
        seen.append(str(path));return original(path,offset)
    monkeypatch.setattr(data,'idx',guarded)
    cfg=configuration();path=tmp_path/'public.pt'
    package=make_package(path,cfg['base']['data_root'])
    assert len(seen)==2
    assert len(package['public_source_indices'])==1000 and len(package['private_source_indices'])==59000
    assert package['indices']['calibration'].numel()==64
    assert package['indices']['validation'].numel()+package['indices']['confirmation'].numel()==936
    assert not set(package['public_source_indices'].tolist())&set(package['private_source_indices'].tolist())


def test_clients_use_exact_remaining_59000_and_test_stays_full(tmp_path):
    torch.set_num_threads(1)
    cfg=configuration();base=copy.deepcopy(cfg['base'])
    full=copy.deepcopy(base);full.pop('public_train_file');full.pop('public_train_sha256')
    original,test=load_datasets(full)
    private,full_test=load_datasets(base)
    _,_,package=read_package(base)
    ids=package['private_source_indices']
    assert len(private)==59000 and len(full_test)==10000
    assert torch.equal(private.tensors[0],original.tensors[0][ids])
    assert torch.equal(private.tensors[1],original.tensors[1][ids])
    assert torch.equal(full_test.tensors[0],test.tensors[0]) and torch.equal(full_test.tensors[1],test.tensors[1])
    parts,_=partition_and_groups(base,private.tensors[1])
    assert sorted(i for part in parts for i in part)==list(range(59000))
    assert not set(ids[torch.tensor([i for part in parts for i in part])].tolist())&set(package['public_source_indices'].tolist())
    proxy=PublicCurvatureProxy(cfg)
    assert len(proxy.images)==64 and proxy.final_indices==list(range(10000))
    damaged=copy.deepcopy(package);damaged['private_source_indices'][0]=damaged['public_source_indices'][0]
    path=tmp_path/'damaged.pt';torch.save(damaged,path)
    bad=copy.deepcopy(base);bad.update(public_train_file=str(path.relative_to(ROOT)),public_train_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='disjoint'):read_package(bad)
    # Data population size does not silently alter the client-level accountant.
    legacy=copy.deepcopy(cfg);legacy['base']=full
    p,q=public_preflight(cfg),public_preflight(legacy)
    assert p.caps.tolist()==q.caps.tolist() and p.orders.tolist()==q.orders.tolist()
    assert p.z0==q.z0 and p.radii.tolist()==q.radii.tolist()


@pytest.mark.parametrize('change',['missing_sha','test_public_mix','synthetic'])
def test_incompatible_or_unpinned_data_protocol_rejected(change):
    cfg=configuration()
    if change=='missing_sha':cfg['base'].pop('public_train_sha256')
    elif change=='test_public_mix':
        cfg['base'].update(evaluation_public_file='data/public_fmnist_proxy.pt',evaluation_public_sha256='0'*64)
    else:cfg['base']['dataset_mode']='synthetic'
    with pytest.raises(ValueError):resolve(cfg)


def test_new_protocol_pipeline_full_test_and_replay(tmp_path,monkeypatch):
    from src.dvpa_runner import run
    from src.dvpa_audit import audit
    cfg=configuration()
    cfg['base'].update(clients=6,rounds=4,local_epochs=1,batch_size=16,transition_round=3,warmup_rounds=1,device='cpu')
    cfg['feedback']['warmup_rounds']=1
    cfg['controller'].update(virtual_observations=8,quadrature_nodes=8,simplex_units=2,time_multipliers=[0.5,1,2],
                             ranking_coarse_repeats=2,ranking_repeats=2,ranking_samples=16,ranking_max_samples=32,
                             ranking_quadrature_nodes=12,ranking_validation_nodes=24)
    calls=[]
    def zero_update(self,global_model,registry,return_update=True):
        assert len(self.data.dataset)==59000
        calls.append(self.cfg['local_epochs'])
        for _ in range(self.cfg['local_epochs']):
            self.optimizer.step();self.scheduler.step();self.epochs_completed+=1
        return {n:torch.zeros(d,device=self.device) for n,d in registry.dimensions.items()} if return_update else None
    monkeypatch.setattr(PersistentSourceClient,'train',zero_update)
    out=tmp_path/'engineering_zero_clients'
    summary=run(resolve(cfg),out)
    assert len(calls)==24 and summary['evaluation_examples']==10000 and summary['client_training_examples']==59000
    assert audit(out)['status']=='passed'
    path=out/'data_partition_protocol.json'
    record=json.loads(path.read_text(encoding='utf-8'))
    record['private_source_indices'][0]=record['public_source_indices'][0]
    path.write_text(json.dumps(record),encoding='utf-8')
    with pytest.raises(ValueError,match='training_public_split'):audit(out)
