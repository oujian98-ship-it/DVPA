"""Frozen six-job plan, peak aggregation, queue failure behavior."""
import copy
import json
import numpy as np
import pytest
from scripts import run_train_public_three_seeds as suite
from src.dvpa_config import load_config


def test_requested_six_jobs_preserve_protocol_and_only_change_seed():
    rows=suite.jobs();reference=load_config(suite.DVPA/'config.yaml')
    assert [(r['seed'],r['label']) for r in rows]==[(s,l) for s in (43,44) for l in suite.LABELS]
    assert len({r['name'] for r in rows})==6
    for r in rows:
        cfg=copy.deepcopy(r['config'])
        if r['label']=='DVPA':
            assert cfg['controller']['information_value_weight']==10
            cfg['base']['seed']=42;assert cfg==reference
        else:
            cfg['seed']=42;cfg['method']='JOINT';assert cfg==reference['base']
    assert len({r['config']['base']['public_train_sha256'] if r['label']=='DVPA' else r['config']['public_train_sha256'] for r in rows})==1


def test_statistics_use_peaks_and_sample_sd_and_reject_duplicates():
    rows=[dict(seed=s,label='DVPA',summary=dict(best_test_accuracy=v,final_test_accuracy=10.0))
          for s,v in zip((42,43,44),(68.0,70.0,73.0))]
    result=suite.statistics(rows)[0]
    assert result['n']==3 and result['complete_seed_set']
    assert result['peak_accuracy_mean']==pytest.approx(np.mean([68,70,73]))
    assert result['peak_accuracy_sample_std']==pytest.approx(np.std([68,70,73],ddof=1))
    assert suite.statistics(rows[:1])[0]['peak_accuracy_sample_std'] is None
    with pytest.raises(ValueError,match='Duplicate'):suite.statistics(rows+rows[:1])


def test_queue_stops_on_failure_keeps_seed42_and_verified_first_job(tmp_path,monkeypatch):
    monkeypatch.setattr(suite,'report',lambda *_:None)
    reused=[dict(seed=42,label=l,reused=True) for l in suite.LABELS]
    calls=[]
    def execute(job,out,expected):
        calls.append(job['name'])
        if len(calls)==2:raise RuntimeError('engineering failure')
        return dict(seed=job['seed'],label=job['label'],name=job['name'],reused=False,summary=dict(best_test_accuracy=1.0))
    out=tmp_path/'queue'
    with pytest.raises(RuntimeError,match='engineering failure'):
        suite.queue(out,suite.jobs(),reused,{},dict(status='engineering'),execute=execute)
    state=json.loads((out/'suite_state.json').read_text(encoding='utf-8'))
    assert calls==['DVPA_seed43','IDP_FedAvg_seed43']
    assert state['status']=='failed' and len(state['completed'])==4
    assert state['completed'][:3]==reused and state['current_job']=='IDP_FedAvg_seed43'
