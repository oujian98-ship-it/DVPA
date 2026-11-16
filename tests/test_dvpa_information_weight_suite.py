"""Queue/configuration/aggregation checks; no real FL training in these tests."""
import copy
import json
import numpy as np
import pytest
from scripts import run_dvpa_information_weight_suite as suite
from src.dvpa_config import load_config


def test_exact_five_jobs_change_weight_only_and_preserve_transfer_protocol():
    rows=suite.jobs()
    assert [(r['dataset'],r['seed']) for r in rows]==list(suite.REQUESTED)
    assert len({r['name'] for r in rows})==5
    for row in rows:
        cfg=copy.deepcopy(row['config']);assert cfg['controller']['information_value_weight']==10.0
        cfg['controller']['information_value_weight']=1.0
        old=load_config(suite.PRIOR/f"{row['dataset']}_DVPA_seed{row['seed']}"/'config.yaml')
        assert cfg==old
        assert cfg['feedback']['extra_covariance']=='dp_innovation'
        assert cfg['feedback']['covariance_shrinkage']==0.1
        assert cfg['base']['evaluation_public_file']==cfg['feedback']['public_proxy_file']


def test_per_dataset_sample_std_and_duplicate_rejection():
    rows=[dict(dataset='fmnist',seed=s,label=suite.LABEL,summary={'final_test_accuracy':a})
          for s,a in zip([42,43,44],[70.0,71.0,73.0])]
    rows.append(dict(dataset='mnist',seed=42,label=suite.LABEL,summary={'final_test_accuracy':80.0}))
    stats=suite.statistics(rows)
    assert stats[0]['mean']==pytest.approx(214/3)
    assert stats[0]['sample_std']==pytest.approx(np.std([70,71,73],ddof=1))
    assert stats[0]['complete_seed_set'] and stats[1]['sample_std'] is None
    with pytest.raises(ValueError,match='Duplicate'):suite.statistics(rows+[rows[0]])


def test_queue_failure_stops_remaining_jobs_and_retains_completed(tmp_path,monkeypatch):
    monkeypatch.setattr(suite,'report',lambda *_:None)
    calls=[]
    def execute(job,out):
        calls.append(job['name'])
        if len(calls)==2:raise RuntimeError('deliberate engineering failure')
        return dict(name=job['name'],summary={'final_test_accuracy':1.0},reused=False)
    reused=dict(name='reused_fmnist42',reused=True)
    out=tmp_path/'queue'
    with pytest.raises(RuntimeError,match='deliberate engineering failure'):
        suite.queue(out,suite.jobs(),reused,execute=execute)
    state=json.loads((out/'suite_state.json').read_text(encoding='utf-8'))
    assert len(calls)==2 and state['status']=='failed'
    assert state['completed'][0]==reused and len(state['completed'])==2
    assert state['current_job']=='fmnist_DVPA_weight10_seed44'
