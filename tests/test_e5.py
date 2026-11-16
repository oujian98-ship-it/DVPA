import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from src.e5_analysis import METHODS,factorial_summary
from src.p1_validation import check_protocol
from src.p3_freezing import validate_feedback_run
from scripts.validate_e5_joint import configurations,write_report


def cells():
    return [dict(seed=s,method=m,best_test_acc=v,final_test_acc=v-1) for s in [42,43,44]
            for m,v in zip(METHODS,[60.,62.,61.,66.])]


def test_factorial_contrasts_separate_main_effects_from_interaction():
    r=factorial_summary(cells(),[42,43,44])
    contrasts={x['contrast']:x for x in r['contrasts']['best_test_acc']}
    for name,expected in [('4A_under_static',2),('dynamic_under_equal',1),('dynamic_under_4A',4),('joint_vs_static_equal',6),('interaction',3)]:
        assert contrasts[name]['gaps']==[expected]*3
        assert contrasts[name]['mean']==expected
        assert contrasts[name]['exact_two_sided_sign_flip_p']==.25
    assert len(r['main_table'])==4
    assert all(x['best']['n']==3 for x in r['main_table'])


def test_factorial_rejects_missing_duplicate_unregistered_or_invalid_cells():
    for records in [cells()[:-1],cells()+[cells()[0]],cells()+[dict(seed=99,method=METHODS[0],best_test_acc=60,final_test_acc=59)]]:
        with pytest.raises(ValueError):factorial_summary(records,[42,43,44])
    rows=cells();rows[0]['best_test_acc']=float('nan')
    with pytest.raises(ValueError):factorial_summary(rows,[42,43,44])


def test_joint_protocol_registers_full_method_prior_guards_and_private_entropy(tmp_path):
    fixed,methods=configurations();archive=tmp_path/'privacy.npz';np.savez(archive,orders=[2.,3.])
    registered={};hashes={}
    for name,method in methods.items():
        secret=tmp_path/(name+'.json');secret.write_text(json.dumps(dict(sampling=len(hashes),noise=len(hashes)+10)))
        registered[name]={**method,'diagnostic_rng_path':str(secret)}
        hashes[str(secret)]=hashlib.sha256(secret.read_bytes()).hexdigest()
    path=tmp_path/'protocol.json'
    protocol=dict(kind='controlled_joint_allocation_freezing_v1',data_scope='public_benchmark',fixed_config=fixed,
        methods_by_seed={'42':registered},training_seeds=[42],private_rng_sha256=hashes,source_sha256={},
        privacy_preflight=archive.name,privacy_preflight_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    path.write_text(json.dumps(protocol))
    for method in registered.values():
        cfg=dict(fixed,**method,seed=42,mask_seed=42,validation_protocol=str(path))
        validate_feedback_run(cfg)
        with check_protocol(cfg) as a:assert len(a['orders'])==2
    cfg=copy.deepcopy(cfg)
    changes=[dict(seed=43,mask_seed=43),dict(mask_seed=1),dict(freeze_rounds=[9,17]),
             dict(allocation={'layer_clip':'param_count'}),dict(diagnostic_freezing={**cfg['diagnostic_freezing'],'min_freeze_round':4})]
    for change in changes:
        with pytest.raises(ValueError):check_protocol({**cfg,**change})
    Path(cfg['diagnostic_rng_path']).write_text('changed')
    with pytest.raises(ValueError,match='random state changed'):check_protocol(cfg)


def test_e5_static_and_dynamic_policies_fixed_before_allocation_crossing():
    fixed,methods=configurations()
    for left,right in [('STATIC_EQUAL','STATIC_4A'),('DYNAMIC_EQUAL','DYNAMIC_4A')]:
        assert {k:v for k,v in methods[left].items() if k!='allocation'}=={k:v for k,v in methods[right].items() if k!='allocation'}
    assert methods['STATIC_EQUAL']['freeze_rounds']==[15,17]
    assert methods['DYNAMIC_EQUAL']['diagnostic_freezing']['protected_groups']==['fc1']
    assert methods['DYNAMIC_EQUAL']['feedback']['patience']==5
    methods['DYNAMIC_4A']['allocation']['prior_decay']['fc1']=99
    assert configurations()[1]['DYNAMIC_4A']['allocation']['prior_decay']['fc1']==.05
