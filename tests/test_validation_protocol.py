import hashlib
import json
import numpy as np
import pytest
import copy
from src.p1_validation import check_protocol,check_round


def test_feedback_protocol_locks_methods_equal_clipping_and_seed(tmp_path):
    from src.p3_freezing import validate_feedback_run
    from src.dp_feedback_freezer import DEFAULTS
    archive=tmp_path/'privacy.npz';np.savez(archive,orders=[2.,3.])
    path=tmp_path/'protocol.json'
    methods={'static':dict(mode='BU',freeze_rounds=[9,17]),
             'ema':dict(mode='ADAPTIVE',freeze_rounds=[],feedback=dict(DEFAULTS))}
    protocol=dict(kind='controlled_feedback_comparison_v1',fixed_config=dict(rounds=25,allocation={'layer_clip':'equal_active'}),
                  methods=methods,training_seeds=[42],source_sha256={},privacy_preflight=archive.name,
                  privacy_preflight_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    path.write_text(json.dumps(protocol))
    for method in methods.values():
        cfg=dict(protocol['fixed_config'],**method,seed=42,mask_seed=42,validation_protocol=str(path))
        validate_feedback_run(cfg)
        with check_protocol(cfg) as result:assert len(result['orders'])==2
    cfg=copy.deepcopy(cfg)
    changes=[dict(seed=43,mask_seed=43),dict(mask_seed=1),dict(freeze_rounds=[9]),
             dict(allocation={'layer_clip':'param_count'}),dict(feedback={**DEFAULTS,'patience':99})]
    for change in changes:
        with pytest.raises(ValueError):check_protocol({**cfg,**change})
    protocol['kind']='controlled_p1_comparison_v1';path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='Static P1'):validate_feedback_run(cfg)


def test_diagnostic_protocol_locks_guards_and_secret_state(tmp_path):
    from scripts.validate_freezing_diagnostics import configurations
    import yaml
    secret=tmp_path/'secret.json';secret.write_text('{}')
    fixed,methods=configurations(yaml.safe_load(open('configs/p3_ema.yaml')),secret)
    archive=tmp_path/'privacy.npz';np.savez(archive,orders=[2.,3.])
    path=tmp_path/'protocol.json'
    p=dict(kind='controlled_freezing_diagnostics_v1',data_scope='public_benchmark',fixed_config=fixed,
           methods=methods,training_seeds=[42],source_sha256={},privacy_preflight=archive.name,
           privacy_preflight_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
           d1_rng_sha256=hashlib.sha256(secret.read_bytes()).hexdigest())
    path.write_text(json.dumps(p))
    from src.p3_freezing import validate_feedback_run
    for method in methods.values():
        cfg=dict(fixed,**method,seed=42,mask_seed=42,validation_protocol=str(path))
        validate_feedback_run(cfg)
        with check_protocol(cfg) as a:assert len(a['orders'])==2
    changed=copy.deepcopy(cfg);changed['diagnostic_freezing']['min_freeze_round']=2
    with pytest.raises(ValueError,match='Unregistered diagnostic'):check_protocol(changed)
    cfg=dict(fixed,**methods['D1_STATIC_BU'],seed=42,mask_seed=42,validation_protocol=str(path))
    secret.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='random state changed'):check_protocol(cfg)

def test_locked_protocol_rejects_configuration_and_source_drift(tmp_path):
    archive=tmp_path/'privacy.npz';np.savez(archive,orders=[2.,3.])
    source=tmp_path/'source.py';source.write_text('value=1')
    path=tmp_path/'protocol.json'
    protocol=dict(kind='controlled_p1_comparison_v1',fixed_config={'rounds':25},training_seeds=[42],random_mask_seeds=[11],source_sha256={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},privacy_preflight=archive.name,privacy_preflight_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    path.write_text(json.dumps(protocol))
    cfg=dict(rounds=25,seed=42,mode='NF',mask_seed=42,validation_protocol=str(path))
    assert len(check_protocol(cfg)['orders'])==2
    with pytest.raises(ValueError,match='configuration'):check_protocol(dict(cfg,rounds=3))
    with pytest.raises(ValueError,match='seed'):check_protocol(dict(cfg,seed=99))
    source.write_text('value=2')
    with pytest.raises(ValueError,match='Source changed'):check_protocol(cfg)

def test_round_contract_checks_entire_rdp_vector():
    from types import SimpleNamespace
    contract=dict(orders=np.array([2.,3.]),targets=np.array([10.]),rdp=np.array([[[.1,.2]]]),params=np.array([[[.5,2.,1.]]]))
    c=SimpleNamespace(alphas=[2.,3.],rdp_eps_spent=[.1,.2])
    check_round(contract,1,[c],[10.],[.5],[2.],[1.])
    c.rdp_eps_spent=[.1,.2001]
    with pytest.raises(AssertionError):check_round(contract,1,[c],[10.],[.5],[2.],[1.])


def test_nf_allocation_protocol_only_allows_registered_prior_without_freezing(tmp_path):
    archive=tmp_path/'privacy.npz';np.savez(archive,orders=[2.,3.])
    path=tmp_path/'protocol.json'
    protocol=dict(kind='controlled_allocation_comparison_v1',fixed_config=dict(rounds=25,mode='NF',freeze_rounds=[]),
                  training_seeds=[42],allocations={'equal':{'layer_clip':'equal_active'}},source_sha256={},
                  privacy_preflight=archive.name,privacy_preflight_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    path.write_text(json.dumps(protocol))
    cfg=dict(protocol['fixed_config'],seed=42,mask_seed=42,allocation={'layer_clip':'equal_active'},validation_protocol=str(path))
    assert len(check_protocol(cfg)['orders'])==2
    for change in [dict(allocation={'layer_clip':'param_count'}),dict(mask_seed=1),dict(mode='BU'),dict(freeze_rounds=[9])]:
        with pytest.raises(ValueError):check_protocol({**cfg,**change})


def test_bu_allocation_protocol_locks_freeze_schedule_and_single_seed(tmp_path):
    archive=tmp_path/'privacy.npz';np.savez(archive,orders=[2.,3.])
    path=tmp_path/'protocol.json'
    protocol=dict(kind='controlled_bu_allocation_comparison_v1',fixed_config=dict(rounds=25,mode='BU',freeze_rounds=[9,17]),
                  training_seeds=[42],allocations={'equal':{'layer_clip':'equal_active'}},source_sha256={},
                  privacy_preflight=archive.name,privacy_preflight_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    path.write_text(json.dumps(protocol))
    cfg=dict(protocol['fixed_config'],seed=42,mask_seed=42,allocation={'layer_clip':'equal_active'},validation_protocol=str(path))
    assert len(check_protocol(cfg)['orders'])==2
    for change in [dict(seed=43,mask_seed=43),dict(mode='NF'),dict(freeze_rounds=[]),dict(freeze_rounds=[10,18]),dict(allocation={'layer_clip':'param_count'})]:
        with pytest.raises(ValueError):check_protocol({**cfg,**change})
