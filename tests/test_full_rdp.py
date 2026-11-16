import numpy as np
import yaml
from src.original_scheduler import PrivacyScheduler
from src.accountant import Ledger

def test_full_rdp_uniform_noise_is_constant_and_budget_exhausted():
    c=yaml.safe_load(open('configs/original_fmnist.yaml'))
    c['data']['clients']=6;c['train']['rounds']=3
    c['privacy']['calibration']='full_rdp_remaining'
    c['privacy']['q_save']=[.9]*3
    s=PrivacyScheduler(c);ledger=Ledger(s.targets,1e-5)
    for t in range(1,4):
        np.testing.assert_allclose(s.params(t).sigma_n,s.params(1).sigma_n,rtol=1e-7)
        ledger.record(s.params(t))
    np.testing.assert_allclose(ledger.epsilons(),s.targets,rtol=1e-7)

def test_full_rdp_saving_noise_decreases_and_no_budget_excess():
    c=yaml.safe_load(open('configs/original_fmnist.yaml'))
    c['data']['clients']=6;c['train']['rounds']=4;c['privacy']['transition_round']=3
    c['privacy']['calibration']='full_rdp_remaining'
    s=PrivacyScheduler(c);ledger=Ledger(s.targets,1e-5)
    for t in range(1,5):ledger.record(s.params(t))
    assert np.all(s.params(3).sigma_n<s.params(1).sigma_n)
    np.testing.assert_allclose(ledger.epsilons(),s.targets,rtol=1e-7)

def test_idp_default_is_uniform_without_opt_in_flag():
    c=yaml.safe_load(open('configs/reference_idp_fedavg_fmnist.yaml'))
    c['data']['clients']=6;c['train']['rounds']=3
    s=PrivacyScheduler(c);ledger=Ledger(s.targets,1e-5)
    for t in range(1,4):
        np.testing.assert_array_equal(s.params(t).sigma_n,s.params(1).sigma_n)
        ledger.record(s.params(t))
    np.testing.assert_allclose(ledger.epsilons(),s.targets,rtol=1e-7)
