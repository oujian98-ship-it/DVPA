import copy
import csv
import math
from pathlib import Path

import pytest
import torch
import yaml

from src.st_accountant import GaussianRDPFilter, gaussian_rho
from src.st_budget import TimeLayerBudget
from src.st_mechanism import clip_blocks, release_sum
from src.st_runner import preview, run, validate_config
from src.st_audit import audit


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def allocator(rounds=2, **kwargs):
    return TimeLayerBudget(['a', 'b'], rounds, kwargs)


def plan_for(controller, ledger, t=1, active=None):
    return controller.plan(t, active or dict(a=True, b=True), ledger.remaining,
                           dict(a=2., b=2.), 4., ledger.adjacency_factor)


def test_noise_really_tracks_budget_and_not_clipping_weight():
    ledger = GaussianRDPFilter(10, 1e-5)
    controller = allocator(1, layer_weights=dict(a=1., b=4.))
    plan = plan_for(controller, ledger)
    assert plan.rho['b'] == pytest.approx(4*plan.rho['a'])
    assert plan.noise_std['b'] == pytest.approx(plan.noise_std['a']/2)
    assert plan.clip_radius['a'] == plan.clip_radius['b']
    assert gaussian_rho(plan.sensitivity, plan.noise_std) == pytest.approx(sum(plan.rho.values()))


def test_whitened_joint_cost_bounds_arbitrary_neighbor_updates():
    ledger = GaussianRDPFilter(4, 1e-5, 'replace_client')
    plan = plan_for(allocator(1, layer_weights=dict(a=3., b=1.)), ledger)
    # Opposite clipped clients attain the replace-client bound in every block.
    x = clip_blocks(dict(a=torch.tensor([20., 0.]), b=torch.tensor([-20.])),
                    dict(a=True, b=True), dict(a=2., b=2.))
    whitened = sum(float(((2*x[g]/4)/plan.noise_std[g]).double().square().sum()) for g in x)
    assert whitened/2 <= sum(plan.rho.values())*(1+1e-6)
    assert whitened/2 == pytest.approx(sum(plan.rho.values()), rel=1e-6)


def test_replace_adjacency_doubles_noise_at_same_budget():
    add = GaussianRDPFilter(10, 1e-5)
    replace = GaussianRDPFilter(10, 1e-5, 'replace_client')
    a = plan_for(allocator(1), add)
    b = plan_for(allocator(1), replace)
    assert b.noise_std['a'] == pytest.approx(2*a.noise_std['a'])
    assert gaussian_rho(a.sensitivity, a.noise_std) == pytest.approx(gaussian_rho(b.sensitivity, b.noise_std))


@pytest.mark.parametrize('epsilon,delta', [(1., 1e-5), (10., 1e-5), (.01, 1e-10), (100., .001)])
def test_full_horizon_composes_to_target(epsilon, delta):
    ledger = GaussianRDPFilter(epsilon, delta)
    controller = allocator(25, time_weights=list(range(1,26)))
    costs = []
    for t in range(1,26):
        plan = plan_for(controller, ledger, t, dict(a=t < 15, b=True))
        row = ledger.record(t, plan.sensitivity, plan.noise_std)
        costs.append(row['rho_increment'])
        assert row['epsilon_bound'] <= epsilon
        assert row['rdp_increment'] == pytest.approx(ledger.alpha*sum(plan.rho.values()))
    assert ledger.epsilon_bound == pytest.approx(epsilon, rel=1e-8)
    assert costs[-1] > costs[0]
    assert sum(costs) == pytest.approx(ledger.spent)


def test_filter_rejects_under_noise_atomically():
    ledger = GaussianRDPFilter(1., 1e-5)
    with pytest.raises(RuntimeError, match='budget exceeded'):
        ledger.record(1, dict(a=1.), dict(a=.001))
    assert ledger.spent == 0 and ledger.history == []
    plan = plan_for(allocator(1), ledger)
    ledger.record(1, plan.sensitivity, plan.noise_std)
    with pytest.raises(ValueError, match='sequential'):
        ledger.record(1, plan.sensitivity, plan.noise_std)


def test_nonfinite_private_update_is_bounded_without_private_error():
    result = clip_blocks(dict(a=torch.tensor([float('nan'), float('inf')]), b=torch.ones(3)*20),
                         dict(a=True, b=True), dict(a=1., b=2.))
    assert torch.count_nonzero(result['a']) == 0
    assert float(result['b'].norm()) <= 2.


@pytest.mark.parametrize('sensitivity,noise', [({'a': 1}, {'a': 0}),
    ({'a': 1}, {'b': 1}), ({'a': float('nan')}, {'a': 1}),
    ({'a': 1}, {'a': float('inf')}), ({'a': -1}, {'a': 1})])
def test_bad_release_rejected(sensitivity, noise):
    with pytest.raises(ValueError):
        gaussian_rho(sensitivity, noise)


def test_freeze_redirects_only_unspent_budget():
    ledger = GaussianRDPFilter(10., 1e-5)
    controller = allocator(2)
    first = plan_for(controller, ledger)
    ledger.record(1, first.sensitivity, first.noise_std)
    spent = ledger.spent
    second = plan_for(controller, ledger, 2, dict(a=False, b=True))
    assert second.rho['a'] == second.noise_std['a'] == second.sensitivity['a'] == 0
    assert second.rho['b'] == pytest.approx(ledger.remaining)
    ledger.record(2, second.sensitivity, second.noise_std)
    assert ledger.spent > spent


def test_empty_sample_still_releases_budgeted_independent_noise():
    ledger = GaussianRDPFilter(10., 1e-5)
    controller = allocator(1, layer_weights=dict(a=1., b=4.))
    plan = plan_for(controller, ledger)
    ledger.record(1, plan.sensitivity, plan.noise_std)
    zeros = dict(a=torch.zeros(60000), b=torch.zeros(60000))
    result = release_sum(zeros, dict(a=True, b=True), plan, 4., torch.Generator().manual_seed(231))
    for g in zeros:
        assert float(result[g].var()) == pytest.approx(plan.noise_std[g]**2, rel=.025)
        assert abs(float(result[g].mean())) < .02*plan.noise_std[g]
    cov = float((result['a']*result['b']).mean())
    assert abs(cov) < .02*plan.noise_std['a']*plan.noise_std['b']


def test_feedback_is_delayed_and_affects_both_axes():
    controller = allocator(3, feedback=True, feedback_ema=0., time_feedback_gain=2.)
    baseline = allocator(3)
    ledgers = [GaussianRDPFilter(10, 1e-5) for _ in range(2)]
    p1 = plan_for(controller, ledgers[0])
    q1 = plan_for(baseline, ledgers[1])
    for ledger, plan in zip(ledgers, [p1, q1]):
        ledger.record(1, plan.sensitivity, plan.noise_std)
    with pytest.raises(ValueError, match='preceding'):
        plan_for(controller, ledgers[0], 2)
    controller.observe(1, dict(a=torch.ones(2)*100, b=torch.zeros(2)), dict(a=1., b=1.))
    p2 = plan_for(controller, ledgers[0], 2)
    q2 = plan_for(baseline, ledgers[1], 2)
    assert sum(p2.rho.values()) > sum(q2.rho.values())
    assert p2.rho['a'] > p2.rho['b']
    with pytest.raises(ValueError):
        controller.observe(1, dict(a=torch.ones(2), b=torch.ones(2)), dict(a=1., b=1.))


def smoke_config():
    cfg = yaml.safe_load((Path(__file__).parents[1]/'configs/st_budget_fmnist.yaml').read_text())
    cfg['data'].update(dataset='synthetic', clients=3, limit_train=12, limit_test=8)
    cfg['model']['name'] = 'prototype_cnn'
    cfg['train'].update(rounds=3, batch_size=4, local_epochs_or_iterations=1, device='cpu')
    cfg['budget']['time_weights'] = [1., 2., 3.]
    cfg['freezing']['freeze_rounds'] = dict(conv1=2, conv2=3)
    return cfg


def test_preview_never_reads_dataset(monkeypatch):
    def forbidden(*args):
        raise AssertionError('Preview accessed data')
    monkeypatch.setattr('src.st_runner._datasets', forbidden)
    cfg = smoke_config()
    cfg['budget']['feedback'] = True
    result = preview(cfg)
    assert len(result['layers']) == 12
    assert 'feedback disabled' in result['scope']
    assert result['epsilon_bound'] <= cfg['privacy']['epsilon']


@pytest.mark.parametrize('feedback', [False, True])
def test_training_and_independent_audit(tmp_path, feedback):
    cfg = smoke_config()
    cfg['budget']['feedback'] = feedback
    out = tmp_path/'run'
    result = run(cfg, out)
    assert result['epsilon_bound'] <= cfg['privacy']['epsilon']
    assert audit(out)['status'] == 'passed'
    with (out/'layers.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert float(next(r for r in rows if r['round']=='2' and r['layer']=='conv1')['noise_std']) == 0
    assert float(next(r for r in rows if r['round']=='3' and r['layer']=='fc2')['rho_allocated']) > 0
    text = (out/'summary.json').read_text()
    assert 'sampling_seed' not in text and 'selected_clients' not in text
    # Tamper with actual noise, keeping the nominal budget label unchanged.
    rows[0]['noise_std'] = str(float(rows[0]['noise_std'])/2)
    with (out/'layers.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises((ValueError, RuntimeError)):
        audit(out)


def test_zero_participation_training_still_debits_and_noises(tmp_path, monkeypatch):
    cfg = smoke_config()
    cfg['privacy']['sample_rate'] = 1e-12
    # Public denominator remains fixed; no normalization by realized count.
    def forbidden(*args):
        raise AssertionError('Unexpected client selected at negligible q')
    monkeypatch.setattr('src.st_runner.local_update', forbidden)
    result = run(cfg, tmp_path/'empty')
    assert result['rho_spent'] > 0
    assert audit(tmp_path/'empty')['status'] == 'passed'


def test_reject_legacy_privacy_flags_and_private_evaluation():
    cfg = smoke_config()
    bad = copy.deepcopy(cfg)
    bad['privacy']['privacy_claim_mode'] = 'local_messages_visible'
    with pytest.raises(ValueError):
        validate_config(bad)
    cfg['data']['public_test'] = False
    with pytest.raises(ValueError, match='public test'):
        validate_config(cfg)


def test_realized_feedback_does_not_minimize_rdp_orders():
    ledger = GaussianRDPFilter(10, 1e-5)
    alpha = ledger.alpha
    controller = allocator(2, feedback=True)
    first = plan_for(controller, ledger)
    ledger.record(1, first.sensitivity, first.noise_std)
    controller.observe(1, dict(a=torch.ones(3)*50, b=torch.zeros(3)), dict(a=1., b=1.))
    second = plan_for(controller, ledger, 2)
    ledger.record(2, second.sensitivity, second.noise_std)
    assert ledger.alpha == alpha
    assert ledger.epsilon_accounting_value == pytest.approx(alpha*ledger.spent+math.log(1e5)/(alpha-1))
    assert ledger.epsilon_bound == ledger.target


def test_observed_partial_charge_is_not_reported_as_adaptive_guarantee():
    ledger = GaussianRDPFilter(10, 1e-5)
    plan = plan_for(allocator(25), ledger)
    row = ledger.record(1, plan.sensitivity, plan.noise_std)
    assert row['epsilon_accounting_value'] < row['epsilon_bound']
    assert row['epsilon_bound'] == 10
    assert row['rdp_filter_cap'] > row['rdp_spent']
