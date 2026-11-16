import copy
import csv
import hashlib
import math
from pathlib import Path

import pytest
import torch
import yaml

from src.st_accountant import GaussianRDPFilter, gaussian_rho
from src.st_runner import make_components
from src.tap_baseline import AUTHOR_CLIENT, AUTHOR_ROOT, author_client, audit_baseline, release_plan, run_baseline


def config():
    cfg = yaml.safe_load((Path(__file__).parents[1]/'configs/st_budget_fmnist.yaml').read_text())
    cfg['data'].update(dataset='synthetic', clients=3, limit_train=12, limit_test=8)
    cfg['model']['name'] = 'prototype_cnn'
    cfg['train'].update(rounds=3, batch_size=4, local_epochs_or_iterations=1, device='cpu')
    cfg['budget']['time_weights'] = [1., 2., 3.]
    cfg['freezing']['freeze_rounds'] = dict(conv1=2, conv2=3)
    return cfg


def public_test_schedule(cfg):
    ledger = GaussianRDPFilter(cfg['privacy']['epsilon'], cfg['privacy']['delta'])
    sigmas = [3., 2., 1.]
    scale = math.sqrt(sum(.5/s**2 for s in sigmas)/(ledger.rho_cap*(1-1e-10)))
    return dict(rows=[dict(round=t, q=.5 if t<2 else .9, source_sigma=s,
        common_accountant_sigma=s*scale) for t,s in enumerate(sigmas,1)], sigma_scale=scale,
        source_sha256=hashlib.sha256((AUTHOR_ROOT/AUTHOR_CLIENT).read_bytes()).hexdigest(),
        q_save=.5, transition_round=2)


def test_executes_author_privacy_method_with_original_arguments(monkeypatch):
    captured = []
    monkeypatch.setattr('src.tap_baseline._noise', lambda items: captured.append(dict(items)) or 3.)
    cls = author_client()
    client = cls.__new__(cls)
    client.dp_eps_budget, client.dp_eps_spent, client.delta, client.rounds = 10., 2., 1e-5, 5
    assert client.set_local_sigma_and_sample_rate(.9, 2) == 3.
    assert captured == [dict(target_epsilon=8., target_delta=1e-5, sample_rate=.9, epochs=3)]


def test_joint_sensitivity_and_total_cost_match_both_masks():
    torch.set_num_threads(1)
    cfg = config()
    schedule = public_test_schedule(cfg)
    costs = []
    for frozen in (False, True):
        c = copy.deepcopy(cfg)
        if not frozen:
            c['freezing']['freeze_rounds'] = {}
        _, registry, _, clipping, events, ledger = make_components(c)
        total = 0.
        for row in schedule['rows']:
            active, denominator, plan = release_plan(c, registry, clipping, events, row, ledger)
            cost = gaussian_rho(plan.sensitivity, plan.noise_std)
            assert cost == pytest.approx(.5/row['common_accountant_sigma']**2)
            assert sum((plan.clip_radius[g])**2 for g in active) == pytest.approx(.1**2)
            assert len({plan.noise_std[g] for g in active if active[g]}) == 1
            if frozen and row['round'] >= 2:
                assert plan.noise_std['conv1'] == 0
            ledger.record(row['round'], plan.sensitivity, plan.noise_std)
            total += cost
        costs.append(total)
    assert costs[0] == pytest.approx(costs[1])
    assert costs[0] == pytest.approx(ledger.rho_cap)


@pytest.mark.parametrize('variant', ['TAP_NF', 'TAP_SAME_FREEZE'])
def test_training_audit_and_noise_tampering(tmp_path, variant):
    cfg = config()
    schedule = public_test_schedule(cfg)
    out = tmp_path/variant
    result = run_baseline(cfg, schedule, out, variant)
    assert result['epsilon_bound'] == 10.
    assert audit_baseline(out)['status'] == 'passed'
    with (out/'layers.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    rows[0]['noise_std'] = str(float(rows[0]['noise_std'])/2)
    with (out/'layers.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        audit_baseline(out)
