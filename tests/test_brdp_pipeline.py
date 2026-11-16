import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.brdp_audit import audit
from src.brdp_budget import DirectionalReleasePlan, LayerMechanismPlan
from src.brdp_config import validate_config
from src.brdp_direction import DirectionState, direction_hash
from src.brdp_mechanism import release_directional
from src.brdp_rdp import calibrate
from src.brdp_runner import make_components, run
from src.st_runner import round_inputs


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def config():
    return yaml.safe_load((Path(__file__).parents[1]/'configs/brdp_synthetic_smoke.yaml').read_text(encoding='utf-8'))


def test_direction_is_strictly_delayed_and_clones_state():
    state = DirectionState({'a': 2}, 0.)
    assert state.for_round(1)['a'] is None
    with pytest.raises(ValueError):
        state.for_round(2)
    state.observe(1, {'a': torch.tensor([3., 4.])})
    u = state.for_round(2)['a']
    assert u.tolist() == pytest.approx([.6, .8])
    u.zero_()
    assert state.for_round(2)['a'].norm() == pytest.approx(1.)
    with pytest.raises(ValueError):
        state.observe(1, {'a': torch.ones(2)})


def test_noise_reconstruction_preserves_exact_reference_projection(monkeypatch):
    u = torch.tensor([.6, .8], dtype=torch.float64)
    certificate = calibrate(3.725, .25, 2, .5, .01, 1.)
    layer = LayerMechanismPlan('a', True, 2, .25, 1., 1, direction_hash(u), certificate, 'test')
    plan = DirectionalReleasePlan(2, (layer,))
    raw = torch.tensor([.2, -.1], dtype=torch.float64)
    monkeypatch.setattr('src.brdp_mechanism.sample_scalar_noise', lambda *args: .3)
    published = release_directional({'a': raw}, plan, {'a': u},
        np.random.default_rng(42), torch.Generator().manual_seed(51))['a']
    assert float(torch.dot(published-raw, u)) == pytest.approx(.3, abs=1e-12)
    g = torch.randn(2, generator=torch.Generator().manual_seed(51), dtype=torch.float64)*certificate.sigma_orth
    assert torch.allclose(published-raw, g-torch.dot(g, u)*u+.3*u)


def test_predictable_plan_and_direction_integrity(monkeypatch):
    cfg = validate_config(config())
    _, registry, ledger, directions, allocator = make_components(cfg)
    active, radii, denom = round_inputs(cfg, registry.names, cfg['clipping']['weights'], cfg['freezing']['freeze_rounds'], 1)
    u = directions.for_round(1)
    p1 = allocator.plan(1, active, ledger.remaining, radii, denom, u)
    assert all(p.certificate.recycle_prob == 0 for p in p1.layers)
    ledger.accept(p1)
    observed = {g: torch.ones(d) for g, d in registry.dimensions.items()}
    directions.observe(1, observed)
    allocator.observe(1, observed, p1.expected_noise_energy)
    u = directions.for_round(2)
    active, radii, denom = round_inputs(cfg, registry.names, cfg['clipping']['weights'], cfg['freezing']['freeze_rounds'], 2)
    p2 = allocator.plan(2, active, ledger.remaining, radii, denom, u)
    assert all(p.certificate.recycle_prob > 0 for p in p2.layers)
    ledger.accept(p2)
    private = {g: torch.zeros(d) for g, d in registry.dimensions.items()}
    released = release_directional(private, p2, u, np.random.default_rng(42), torch.Generator().manual_seed(43))
    assert all(torch.count_nonzero(x) for x in released.values())
    changed = {g: -v for g, v in u.items()}
    with pytest.raises(ValueError, match='Direction changed'):
        release_directional(private, p2, changed, np.random.default_rng(42), torch.Generator().manual_seed(43))
    assert all(p.direction_source_round == 1 and p.direction_hash == direction_hash(u[p.layer]) for p in p2.layers)


@pytest.mark.parametrize('feedback', [False, True])
def test_full_training_replay_and_parameter_tamper_detection(tmp_path, feedback):
    cfg = config()
    cfg['budget']['feedback'] = feedback
    output = tmp_path/'run'
    result = run(cfg, output)
    assert result['non_gaussian_layer_releases'] > 0
    assert result['epsilon_guarantee'] == cfg['privacy']['epsilon']
    assert result['rdp_charged_total'] <= result['rdp_filter_cap']
    assert audit(output)['status'] == 'passed'
    safe_summary = json.dumps(result)
    assert 'selected_clients' not in safe_summary and 'sampling_seed' not in safe_summary
    with (output/'layer_mechanisms.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    row = next(x for x in rows if float(x['recycle_prob']) > 0)
    row['recycle_prob'] = str(float(row['recycle_prob'])*2)
    with (output/'layer_mechanisms.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match='Audit mismatch'):
        audit(output)


def test_no_selected_clients_still_release_charge_and_audit(tmp_path, monkeypatch):
    cfg = config()
    class NoSelection:
        def random(self, n):
            return np.ones(n)
    monkeypatch.setattr('src.brdp_runner._private_rngs', lambda device:
        (NoSelection(), np.random.default_rng(12), torch.Generator(device=device).manual_seed(31)))
    def forbidden(*args, **kwargs):
        raise AssertionError('No client should train')
    monkeypatch.setattr('src.brdp_runner.local_update', forbidden)
    result = run(cfg, tmp_path/'empty')
    assert result['rdp_charged_total'] > 0 and result['non_gaussian_layer_releases'] > 0
    assert audit(tmp_path/'empty')['status'] == 'passed'


def test_public_selection_can_honestly_fall_back_to_gaussian(tmp_path):
    cfg = config()
    cfg['mechanism']['selection']['policy'] = 'public_offline_calibration'
    result = run(cfg, tmp_path/'fallback')
    assert result['non_gaussian_layer_releases'] == 0
    assert audit(tmp_path/'fallback')['status'] == 'passed'


@pytest.mark.parametrize('section,key,value', [('accounting', 'bound_mode', 'grid_maximum'),
    ('accounting', 'sampling_amplification', True), ('output', 'save_noise_or_rng_state', True),
    ('direction', 'source', 'current_raw_update'), ('noise', 'isotropic_base', False)])
def test_unsupported_or_unsafe_modes_rejected(section, key, value):
    cfg = config()
    cfg['mechanism'][section][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_replace_client_sensitivity_and_noise_double():
    a, b = validate_config(config()), validate_config(config())
    b['privacy']['adjacency'] = 'replace_client'
    plans = []
    for cfg in [a, b]:
        _, reg, ledger, state, allocator = make_components(cfg)
        active, radii, denom = round_inputs(cfg, reg.names, cfg['clipping']['weights'], cfg['freezing']['freeze_rounds'], 1)
        plans.append(allocator.plan(1, active, ledger.remaining, radii, denom, state.for_round(1)))
    assert plans[1].layers[0].certificate.sigma_parallel == pytest.approx(2*plans[0].layers[0].certificate.sigma_parallel)
