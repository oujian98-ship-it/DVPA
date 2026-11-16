import copy
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize
import torch

from src.dvpa_config import load_config, public_preflight, smoke_config
from src.dvpa_controller import (
    DecisionValueController,
    allocation_risk,
    spatial_optimum,
)
from src.dvpa_feedback import DPHistoryFeedback
from src.dvpa_filter import GroupRDPFilter


@pytest.fixture(scope="module")
def protocol():
    torch.set_num_threads(1)
    cfg = smoke_config(
        load_config(Path(__file__).parents[1] / "configs/dvpa_fmnist.yaml")
    )
    return public_preflight(cfg)


def warmed(protocol, release_value=0.01):
    p = copy.deepcopy(protocol)
    feedback = DPHistoryFeedback(p)
    ledger = GroupRDPFilter(p)
    feedback.prepare(1)
    a, _ = DecisionValueController(p).choose(1, feedback, ledger.remaining())
    cert = ledger.certify(a)
    ledger.reserve(cert)
    released = {n: torch.full((d,), release_value) for n, d in p.dimensions.items()}
    feedback.observe(1, released, cert)
    feedback.prepare(2)
    return p, feedback, ledger


@pytest.mark.parametrize(
    "costs",
    [[1.0, 2.0, 3.0, 4.0], [832.0, 51264.0, 524800.0, 5130.0], [1e-4, 1e3, 2.0, 5.0]],
)
def test_waterfill_matches_constrained_numerical_optimum(costs):
    k = np.array(costs)
    floor = 0.02
    got = spatial_optimum(k, floor)
    expected = minimize(
        lambda x: np.sum(k / x) / k.sum(),
        np.full(4, 0.25),
        method="SLSQP",
        bounds=[(floor, 1)] * 4,
        constraints={"type": "eq", "fun": lambda x: x.sum() - 1},
        options={"ftol": 1e-12, "maxiter": 500},
    )
    assert expected.success
    assert np.sum(k / got) == pytest.approx(expected.fun * k.sum(), rel=1e-7)
    assert np.min(got) >= floor - 1e-13 and got.sum() == pytest.approx(1.0)


def test_more_precision_really_changes_virtual_posterior(protocol):
    p, f, _ = warmed(protocol)
    normals = np.array([[1.0, -1.0, 0.5, -0.5], [-1.0, 1.0, -0.5, 0.5]])
    low, plo, _ = f.posterior(2, p.z0 * 0.25, [0.25] * 4, standard_normals=normals)
    high, phi, _ = f.posterior(2, p.z0 * 3, [0.25] * 4, standard_normals=normals)
    assert np.trace(phi) < np.trace(plo)
    assert not np.allclose(low, high)
    assert not np.allclose(
        f.dp_covariance(p.z0, [0.25] * 4), f.dp_covariance(p.z0, [0.1, 0.2, 0.3, 0.4])
    )


def test_known_cost_zero_information_and_final_round(protocol):
    p, f, ledger = warmed(protocol)
    p.config["feedback"]["lambda1"] = 0.0
    action, log = DecisionValueController(p).choose(2, f, ledger.remaining())
    assert abs(log["information_value"]) < 1e-8
    # Final-stage information must vanish even with uncertain states.
    f.prepared_round = p.rounds
    f.last_observed = p.rounds - 1
    _, last = DecisionValueController(p).choose(p.rounds, f, ledger.remaining())
    assert last["information_value"] == 0.0 and last["tail_loss"] == 0.0


def test_information_about_common_scale_has_zero_spatial_value():
    states = np.array([[1.0, 4.0, 9.0, 16.0], [9.0, 36.0, 81.0, 144.0]])
    prior = states.mean(axis=0)
    assert allocation_risk(prior, 0.02) == pytest.approx(
        allocation_risk(states, 0.02).mean(), abs=1e-10
    )
    changed = np.array([[9.0, 1.0, 1.0, 1.0], [1.0, 9.0, 1.0, 1.0]])
    assert (
        allocation_risk(changed.mean(axis=0), 0.02)
        > allocation_risk(changed, 0.02).mean()
    )


def test_full_tail_each_profile_feasible_same_balance(protocol):
    p, f, ledger = warmed(protocol)
    c = DecisionValueController(p)
    z = p.z0 * 0.5
    balance = ledger.remaining() - p.charges(2, z)
    future, plans = c.tail_plans(2, balance)
    assert plans.shape == (5, 2)
    assert np.all(plans >= p.z_floor)
    assert np.all(p.charges(future[None, :], plans).sum(axis=1) <= balance)
    before = ledger.state_dict()
    a, log = c.choose(2, f, ledger.remaining())
    assert ledger.state_dict() == before  # virtual observations query no private data
    assert np.all(
        p.charges(2, a.precision) + p.charges([3, 4], p.z_floor).sum(axis=0)
        <= ledger.remaining()
    )
    assert log["candidates"] > 1


def test_belief_changes_time_and_layer_actions(protocol):
    p, f, ledger = warmed(protocol)
    p.config["feedback"]["lambda1"] = 100.0
    f.pred_cov = np.eye(8) * 1e-6
    f.pred_mean = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    a, _ = DecisionValueController(p).choose(2, f, ledger.remaining())
    f.pred_mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0])
    b, _ = DecisionValueController(p).choose(2, f, ledger.remaining())
    assert a.precision != pytest.approx(b.precision, rel=1e-5, abs=0.0)
    assert not np.allclose(
        [v for _, v in a.layer_precision], [v for _, v in b.layer_precision]
    )


def test_fixed_basis_no_warmup_reuse_and_stage_change(protocol):
    p, f, ledger = warmed(protocol)
    assert f.observation_rows == []
    assert f.mean == pytest.approx(np.zeros(8))
    old = {n: u.clone() for n, u in f.basis.items()}
    a, _ = DecisionValueController(p).choose(2, f, ledger.remaining())
    cert = ledger.certify(a)
    ledger.reserve(cert)
    released = {n: torch.full((d,), 0.02) for n, d in p.dimensions.items()}
    f.observe(2, released, cert)
    assert len(f.observation_rows) == 4
    with pytest.raises(ValueError):
        f.observe(2, released, cert)
    f.prepare(3)
    assert f.indices(2).tolist() == [0, 1, 2, 3]
    assert f.indices(3).tolist() == [4, 5, 6, 7]
    assert all(torch.equal(u, f.basis[n]) for n, u in old.items())
    assert f.cov[0, 4] != 0.0


@pytest.mark.parametrize("mode", ["dp_innovation", "public_bound"])
def test_sampling_noise_updated_only_after_release(protocol, mode):
    p, f, ledger = warmed(protocol)
    f.cfg["extra_covariance"] = mode
    before = f.extra.copy()
    a, _ = DecisionValueController(p).choose(2, f, ledger.remaining())
    assert np.array_equal(f.extra, before)
    cert = ledger.certify(a)
    f.observe(2, {n: torch.zeros(d) for n, d in p.dimensions.items()}, cert)
    if mode == "public_bound":
        assert np.array_equal(before, f.extra)
    else:
        assert not np.array_equal(before[0], f.extra[0])
        assert np.array_equal(before[1], f.extra[1])
    assert np.linalg.eigvalsh(f.extra[0]).min() >= -1e-10


def test_history_hash_prevents_same_round_leakage(protocol):
    p, f, ledger = warmed(protocol)
    c = DecisionValueController(p)
    a, _ = c.choose(2, f, ledger.remaining())
    from dataclasses import replace

    cert = ledger.certify(replace(a, history_hash="future or unrecorded raw signal"))
    with pytest.raises(ValueError):
        f.observe(2, {n: torch.zeros(d) for n, d in p.dimensions.items()}, cert)
    with pytest.raises(ValueError):
        f.prepare(3)


def test_empirical_jensen_removes_tower_noise_and_keeps_relative_information(protocol):
    c = DecisionValueController(protocol)
    plans = np.array([[1.0, 1.0], [1.5, 0.5]])
    # Even a badly imbalanced empirical distribution of a common scale contains
    # zero allocation information at fixed future profiles.
    common = np.array([1.0, 4.0, 9.0, 16.0])
    posterior = (
        np.array([0.5, 1.0, 2.0, 8.0])[:, None, None]
        * np.stack([common, common])[None, :, :]
    )
    prior, _ = c.tail_value_from_costs(np.stack([common, common])[None, :, :], plans)
    after, voi, raw, correction, _ = c.information_estimate(
        posterior, plans, float(prior[0])
    )
    assert abs(voi) < 1e-10
    assert after == pytest.approx(prior[0])
    assert raw != pytest.approx(0.0) and correction == pytest.approx(raw)
    relative = np.array([[[9.0, 1.0, 1.0, 1.0]] * 2, [[1.0, 9.0, 1.0, 1.0]] * 2])
    before, _ = c.tail_value_from_costs(relative.mean(axis=0, keepdims=True), plans)
    after, voi, _, _, _ = c.information_estimate(relative, plans, float(before[0]))
    assert voi > 0.0 and after < before[0]
