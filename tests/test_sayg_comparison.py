import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from opacus.accountants.analysis import rdp
import pytest
import torch
import yaml

from src.brdp_group_accountant import epsilon_from_rdp
from src.brdp_source_audit import audit
from src.brdp_source_protocol import preflight, validate_config
from src.brdp_source_runner import run
from src.layer_registry import LayerRegistry
from src.sayg_protocol import gaussian_round_costs, source_trace
from src.upstream_runner import source_model


def config():
    cfg = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/brdp_source_smoke.yaml").read_text()
    )
    cfg.update(method="SPEND_AS_YOU_GO", mechanism="gaussian", recycle_prob=0.0)
    return cfg


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def test_source_trace_remainder_and_actual_communication_charges():
    cfg = config()
    trace, _ = source_trace(cfg)
    public = preflight(cfg, LayerRegistry(source_model()("fmnist", 16)).dimensions)
    ledger = np.zeros((3, len(public["orders"])))
    previous = np.zeros(3)
    for t, row in enumerate(trace, 1):
        np.testing.assert_allclose(
            row["remaining_epsilon_before"], np.array(cfg["epsilons"]) - previous
        )
        assert row["get_noise_epochs"] == cfg["rounds"] - t + 1
        assert row["get_noise_implied_steps"] == int(
            (cfg["rounds"] - t + 1) / cfg["q_spend"]
        )
        for g in range(3):
            ledger[g] += rdp.compute_rdp(
                q=row["group_rates"][g],
                noise_multiplier=row["local_sigma"][g],
                steps=1,
                orders=public["orders"],
            )
        previous = np.array(
            [
                epsilon_from_rdp(public["orders"], ledger[g], cfg["delta"])[0]
                for g in range(3)
            ]
        )
        np.testing.assert_allclose(previous, row["source_epsilon_after"], rtol=1e-12)
    assert np.all(ledger <= public["envelope"].sum(axis=0) + 1e-9)


def test_dynamic_harmonic_clipping_and_equal_layer_noise():
    cfg = config()
    dims = LayerRegistry(source_model()("fmnist", 16)).dimensions
    p = preflight(cfg, dims)
    trace = p["schedule_metadata"]["source_trace"]
    assert not np.allclose(p["plans"][0].group_radii, p["plans"][-1].group_radii)
    for row, plan in zip(trace, p["plans"], strict=True):
        assert plan.time_fraction is None
        assert len(set(dict(plan.layer_noise_std).values())) == 1
        assert set(dict(plan.layer_budget_weights).values()) == {0.25}
        assert np.isclose(np.dot(p["counts"] / sum(p["counts"]), plan.group_radii), 250)
        global_sigma = 1 / np.dot(
            p["counts"] / sum(p["counts"]), 1 / np.array(row["local_sigma"])
        )
        np.testing.assert_allclose(
            plan.group_radii, 250 * global_sigma / np.array(row["local_sigma"])
        )
        assert plan.group_rates == tuple(
            cfg["q_save"] if row["round"] < cfg["transition_round"] else [0.9] * 3
        )


def test_matched_brdp_does_not_spend_more_than_original():
    cfg = config()
    dims = LayerRegistry(source_model()("fmnist", 16)).dimensions
    original = preflight(cfg, dims)
    b = copy.deepcopy(cfg)
    b.update(
        method="JOINT",
        mechanism="brdp",
        recycle_prob=0.005,
        calibration_epsilons=original["final_epsilon"],
    )
    matched = preflight(validate_config(b), dims)
    np.testing.assert_allclose(
        matched["final_epsilon"],
        np.array(original["final_epsilon"]) * (1 - 1e-6),
        rtol=1e-10,
    )
    assert np.all(np.array(matched["final_epsilon"]) < original["final_epsilon"])


@pytest.mark.parametrize("bad", [[0, 2, 3], [11, 20, 30], [1, 2], True, [True, 2, 3]])
def test_invalid_matching_targets_rejected(bad):
    cfg = config()
    cfg.update(method="JOINT", calibration_epsilons=bad)
    with pytest.raises(ValueError, match="Calibration targets"):
        validate_config(cfg)


def test_original_method_rejects_non_gaussian_or_hidden_extra_target():
    cfg = config()
    cfg["mechanism"] = "brdp"
    with pytest.raises(ValueError, match="Original spend"):
        validate_config(cfg)
    cfg = config()
    cfg["calibration_epsilons"] = [1, 2, 3]
    with pytest.raises(ValueError, match="Original spend"):
        validate_config(cfg)


def test_exact_gaussian_billing_cannot_accept_soft_recycled_noise():
    with pytest.raises(ValueError, match="Gaussian accountant"):
        gaussian_round_costs(
            np.array([2.0]),
            [1.0],
            1.0,
            {"layer": 1.0},
            {"layer": 1.0},
            [0.5],
            {"layer": (0.005, 1.0)},
        )


def test_full_source_baseline_release_audit_and_trace_tamper(tmp_path):
    result = run(config(), tmp_path / "baseline")
    assert result["non_gaussian_layer_releases"] == 0
    assert audit(tmp_path / "baseline")["status"] == "passed"
    p = tmp_path / "baseline/privacy_preflight.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["schedule_metadata"]["source_trace"][0]["remaining_epsilon_before"][0] += 1
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance/recurrence"):
        audit(tmp_path / "baseline")


def test_full_matched_brdp_audit_with_original_caps(tmp_path):
    cfg = config()
    dims = LayerRegistry(source_model()("fmnist", 16)).dimensions
    actual = preflight(cfg, dims)["final_epsilon"]
    cfg.update(
        method="JOINT",
        mechanism="brdp",
        recycle_prob=0.005,
        calibration_epsilons=actual,
    )
    result = run(cfg, tmp_path / "matched")
    assert result["non_gaussian_layer_releases"] == 8
    check = audit(tmp_path / "matched")
    np.testing.assert_allclose(
        check["group_epsilon_guarantees"], np.array(actual) * (1 - 1e-6), rtol=1e-9
    )
    assert [r["epsilon_target"] for r in result["group_privacy"]] == [10, 20, 30]


def test_new_suite_gate_and_path_safety(tmp_path, monkeypatch):
    import scripts.run_sayg_comparison as suite

    monkeypatch.setattr(suite, "ROOT", tmp_path)
    manifest = {}
    for name in suite.REQUIRED:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("tested")
        manifest[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path / "gate.json").write_text(
        json.dumps(dict(status="passed", tests_passed=90, source_sha256=manifest))
    )
    assert suite.verify_gate("gate.json")["status"] == "passed"
    with pytest.raises(ValueError, match="inside BR_DP"):
        suite.inside("../trynew")
    (tmp_path / "src/sayg_protocol.py").write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        suite.verify_gate("gate.json")
