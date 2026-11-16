import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.run_three_methods_seeds import METHODS, SEEDS, jobs, stats
from src.brdp_source_audit import audit
from src.brdp_source_protocol import preflight, validate_config
from src.brdp_source_runner import run
from src.layer_registry import LayerRegistry
from src import sayg_protocol
from src.upstream_runner import source_model


def config():
    cfg = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/brdp_source_smoke.yaml").read_text()
    )
    cfg.update(method="IDP_FEDAVG", mechanism="gaussian", recycle_prob=0.0)
    return cfg


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def test_idp_source_calibrates_once_per_group(monkeypatch):
    calls = []
    original = sayg_protocol.checked_source_noise

    def traced(items, orders):
        calls.append(dict(items))
        return original(items, orders)

    monkeypatch.setattr(sayg_protocol, "checked_source_noise", traced)
    sayg_protocol._idp_trace.cache_clear()
    cfg = config()
    rows, _ = sayg_protocol.idp_trace(cfg)
    assert len(calls) == 3
    assert [c["target_epsilon"] for c in calls] == [10, 20, 30]
    assert all(c["epochs"] == 3 and c["sample_rate"] == 0.9 for c in calls)
    assert all(r["group_rates"] == [0.9] * 3 for r in rows)
    assert all(r["local_sigma"] == rows[0]["local_sigma"] for r in rows)


def test_idp_uniform_joint_accounting_and_constant_radii():
    cfg = config()
    p = preflight(
        validate_config(cfg), LayerRegistry(source_model()("fmnist", 16)).dimensions
    )
    for plan in p["plans"]:
        assert plan.group_rates == (
            0.9,
            0.9,
            0.9,
        ) and plan.denominator == pytest.approx(5.4)
        assert plan.group_radii == p["plans"][0].group_radii
        assert len(set(dict(plan.layer_noise_std).values())) == 1
        assert np.dot(
            p["counts"] / sum(p["counts"]), plan.group_radii
        ) == pytest.approx(250)
    np.testing.assert_array_equal(
        p["envelope"], np.repeat(p["envelope"][:1], 3, axis=0)
    )
    assert all(v <= cap for v, cap in zip(p["final_epsilon"], cfg["epsilons"]))


def test_idp_full_release_and_audit(tmp_path):
    summary = run(config(), tmp_path / "idp")
    assert summary["non_gaussian_layer_releases"] == 0
    assert audit(tmp_path / "idp")["status"] == "passed"


def test_idp_disallows_non_gaussian_or_secondary_budget():
    cfg = config()
    cfg["mechanism"] = "brdp"
    with pytest.raises(ValueError):
        validate_config(cfg)
    cfg = config()
    cfg["calibration_epsilons"] = [1, 2, 3]
    with pytest.raises(ValueError):
        validate_config(cfg)


def synthetic_results():
    rows = []
    for i, seed in enumerate(SEEDS):
        for method, value in zip(METHODS, [60 + i, 65 + 2 * i, 67 + i]):
            rows.append(
                dict(
                    seed=seed,
                    label=method,
                    final_test_accuracy=value,
                    best_test_accuracy=value + 1,
                )
            )
    return rows


def test_statistics_sample_sd_and_paired_differences():
    result = stats(synthetic_results())
    ours = next(r for r in result["means"] if r["label"] == "BRDP_JOINT")
    assert (
        ours["final_test_accuracy_mean"] == 68
        and ours["final_test_accuracy_sample_std"] == 1
    )
    paired = result["paired_summary"]
    assert paired["brdp_minus_idp_final_pp"] == dict(
        n=3, mean=7.0, sample_std=0.0, positive_seeds=3
    )
    assert paired["brdp_minus_sayg_final_pp"] == dict(
        n=3, mean=1.0, sample_std=1.0, positive_seeds=2
    )


def test_partial_statistics_do_not_invent_zero_standard_deviation():
    result = stats(synthetic_results()[:1])
    assert result["means"][0]["n"] == 1
    assert result["means"][0]["final_test_accuracy_sample_std"] is None
    assert result["paired"] == []


def test_statistics_reject_duplicate_seed_method():
    rows = synthetic_results()
    with pytest.raises(ValueError, match="Duplicate"):
        stats(rows + [copy.deepcopy(rows[0])])


def test_nine_jobs_are_fresh_full_caps_and_time_adaptive_is_sayg():
    records = jobs()
    assert len(records) == 9
    assert {(r["seed"], r["label"]) for r in records} == {
        (s, m) for s in SEEDS for m in METHODS
    }
    assert all("calibration_epsilons" not in r["config"] for r in records)
    assert all(
        r["config"]["method"] == "SPEND_AS_YOU_GO"
        for r in records
        if r["label"] == "Time_Adaptive"
    )


def test_new_gate_requires_configs_and_rejects_escape(tmp_path, monkeypatch):
    import scripts.run_three_methods_seeds as suite

    monkeypatch.setattr(suite, "ROOT", tmp_path)
    names = suite.REQUIRED | {
        f"configs/three_method_seeds/{m}_seed{s}.yaml" for s in SEEDS for m in METHODS
    }
    manifest = {}
    for name in names:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("tested")
        manifest[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path / "gate.json").write_text(
        json.dumps(dict(status="passed", tests_passed=100, source_sha256=manifest))
    )
    assert suite.verify_gate("gate.json")["status"] == "passed"
    with pytest.raises(ValueError):
        suite.inside("../trynew")
    (tmp_path / "src/sayg_protocol.py").write_text("changed")
    with pytest.raises(RuntimeError):
        suite.verify_gate("gate.json")


def test_copied_scheme1_matches_original_without_importing_parent():
    root = Path(__file__).parents[1]
    for folder in ["clients", "server"]:
        filename = f"src/{folder}/{folder[:-1] if folder == 'clients' else folder}_scheme1_baseline.py"
        a = root / "fl-experiments-code-submission" / filename
        b = root.parent / "fl-experiments-code-submission" / filename
        assert (
            hashlib.sha256(a.read_bytes()).digest()
            == hashlib.sha256(b.read_bytes()).digest()
        )
