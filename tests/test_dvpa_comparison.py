import copy
import json
from pathlib import Path
import tempfile

import pytest
import torch

import scripts.run_dvpa_comparison as comparison
from src.dvpa_config import ROOT
from src.brdp_source_protocol import preflight


@pytest.fixture
def workspace():
    parent = ROOT / "reports/dvpa_comparison"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test-", dir=parent) as path:
        yield Path(path)


def test_jobs_share_training_and_caps_preserve_full_method_labels():
    jobs = comparison.jobs([42, 43, 44])
    assert len(jobs) == 9
    for seed in [42, 43, 44]:
        selected = [j for j in jobs if j["seed"] == seed]
        base = selected[0]["config"]["base"]
        for j in selected[1:]:
            assert all(
                j["config"][key] == v for key, v in base.items() if key != "method"
            )
        assert [j["label"] for j in selected] == list(comparison.LABELS)
        assert (
            base["clients"] == 100
            and base["rounds"] == 25
            and base["local_epochs"] == 30
        )


@pytest.mark.parametrize("seeds", [[], [42, 42], [-1]])
def test_invalid_seed_queue_rejected(seeds):
    with pytest.raises(ValueError):
        comparison.jobs(seeds)


def test_native_baseline_preflights_have_correct_sampling():
    jobs = comparison.jobs([42], smoke=True)
    dims = dict(conv1=832, conv2=51264, fc1=524800, fc=5130)
    for job in jobs[1:]:
        p = preflight(job["config"], dims)
        assert all(a <= b for a, b in zip(p["final_epsilon"], [10, 20, 30]))
        if job["label"] == "IDP_FedAvg":
            assert all(plan.group_rates == (0.9, 0.9, 0.9) for plan in p["plans"])
        else:
            assert p["plans"][0].group_rates == (0.5, 0.6, 0.7)
            assert p["plans"][2].group_rates == (0.9, 0.9, 0.9)
            assert (
                p["schedule_metadata"]["method"]
                == "source_scheme2_epsilon_remainder_spend_as_you_go"
            )


def test_fresh_three_method_queue_identity_audits_and_privacy(workspace):
    result = comparison.run_suite(
        workspace / "run", seeds=[42], smoke=True, device="cpu", enforce_gate=False
    )
    assert result["status"] == "completed" and result["completed"] == 3
    state = json.loads((workspace / "run/suite_state.json").read_text(encoding="utf-8"))
    assert len(state["identity_checks"]) == 3
    assert all(
        x["initial_models_identical"] and x["partitions_and_groups_identical"]
        for x in state["identity_checks"]
    )
    assert [j["label"] for j in state["completed"]] == list(comparison.LABELS)
    for done in state["completed"]:
        assert done["audit"]["status"] == "passed"
        assert done["summary"]["dataset_scope"] == "synthetic engineering"
        assert all(
            p["epsilon_guarantee"] <= cap
            for p, cap in zip(done["group_privacy"], [10, 20, 30])
        )
        assert all(not p["path_is_global_guarantee"] for p in done["group_privacy"])
    directory = workspace / "run/DVPA_seed42"
    inputs = json.loads(
        (workspace / "run/formal_inputs.json").read_text(encoding="utf-8")
    )
    expected = inputs["expected_identity"]["42"]
    bad = copy.deepcopy(expected)
    bad["group_ids"][0] = (bad["group_ids"][0] + 1) % 3
    with pytest.raises(ValueError):
        comparison.verify_identity(directory, bad)
    assert comparison.verify_identity(directory, expected)["initial_models_identical"]
    assert all(
        (workspace / f"run/{label}_seed42.stderr.log").read_text() == ""
        for label in comparison.LABELS
    )
    report = json.loads((workspace / "run/comparison.json").read_text())
    assert len(report["paired"]) == 1 and all(
        s["final_accuracy_sample_std"] is None for s in report["statistics"]
    )


def test_model_identity_digest_is_content_not_serialization():
    one = {"a": torch.arange(3, dtype=torch.float32)}
    two = {"a": one["a"].clone()}
    assert comparison.tensor_digest(one) == comparison.tensor_digest(two)
    two["a"][0] += 1
    assert comparison.tensor_digest(one) != comparison.tensor_digest(two)
