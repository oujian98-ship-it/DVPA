"""Cross-dataset identity, held-out evaluation, transfer labels and execution."""

import copy
import json

import numpy as np
import pytest
import torch

from scripts.run_dvpa_datasets import configuration, jobs, run_suite, statistics
from src.brdp_source_protocol import load_datasets, validate_config
from src.dvpa_config import ROOT
from src.dvpa_dataset_support import (
    PublicCurvatureProxy, baseline_evaluation, read_package,
    verify_forecast_validation, verify_public_validation,
)


def test_full_eighteen_jobs_preserve_current_controller_and_common_training():
    records = jobs()
    assert len(records) == 18 and len({r["name"] for r in records}) == 18
    reference = configuration("fmnist", 42)
    for dataset in ("fmnist", "mnist"):
        for seed in (42, 43, 44):
            group = [j for j in records if j["dataset"] == dataset and j["seed"] == seed]
            cfg = group[0]["config"]
            assert cfg["controller"] == reference["controller"]
            assert cfg["feedback"]["extra_covariance"] == "dp_innovation"
            assert cfg["feedback"]["curvature_forecast"] == "damped_log_trend"
            assert cfg["base"]["rounds"] == 25 and cfg["base"]["local_epochs"] == 30
            for j in group[1:]:
                assert all(j["config"][k] == v for k, v in cfg["base"].items() if k != "method")


def test_mnist_cannot_use_fmnist_public_inputs_or_fake_empirical_benefit():
    cfg = configuration("mnist", 42)
    public = PublicCurvatureProxy(cfg)
    assert public.metadata["dataset"] == "MNIST official test split"
    assert len(public.final_indices) == 9000 and len(public.images) == 64
    bad = copy.deepcopy(cfg)
    bad["feedback"].update(reference := {k: configuration("fmnist", 42)["feedback"][k]
                          for k in ("public_proxy_file", "public_proxy_sha256")})
    with pytest.raises(ValueError, match="identity"):
        PublicCurvatureProxy(bad)
    proof = verify_public_validation(cfg)
    assert proof["empirical_benefit_validated"] is False
    assert verify_forecast_validation(cfg) == proof
    altered = copy.deepcopy(proof)
    altered["empirical_benefit_validated"] = True
    with pytest.raises(ValueError, match="verification"):
        verify_forecast_validation(cfg, altered)
    altered = copy.deepcopy(proof)
    altered["parameters"]["ema"] = 0.5
    with pytest.raises(ValueError, match="verification"):
        verify_forecast_validation(cfg, altered)


def test_official_heldout_split_matches_actual_images_and_rejects_wrong_dataset():
    torch.set_num_threads(1)
    cfg = configuration("mnist", 42)["base"]
    train, test = load_datasets(cfg)
    held, evaluation = baseline_evaluation(cfg, test)
    assert len(train) == 60000 and len(held) == 9000
    assert evaluation["dataset"] == "MNIST"
    assert torch.equal(held.dataset.tensors[0], test.tensors[0])
    wrong = copy.deepcopy(cfg)
    wrong["dataset"] = "fmnist"
    with pytest.raises(ValueError, match="identity"):
        baseline_evaluation(wrong, test)
    bad = copy.deepcopy(cfg)
    bad.pop("evaluation_public_sha256")
    with pytest.raises(ValueError, match="together"):
        validate_config(bad)


def test_statistics_use_sample_sd_and_separate_datasets_and_duplicates():
    rows = [dict(dataset="mnist", label="DVPA", seed=s,
            summary=dict(final_test_accuracy=v, best_test_accuracy=v+2))
            for s, v in zip((42, 43, 44), (70.0, 71.0, 72.0))]
    result = statistics(rows)
    assert len(result) == 1 and result[0]["n"] == 3
    assert result[0]["final_accuracy_mean"] == 71.0
    assert result[0]["final_accuracy_sample_std"] == 1.0
    assert result[0]["complete_seed_set"]
    with pytest.raises(ValueError, match="Duplicate"):
        statistics(rows + rows[:1])
    assert statistics(rows[:1])[0]["final_accuracy_sample_std"] is None


def test_both_dataset_routes_train_replay_and_aggregate_without_old_results(tmp_path):
    output = tmp_path / "two_dataset_queue"
    result = run_suite(output, seeds=(42,), smoke=True, enforce_gate=False)
    assert result["status"] == "completed" and result["completed_runs"] == 6
    state = json.loads((output / "suite_state.json").read_text(encoding="utf-8"))
    assert len(state["identity_checks"]) == 6
    for done in state["completed"]:
        assert done["audit"]["status"] == "passed"
        assert done["summary"]["evaluation_examples"] == 40
        assert done["summary"]["rounds"] == 4
        assert done["evaluation_recheck"]["accuracy"] == done["summary"]["final_test_accuracy"]
        assert all(p["epsilon_guarantee"] <= cap for p, cap in
                   zip(done["group_privacy"], (10, 20, 30)))
    assert all(v["initial_models_identical"] and v["partitions_and_groups_identical"]
               for v in state["identity_checks"])
    summary = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert len(summary["statistics"]) == 6
    assert all(v["n"] == 1 and v["final_accuracy_sample_std"] is None for v in summary["statistics"])
