"""Cross-dataset identity, held-out evaluation, transfer labels and execution."""

import copy
import json

import numpy as np
import pytest
import torch

from scripts.run_dvpa_datasets import configuration, jobs, run_suite, statistics
from src.brdp_source_protocol import load_datasets, validate_config
from src.dvpa_config import ROOT, load_config, resolve
from src.dvpa_dataset_support import (
    PublicCurvatureProxy,
    verify_forecast_validation, verify_public_validation,
)
from src.dvpa_test_public64 import evaluation_data, read_package


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
            assert cfg["feedback"]["forecast_horizon"] == "full"
            assert cfg["base"]["rounds"] == 25 and cfg["base"]["local_epochs"] == 30
            for j in group[1:]:
                assert all(j["config"][k] == v for k, v in cfg["base"].items() if k != "method")


def test_mnist_cannot_use_fmnist_public_inputs_or_fake_empirical_benefit():
    # Check the legacy certificate against the original, archived configuration.
    cfg = load_config(ROOT / "configs/dvpa_datasets_20261105/mnist_DVPA_seed42.yaml")
    public = PublicCurvatureProxy(cfg)
    assert public.metadata["dataset"] == "MNIST official test split"
    assert len(public.final_indices) == 9000 and len(public.images) == 64
    bad = copy.deepcopy(cfg)
    legacy_fmnist = load_config(ROOT / "configs/dvpa_datasets_20261105/fmnist_DVPA_seed42.yaml")
    bad["feedback"].update({k: legacy_fmnist["feedback"][k]
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


@pytest.mark.parametrize("dataset", ["mnist", "fmnist"])
def test_official_heldout_split_matches_actual_images_and_rejects_wrong_dataset(dataset):
    torch.set_num_threads(1)
    full_cfg = configuration(dataset, 42)
    cfg = full_cfg["base"]
    train, test = load_datasets(cfg)
    held, evaluation = evaluation_data(cfg, test)
    assert len(train) == 60000 and len(test) == 10000 and len(held) == 9936
    assert evaluation["dataset"] == dataset.upper() and evaluation["evaluation_split"] == "official_test_holdout"
    assert held.dataset is test
    public = evaluation["public_test_indices"]
    assert len(public) == 64 and not (set(public) & set(held.indices))
    assert set(public) | set(held.indices) == set(range(len(test)))
    assert configuration(dataset, 43)["base"]["evaluation_public_sha256"] == cfg["evaluation_public_sha256"]
    _, _, package = read_package(cfg)
    assert torch.equal(package["images"], test.tensors[0][public])
    assert "labels" not in package
    test.tensors[0][public[0]].zero_()
    with pytest.raises(ValueError, match="differ from"):
        evaluation_data(cfg, test)
    wrong = copy.deepcopy(cfg)
    wrong["dataset"] = "fmnist" if dataset == "mnist" else "mnist"
    with pytest.raises(ValueError, match="identity"):
        evaluation_data(wrong, test)
    bad = copy.deepcopy(cfg)
    bad.pop("evaluation_public_sha256")
    with pytest.raises(ValueError, match="together"):
        validate_config(bad)
    with pytest.raises(ValueError, match="full training split"):
        validate_config(dict(cfg, validation_count=5000))
    full_cfg["feedback"]["public_proxy_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="reserved official"):
        resolve(full_cfg)


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
