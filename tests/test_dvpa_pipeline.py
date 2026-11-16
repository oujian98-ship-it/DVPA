import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from src.dvpa_audit import audit
from src.dvpa_config import ROOT, load_config, smoke_config
from src.dvpa_runner import run


@pytest.fixture
def workspace():
    parent = ROOT / "reports/dvpa"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test-", dir=parent) as directory:
        assert Path(directory).resolve().is_relative_to(parent.resolve())
        yield Path(directory)


def configuration():
    return smoke_config(load_config(ROOT / "configs/dvpa_fmnist.yaml"))


def test_full_training_replay_and_safe_artifacts(workspace, monkeypatch):
    from src.brdp_source_protocol import PersistentSourceClient
    from src.dvpa_filter import GroupRDPFilter

    calls = []
    original_reserve = GroupRDPFilter.reserve
    original_train = PersistentSourceClient.train

    def reserve(self, cert):
        result = original_reserve(self, cert)
        calls.append(("reserve", cert.action.round_index))
        return result

    def train(self, *args, **kwargs):
        assert calls and any(c[0] == "reserve" for c in calls)
        calls.append(("train", self.epochs_completed + 1))
        return original_train(self, *args, **kwargs)

    monkeypatch.setattr(GroupRDPFilter, "reserve", reserve)
    monkeypatch.setattr(PersistentSourceClient, "train", train)
    result = run(configuration(), workspace / "run")
    assert result["dataset_scope"] == "synthetic engineering"
    assert result["total_client_local_epochs"] == 24
    assert [c[0] for c in calls] == [
        kind for _ in range(4) for kind in ["reserve"] + ["train"] * 6
    ]
    assert audit(workspace / "run")["status"] == "passed"
    state = torch.load(workspace / "run/feedback_state.pt", weights_only=True)
    assert state["last_observed"] == 4
    assert state["basis"] is not None
    assert (
        "rng"
        not in json.dumps(
            {k: v for k, v in state.items() if k not in {"basis", "direction"}}
        ).lower()
    )
    assert not any(
        "private" in p.name and p.suffix == ".pt"
        for p in (workspace / "run").rglob("*")
    )


@pytest.mark.parametrize("mode", ["static", "frozen_info", "entropy"])
def test_comparison_modes_train_and_replay(workspace, mode):
    cfg = configuration()
    cfg["controller"]["mode"] = mode
    result = run(cfg, workspace / mode)
    assert result["controller_mode"] == mode
    assert audit(workspace / mode)["status"] == "passed"


def test_empty_sampling_still_trains_charges_and_releases(workspace, monkeypatch):
    import src.dvpa_runner as runner

    old = runner._private_rngs

    class EmptySample:
        def random(self, n):
            return np.ones(n)

    def rngs(device):
        _, scalar, noise = old(device)
        return EmptySample(), scalar, noise

    monkeypatch.setattr(runner, "_private_rngs", rngs)
    result = run(configuration(), workspace / "empty")
    assert result["total_client_local_epochs"] == 24
    assert all(r["rdp_spent"] > 0 for r in result["group_privacy"])
    assert len(list((workspace / "empty/released").glob("*.pt"))) == 4
    assert audit(workspace / "empty")["status"] == "passed"


def test_refusal_precedes_training_and_preserves_failed_reservation(
    workspace, monkeypatch
):
    import src.dvpa_runner as runner
    from dataclasses import replace

    original = runner.DecisionValueController.choose

    def choose(self, *args, **kwargs):
        a, log = original(self, *args, **kwargs)
        return replace(a, precision=a.precision * 1000), log

    monkeypatch.setattr(runner.DecisionValueController, "choose", choose)
    monkeypatch.setattr(
        runner.PersistentSourceClient,
        "train",
        lambda *_a, **_k: pytest.fail("Training preceded privacy refusal"),
    )
    with pytest.raises(RuntimeError, match="Privacy filter rejected"):
        run(configuration(), workspace / "refused")
    state = json.loads((workspace / "refused/run_state.json").read_text())
    assert state["reserved_round"] == 0 and state["completed_rounds"] == 0
    assert state["reserved_costs_not_refunded"]


def test_reserved_cost_survives_crash_and_no_overwrite(workspace, monkeypatch):
    import src.dvpa_runner as runner

    def crash(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(runner.PersistentSourceClient, "train", crash)
    with pytest.raises(RuntimeError):
        run(configuration(), workspace / "crash")
    state = json.loads((workspace / "crash/run_state.json").read_text())
    assert state["reserved_round"] == 1 and min(state["spent"]) > 0
    with pytest.raises(FileExistsError):
        run(configuration(), workspace / "crash")
    with pytest.raises((ValueError, FileNotFoundError)):
        audit(workspace / "crash")


@pytest.mark.parametrize(
    "target", ["sigma", "history", "order", "feedback", "model", "privacy_label"]
)
def test_audit_rejects_tampered_public_trajectory(workspace, target):
    out = workspace / "tamper"
    run(configuration(), out)
    if target in {"sigma", "history", "order"}:
        lines = [
            json.loads(x) for x in (out / "actions.jsonl").read_text().splitlines()
        ]
        if target == "sigma":
            lines[1]["certified"]["source_plan"]["layer_noise_std"][0][1] *= 0.9
        elif target == "history":
            lines[1]["certified"]["action"]["history_hash"] = "forged"
        else:
            lines[0], lines[1] = lines[1], lines[0]
        # Rehash to demonstrate validation beyond unkeyed checksums.
        from src.dvpa_filter import digest

        for line in lines:
            line["record_hash"] = digest(
                {k: v for k, v in line.items() if k != "record_hash"}
            )
        (out / "actions.jsonl").write_text(
            "\n".join(json.dumps(x) for x in lines) + "\n"
        )
    elif target in {"feedback", "model"}:
        file = out / ("feedback_state.pt" if target == "feedback" else "model.pt")
        state = torch.load(file, weights_only=True)
        if target == "feedback":
            state["mean"][0] += 1.0
        else:
            state[next(iter(state))].flatten()[0] += 1.0
        torch.save(state, file)
    else:
        text = (out / "group_privacy.csv").read_text().replace("False", "True")
        (out / "group_privacy.csv").write_text(text)
    with pytest.raises(ValueError):
        audit(out)


def test_suite_gate_rejects_changed_sources_and_formal_protocol(workspace, monkeypatch):
    import scripts.run_dvpa_suite as suite

    monkeypatch.setattr(suite, "ROOT", workspace)
    monkeypatch.setattr(suite, "manifest", lambda: {"src/current.py": "current"})
    parent = workspace / "reports/dvpa"
    parent.mkdir(parents=True)
    with pytest.raises(RuntimeError):
        suite.verify_gate()
    gate = {
        "status": "passed",
        "tests_passed": 200,
        "test_modules": [
            "tests." + name.removesuffix(".py") for name in suite.TEST_FILES
        ],
        "source_sha256": {"src/current.py": "outdated"},
    }
    (parent / "test_gate.json").write_text(json.dumps(gate))
    with pytest.raises(RuntimeError):
        suite.verify_gate()
    gate["source_sha256"] = suite.manifest()
    (parent / "test_gate.json").write_text(json.dumps(gate))
    assert suite.verify_gate()["status"] == "passed"
    with pytest.raises(ValueError):
        suite.verify_formal(configuration())
