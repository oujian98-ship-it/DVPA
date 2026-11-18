import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from src.dvpa_audit import audit
from src.dvpa_config import ROOT, load_config, public_preflight, smoke_config
from src.dvpa_convex import ConvexPrecision
from src.dvpa_runner import run


def configuration():
    cfg = smoke_config(load_config(ROOT / "configs/dvpa_fmnist.yaml"))
    cfg["base"].update(rounds=5, device="cpu")
    cfg["controller"].update(tail_solver="convex", mode="decision_value",
                             ranking_enabled=False, local_search_enabled=False)
    return cfg


def test_scaled_rdp_gradient_and_uneven_optimum():
    protocol = public_preflight(configuration())
    rounds = np.arange(1, protocol.rounds + 1)
    planner = ConvexPrecision(protocol, rounds, protocol.caps)
    x = np.full(len(rounds), 1.0)
    costs, jacobian = planner.costs(x)
    assert np.allclose(costs, planner.exact(x), rtol=1e-12, atol=1e-12)
    for j in range(len(rounds)):
        shift = np.eye(len(rounds))[j] * 1e-5
        derivative = (planner.exact(x+shift) - planner.exact(x-shift)) / 2e-5
        assert np.allclose(derivative, jacobian[:, j], rtol=1e-8, atol=1e-9)
    risks = np.array([1, 2, 3, 4, 20])
    plan = planner.solve(risks)
    assert planner.fallbacks == 0
    assert np.all(plan >= protocol.z_floor)
    assert np.all(protocol.charges(rounds, plan).sum(axis=0) <= protocol.caps)
    assert np.dot(risks, 1/plan) < np.dot(risks, 1/(planner.uniform*protocol.z0))


def test_convex_training_preserves_full_replay():
    with tempfile.TemporaryDirectory(prefix="convex-", dir=ROOT / "reports") as directory:
        folder = Path(directory) / "run"
        output = run(configuration(), folder)
        assert output["rounds"] == 5
        assert audit(folder)["status"] == "passed"
        actions = [json.loads(row) for row in (folder / "actions.jsonl").read_text().splitlines()]
        assert len(actions) == 5


def test_fixed_schedule_exponent_preserves_full_replay():
    cfg = configuration()
    reference = public_preflight(cfg)
    warmup = cfg["feedback"]["warmup_rounds"]
    z0 = float(reference.z0)
    schedule = [z0] * warmup + [0.9 * z0] * (reference.rounds - warmup)
    cfg["controller"].update(mode="frozen_info", information_value_weight=0,
                             fixed_precision_schedule=schedule, layer_gamma=0.25)
    with tempfile.TemporaryDirectory(prefix="fixed-precision-", dir=ROOT / "reports") as directory:
        folder = Path(directory) / "run"
        run(cfg, folder)
        assert audit(folder)["status"] == "passed"
        actions = [json.loads(row)["certified"]["action"] for row in
                   (folder / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [record["precision"] for record in actions] == schedule
    cfg["controller"]["fixed_precision_schedule"] = [z0] * warmup + [10 * z0] * (reference.rounds - warmup)
    with pytest.raises(ValueError, match="fit all three RDP caps"):
        public_preflight(cfg)
