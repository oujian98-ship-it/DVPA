"""Numerical evidence checks and a replayable ranking-enabled training path."""

import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from src.dvpa_audit import audit
from src.dvpa_config import ROOT, load_config, public_preflight, resolve, smoke_config
from src.dvpa_controller import DecisionValueController
from src.dvpa_feedback import DPHistoryFeedback
from src.dvpa_filter import GroupRDPFilter
from src.dvpa_ranking import paired_evidence
from src.dvpa_runner import run


def test_paired_uncertainty_cancels_common_objective_offsets():
    common = np.array([1e6, -1e6, 2e6, -2e6])
    hi = np.column_stack([common, common + 0.01])
    lo = hi + 100.0
    check = paired_evidence(lo, hi, 3.0, 1e-9)
    assert check["stable"] and check["winner"] == 0
    assert check["gap"] == pytest.approx(0.01)
    assert check["mc_gap_se"] < 1e-9
    assert check["quad_gap_error"] < 1e-9


def test_mc_overlap_and_quadrature_reversal_are_unresolved():
    hi = np.array([[0.0, 1.0], [0.0, -1.0], [0.0, 1.0], [0.0, -1.0]])
    assert not paired_evidence(hi, hi, 3.0, 1e-9)["stable"]
    high = np.tile([0.0, 0.01], (4, 1))
    low = np.tile([0.0, -0.01], (4, 1))
    check = paired_evidence(low, high, 3.0, 1e-9)
    assert not check["stable"] and check["quad_gap_error"] == pytest.approx(0.02)


@pytest.fixture(scope="module")
def protocol():
    torch.set_num_threads(1)
    cfg = smoke_config(load_config(ROOT / "configs/dvpa_fmnist_ranking_seed42.yaml"))
    cfg["controller"].update(
        ranking_samples=32,
        ranking_max_samples=64,
        ranking_repeats=3,
        ranking_coarse_repeats=2,
        ranking_quadrature_nodes=12,
        ranking_validation_nodes=24,
    )
    return public_preflight(cfg)


def prepared(protocol):
    p = copy.deepcopy(protocol)
    f = DPHistoryFeedback(p)
    ledger = GroupRDPFilter(p)
    f.prepare(1)
    action, _ = DecisionValueController(p).choose(1, f, ledger.remaining())
    cert = ledger.certify(action)
    ledger.reserve(cert)
    f.observe(1, {n: torch.ones(d) * 0.01 for n, d in p.dimensions.items()}, cert)
    f.prepare(2)
    return p, f, ledger


def test_numerical_repeats_are_replayable_and_do_not_query_or_charge(protocol):
    p, f, ledger = prepared(protocol)
    old = copy.deepcopy(ledger.state_dict())
    belief = copy.deepcopy(f.state_dict())
    c = DecisionValueController(p)
    a, log = c.choose(2, f, ledger.remaining())
    first = copy.deepcopy(c.last_ranking)
    b, again = c.choose(2, f, ledger.remaining())
    assert a == b and first == c.last_ranking
    assert log["ranking_shortlist"] >= 2
    assert ledger.state_dict() == old
    assert np.array_equal(f.mean, belief["mean"])
    assert np.array_equal(f.cov, belief["covariance"])
    assert ledger.can_reserve(ledger.certify(a))
    assert all(log[k] == again[k] for k in log if k != "controller_seconds")


def test_unresolved_ranking_increases_samples_then_uses_frozen(protocol, monkeypatch):
    import src.dvpa_ranking as ranking

    p, f, ledger = prepared(protocol)
    original = ranking.paired_evidence
    counts = []

    def inconclusive(low, high, *args):
        counts.append(len(high))
        r = original(low, high, *args)
        r["stable"] = False
        return r

    monkeypatch.setattr(ranking, "paired_evidence", inconclusive)
    c = DecisionValueController(p)
    a, log = c.choose(2, f, ledger.remaining())
    assert len(counts) == 2
    assert log["ranking_escalated"] == 1
    assert log["ranking_samples"] == p.config["controller"]["ranking_max_samples"]
    assert log["fallback_reason"] == "unresolved_numerical_ranking_use_frozen_info"
    assert c.last_ranking["selected_index"] == c.last_ranking["frozen_index"]
    assert ledger.can_reserve(ledger.certify(a))


@pytest.mark.parametrize(
    "key,value",
    [
        ("ranking_repeats", 1),
        ("ranking_top_k", 1),
        ("ranking_samples", 33),
        ("ranking_max_samples", 2),
        ("ranking_validation_nodes", 1),
        ("ranking_gap_factor", float("nan")),
    ],
)
def test_invalid_ranking_controls_rejected(protocol, key, value):
    cfg = copy.deepcopy(protocol.config)
    cfg["controller"][key] = value
    with pytest.raises(ValueError):
        resolve(cfg)


def test_numerical_only_configuration_preserves_scientific_protocol():
    old = load_config(ROOT / "configs/dvpa_fmnist.yaml")
    new = load_config(ROOT / "configs/dvpa_fmnist_ranking_seed42.yaml")
    assert old["base"] == new["base"]
    assert all(old[s] == new[s] for s in ("feedback", "privacy", "output"))
    assert all(
        old["controller"][k] == v
        for k, v in new["controller"].items()
        if not k.startswith("ranking_")
    )


def test_ranking_training_audit_and_tampered_diagnostic_rejected(protocol):
    parent = ROOT / "reports/dvpa_ranking"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent, prefix="test-") as folder:
        out = Path(folder) / "run"
        summary = run(protocol.config, out)
        assert summary["numerical_ranking"]["checked_nonterminal_rounds"] == 2
        assert audit(out)["status"] == "passed"
        path = out / "ranking/round_002.json"
        record = json.loads(path.read_text())
        record["selected_index"] += 1
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError):
            audit(out)
