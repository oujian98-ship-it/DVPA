"""Local refinement must improve a fixed proxy reference beyond paired error."""

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
from src.dvpa_local_search import (
    accept_proposal,
    action_key,
    decrease_evidence,
    local_pool,
    neighborhood,
)
from src.dvpa_runner import run


def test_changed_action_without_supported_decrease_is_rejected():
    hi = np.array(
        [[10.0, 9.0, 8.0], [10.0, 12.0, 8.0], [10.0, 9.0, 8.0], [10.0, 12.0, 8.0]]
    )
    selected, gain, _, info = accept_proposal(
        hi, hi, proposal=1, frozen=0, incumbent=0, factor=3.0, tolerance=1e-8
    )
    assert selected == 0 and not gain["supported"] and not info


def test_supported_info_gain_must_beat_incumbent_and_same_pool_frozen():
    hi = np.tile([10.0, 9.9, 9.8], (4, 1))
    selected, gain, contrast, info = accept_proposal(
        hi, hi, proposal=2, frozen=1, incumbent=0, factor=3.0, tolerance=1e-8
    )
    assert selected == 2 and gain["supported"] and contrast["supported"] and info


def test_noisy_information_advantage_uses_only_supported_frozen_gain():
    hi = np.column_stack(
        [np.full(4, 10.0), np.full(4, 9.0), np.array([8.0, 10.0, 8.0, 10.0])]
    )
    selected, gain, _, info = accept_proposal(
        hi, hi, proposal=2, frozen=1, incumbent=0, factor=3.0, tolerance=1e-8
    )
    assert (
        selected == 1
        and gain["supported"]
        and gain["gain"] == pytest.approx(1.0)
        and not info
    )


def test_quadrature_disagreement_prevents_false_improvement():
    hi = np.tile([10.0, 9.99], (4, 1))
    lo = np.tile([10.0, 10.01], (4, 1))
    e = decrease_evidence(lo, hi, 0, 1, 3.0, 1e-8)
    assert (
        e["gain"] > 0 and e["quad_error"] == pytest.approx(0.02) and not e["supported"]
    )


@pytest.fixture(scope="module")
def protocol():
    torch.set_num_threads(1)
    cfg = smoke_config(load_config(ROOT / "configs/dvpa_fmnist_local_seed42.yaml"))
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
    a, _ = DecisionValueController(p).choose(1, f, ledger.remaining())
    cert = ledger.certify(a)
    ledger.reserve(cert)
    f.observe(1, {n: torch.ones(d) * 0.01 for n, d in p.dimensions.items()}, cert)
    f.prepare(2)
    return p, f, ledger


def test_local_candidates_preserve_simplex_and_all_group_future_reserves(protocol):
    p, f, ledger = prepared(protocol)
    c = DecisionValueController(p)
    actions, tails = c.candidate_context(2, f, ledger.remaining())
    center = actions[0]
    adjacent = list(neighborhood(c, center))
    assert any(
        a.precision == pytest.approx(center.precision * 1.01, rel=1e-12, abs=0)
        for a in adjacent
    )
    pool = local_pool(c, 2, ledger.remaining(), [center], tails)
    assert action_key(center) in {action_key(a) for a in pool}
    reserve = p.charges([3, 4], p.z_floor).sum(axis=0)
    for a in pool:
        weights = np.array([v for _, v in a.layer_precision])
        assert (
            weights.sum() == pytest.approx(1.0) and np.min(weights) >= p.p_min - 1e-14
        )
        assert np.all(p.charges(2, a.precision) + reserve <= ledger.remaining())
        assert ledger.can_reserve(ledger.certify(a))


def test_local_refinement_is_deterministic_and_never_charges_virtual_queries(protocol):
    p, f, ledger = prepared(protocol)
    state = copy.deepcopy(ledger.state_dict())
    c = DecisionValueController(p)
    a, log = c.choose(2, f, ledger.remaining())
    saved = copy.deepcopy(c.last_local_search)
    b, _ = c.choose(2, f, ledger.remaining())
    assert a == b and saved == c.last_local_search and ledger.state_dict() == state
    assert ledger.can_reserve(ledger.certify(a))
    assert saved["search_stream"] != saved["validation_stream"]
    if log["local_accepted"]:
        assert log["local_proxy_gain"] > 0 and log["local_gain_margin"] > 0
    if log["local_information_changed_action"]:
        assert action_key(a) != action_key(
            type(a)(
                **{
                    **a.as_dict(),
                    "precision": saved["frozen_action"]["precision"],
                    "layer_precision": tuple(
                        tuple(x) for x in saved["frozen_action"]["layer_precision"]
                    ),
                }
            )
        )
        assert log["local_information_margin"] > 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("local_search_enabled", 1),
        ("ranking_enabled", False),
        ("local_search_steps", 2),
        ("local_time_offsets", [0.0]),
        ("local_time_offsets", [0.01, 0.01]),
        ("local_layer_transfers", [float("nan")]),
        ("local_min_decrease", 0.0),
    ],
)
def test_bad_local_controls_rejected(protocol, key, value):
    cfg = copy.deepcopy(protocol.config)
    cfg["controller"][key] = value
    with pytest.raises(ValueError):
        resolve(cfg)


def test_only_search_scope_changed_in_new_configuration():
    old = load_config(ROOT / "configs/dvpa_fmnist_ranking_seed42.yaml")
    new = load_config(ROOT / "configs/dvpa_fmnist_local_seed42.yaml")
    assert all(old[s] == new[s] for s in ("base", "feedback", "privacy", "output"))
    assert all(
        old["controller"][k] == v
        for k, v in new["controller"].items()
        if not k.startswith("local_")
    )


def test_local_pipeline_replays_and_rejects_forged_gain(protocol):
    parent = ROOT / "reports/dvpa_local"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent, prefix="test-") as folder:
        out = Path(folder) / "run"
        result = run(protocol.config, out)
        assert result["local_search"]["checked_rounds"] == 3
        assert audit(out)["status"] == "passed"
        file = out / "local_search/round_002.json"
        record = json.loads(file.read_text())
        record["final_improvement"]["gain"] += 1.0
        file.write_text(json.dumps(record))
        with pytest.raises(ValueError):
            audit(out)
