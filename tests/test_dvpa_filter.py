import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from opacus.accountants.analysis.rdp import compute_rdp

from src.dvpa_config import load_config, public_preflight, resolve, smoke_config
from src.dvpa_filter import GroupRDPFilter, PrecisionAction, sgm_cost


@pytest.fixture(scope="module")
def protocol():
    return public_preflight(
        smoke_config(
            load_config(Path(__file__).parents[1] / "configs/dvpa_fmnist.yaml")
        )
    )


def action(p, t=1, z=None, weights=(0.1, 0.2, 0.3, 0.4)):
    return PrecisionAction(
        t,
        p.z0 if z is None else z,
        tuple(zip(p.dimensions, weights)),
        "basis",
        "history",
    )


@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("alpha", [3, 4, 16])
def test_integer_moments_match_independent_opacus(q, alpha):
    squared = np.array([0.001, 0.1, 2.0])
    got = sgm_cost(alpha, q, squared)
    expected = np.array(
        [
            compute_rdp(q=q, noise_multiplier=1 / np.sqrt(s), steps=1, orders=[alpha])[
                0
            ]
            for s in squared
        ]
    )
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_formal_preview_matches_public_anchors_without_data(monkeypatch):
    import src.brdp_source_protocol as source

    monkeypatch.setattr(
        source, "load_datasets", lambda *_: pytest.fail("Preview read data")
    )
    p = public_preflight(
        load_config(Path(__file__).parents[1] / "configs/dvpa_fmnist.yaml")
    )
    assert p.orders.tolist() == [4, 3, 3]
    assert p.counts.tolist() == [34, 43, 23]
    assert p.radii == pytest.approx([161.5977823552, 263.3020979634, 355.8123994564])
    assert p.z0 == pytest.approx(6.46568485e-6, rel=2e-8)
    assert np.all(p.charges(np.arange(1, 26), p.z0).sum(axis=0) <= p.caps)


def test_actual_joint_noise_not_layerwise_amplification(protocol):
    ledger = GroupRDPFilter(protocol)
    a = ledger.certify(action(protocol))
    b = ledger.certify(action(protocol, weights=(0.25,) * 4))
    assert a.group_rdp_increment == pytest.approx(b.group_rdp_increment)
    z = protocol.radii[0] ** 2 * protocol.z0
    q = protocol.rates[0, 0]
    order = int(protocol.orders[0])
    assert sgm_cost(order, q, z) > 4 * sgm_cost(order, q, z / 4)


@pytest.mark.parametrize(
    "change", ["sigma", "radius", "rate", "scale", "weights", "shape", "nan"]
)
def test_actual_plan_tampering_rejected(protocol, change):
    ledger = GroupRDPFilter(protocol)
    cert = ledger.certify(action(protocol))
    plan = cert.source_plan
    if change in {"sigma", "nan"}:
        stds = list(plan.layer_noise_std)
        stds[0] = (stds[0][0], stds[0][1] * 0.9 if change == "sigma" else float("nan"))
        changed = replace(plan, layer_noise_std=tuple(stds))
    elif change == "radius":
        changed = replace(plan, group_radii=(1.0, 2.0, 3.0))
    elif change == "rate":
        changed = replace(plan, group_rates=(1.0, 1.0, 1.0))
    elif change == "scale":
        changed = replace(plan, final_scale=plan.final_scale * 2)
    elif change == "weights":
        changed = replace(
            plan, layer_clip_weights=tuple((n, 0.1) for n, _ in plan.layer_clip_weights)
        )
    else:
        changed = replace(
            plan, planned_shapes=tuple((n, 0.1, 1.0) for n, _, _ in plan.planned_shapes)
        )
    before = ledger.state_dict()
    with pytest.raises(ValueError):
        ledger.certify(cert.action, changed)
    assert not ledger.can_reserve(replace(cert, source_plan=changed))
    assert ledger.state_dict() == before


def test_atomic_group_refusal_forged_receipt_and_duplicate(protocol):
    ledger = GroupRDPFilter(protocol)
    certified = ledger.certify(action(protocol, z=protocol.z0 * 20))
    before = copy.deepcopy(ledger.state_dict())
    with pytest.raises(RuntimeError):
        ledger.reserve(certified)
    assert ledger.state_dict() == before
    safe = ledger.certify(action(protocol))
    assert not ledger.can_reserve(replace(safe, group_rdp_increment=(0.0, 0.0, 0.0)))
    ledger.reserve(safe)
    used = copy.deepcopy(ledger.state_dict())
    with pytest.raises(RuntimeError):
        ledger.reserve(safe)
    assert ledger.state_dict() == used
    assert [r["epsilon_guarantee"] for r in ledger.rows()] == [10.0, 20.0, 30.0]
    assert all(r["is_global_guarantee"] is False for r in ledger.rows())


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_precision_never_charged(protocol, value):
    ledger = GroupRDPFilter(protocol)
    with pytest.raises(ValueError):
        ledger.certify(action(protocol, z=value))
    assert not ledger.receipts and not ledger.spent.any()


def test_strict_config_and_gaussian_only(protocol):
    cfg = copy.deepcopy(protocol.config)
    cfg["controller"]["use_raw_gradients"] = True
    with pytest.raises(ValueError):
        resolve(cfg)
    cfg = copy.deepcopy(protocol.config)
    cfg["base"]["mechanism"] = "brdp"
    with pytest.raises(ValueError):
        resolve(cfg)
    cfg = copy.deepcopy(protocol.config)
    cfg["output"]["resume"] = True
    with pytest.raises(ValueError):
        resolve(cfg)


def test_random_history_dependent_spending_all_paths_filtered(protocol):
    rng = np.random.default_rng(42)
    for _ in range(20):
        ledger = GroupRDPFilter(protocol)
        for t in range(1, protocol.rounds + 1):
            # Deliberately depend on previous noisy pseudo-observations.
            z = protocol.z0 * np.exp(rng.normal())
            cert = ledger.certify(action(protocol, t, z))
            if ledger.can_reserve(cert):
                ledger.reserve(cert)
            else:
                cert = ledger.certify(action(protocol, t, protocol.z_floor))
                if ledger.can_reserve(cert):
                    ledger.reserve(cert)
                else:
                    break
            assert np.all(ledger.spent <= protocol.caps)
