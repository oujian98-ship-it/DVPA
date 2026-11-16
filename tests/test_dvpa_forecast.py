"""Causal forecast improves future-only planning, never the privacy certificate."""

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
from src.dvpa_forecast import (
    forecast_coefficients,
    verify_forecast_validation,
)
from src.dvpa_runner import run


def configuration():
    return load_config(ROOT / "configs/dvpa_fmnist_forecast_seed42.yaml")


def test_log_trend_exact_current_common_scale_invariance_and_bounded_horizon():
    history = np.array(
        [[1.0, 2.0, 3.0, 4.0], [1.1, 1.9, 3.0, 4.0], [1.2, 1.8, 3.0, 4.0]]
    )
    params = dict(ema=0.25, damping=0.75, clip=0.1, horizon=3)
    forecast = forecast_coefficients(history, np.array([0, 1, 2, 3, 10]), **params)
    assert np.array_equal(forecast[0], history[-1])
    assert np.all(forecast > 0) and forecast.sum(1) == pytest.approx(np.full(5, 10.0))
    assert np.array_equal(forecast[3], forecast[4])
    assert forecast[1, 0] > history[-1, 0] and forecast[1, 1] < history[-1, 1]
    scaled = history * np.array([3.0, 0.5, 2.0])[:, None]
    got = forecast_coefficients(scaled, np.array([1, 2]), **params)
    assert got / got.sum(1, keepdims=True) == pytest.approx(forecast[1:3] / 10)


def test_one_observation_and_zero_velocity_keep_current_curve():
    history = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert forecast_coefficients(
        history, np.array([0, 1, 5]), 0.25, 0.75
    ) == pytest.approx(np.broadcast_to(history, (3, 4)))
    history = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    assert forecast_coefficients(history, np.array([1, 2]), 0, 0.75) == pytest.approx(
        np.broadcast_to(history[-1], (2, 4))
    )


@pytest.mark.parametrize(
    "values,steps",
    [
        (np.zeros((2, 4)), np.array([1])),
        (np.ones((2, 4)), np.array([-1])),
        (np.ones((2, 4)), np.array([0.5])),
    ],
)
def test_bad_forecast_inputs_rejected(values, steps):
    with pytest.raises(ValueError):
        forecast_coefficients(values, steps, 0.25, 0.75)


def test_only_future_costs_change_and_covariance_and_fixed_action_fees_stay_same():
    cfg = smoke_config(configuration())
    p = public_preflight(cfg)
    f = DPHistoryFeedback(p)
    other = copy.deepcopy(cfg)
    other["feedback"]["curvature_forecast"] = "hold_current"
    old = public_preflight(other)
    of = DPHistoryFeedback(old)
    ledger = GroupRDPFilter(p)
    for t in (1, 2):
        for fb in (f, of):
            fb.prepare(t)
            coefficients = np.array([1 + t, 3.0, 4.0, 5 - t])
            coefficients *= sum(p.dimensions.values()) / coefficients.sum()
            fb.set_public_coefficients(
                t,
                dict(
                    round=t,
                    names=list(p.dimensions),
                    coefficients=coefficients,
                    public_data_sha256=cfg["feedback"]["public_proxy_sha256"],
                ),
            )
        if t == 2:
            break
        action, _ = DecisionValueController(p).choose(t, f, ledger.remaining())
        cert = ledger.certify(action)
        ledger.reserve(cert)
        released = {n: torch.zeros(d) for n, d in p.dimensions.items()}
        for fb in (f, of):
            fb.observe(t, released, cert)
    args = (f.pred_mean, f.pred_cov, [2, 3, 4], 2)
    current = f.expected_costs(*args)
    baseline = of.expected_costs(*args)
    assert np.array_equal(current[:, 0], baseline[:, 0])
    assert not np.allclose(current[:, 1:], baseline[:, 1:], rtol=1e-8, atol=0)
    assert np.array_equal(f.extra, of.extra) and np.array_equal(f.pred_cov, of.pred_cov)
    action, _ = DecisionValueController(p).choose(2, f, ledger.remaining())
    assert (
        ledger.certify(action).as_dict()
        == GroupRDPFilter(old).certify(action).as_dict()
    )
    assert len(f.state_dict()["public_proxy"]["coefficient_history"]) == 2


@pytest.mark.parametrize(
    "key,value",
    [
        ("forecast_ema", 2),
        ("forecast_damping", float("nan")),
        ("forecast_clip", 0),
        ("forecast_horizon", 0),
        ("forecast_validation", "../../outside.json"),
        ("utility_proxy", "dimension_signal"),
    ],
)
def test_invalid_forecast_configuration(key, value):
    cfg = configuration()
    cfg["feedback"][key] = value
    with pytest.raises(ValueError):
        resolve(cfg)


def test_forecast_verification_binds_selected_parameters():
    cfg = configuration()
    cfg["feedback"]["forecast_ema"] = 0.5
    with pytest.raises(ValueError, match="verification"):
        verify_forecast_validation(cfg)


def test_training_replay_uses_only_causal_prefix_and_detects_future_curve_tampering(
    monkeypatch,
):
    import src.dvpa_forecast as module

    original = module.forecast_coefficients
    lengths = []

    def wrapped(history, steps, **parameters):
        lengths.append(len(history))
        return original(history, steps, **parameters)

    monkeypatch.setattr(module, "forecast_coefficients", wrapped)
    cfg = smoke_config(configuration())
    cfg["feedback"]["public_proxy_probes"] = 2
    parent = ROOT / "reports/dvpa"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="forecast-", dir=parent) as directory:
        out = Path(directory) / "run"
        result = run(cfg, out)
        assert result["rounds"] == 4 and lengths == [1, 2, 3, 4]
        assert audit(out)["status"] == "passed"
        report_path = out / "public_proxy/round_002.json"
        report = json.loads(report_path.read_text())
        report["forecast_coefficients"][1][0] += 1.0
        report_path.write_text(json.dumps(report))
        with pytest.raises(ValueError, match="public_proxy"):
            audit(out)
