"""Causal short-horizon forecasts of relative public layer utility."""

import numpy as np


def parameters_from_config(cfg):
    return dict(
        ema=cfg["forecast_ema"],
        damping=cfg["forecast_damping"],
        clip=cfg["forecast_clip"],
        horizon=cfg["forecast_horizon"],
    )


def centered_log(values):
    x = np.asarray(values, dtype=float)
    if (
        x.ndim != 2
        or x.shape[1] != 4
        or not len(x)
        or not np.isfinite(x).all()
        or (x <= 0).any()
    ):
        raise ValueError("Finite positive four-layer history required")
    logs = np.log(x)
    return logs - logs.mean(axis=1, keepdims=True)


def forecast_coefficients(history, steps, ema, damping, clip=0.1, horizon=3):
    """Only observed coefficient vectors enter. Beyond horizon hold its forecast."""
    history = np.asarray(history, dtype=float)
    logs = centered_log(history)
    steps = np.asarray(steps)
    if (
        steps.ndim != 1
        or not np.issubdtype(steps.dtype, np.integer)
        or (steps < 0).any()
        or not 0 <= ema <= 1
        or not 0 <= damping <= 1
        or clip <= 0
        or type(horizon) is not int
        or horizon < 1
    ):
        raise ValueError("Invalid causal forecast settings")
    velocity = np.zeros(4)
    for change in np.diff(logs, axis=0):
        velocity = (1 - ema) * velocity + ema * change
    velocity = np.clip(velocity, -clip, clip)
    velocity -= velocity.mean()
    gains = np.array(
        [sum(damping**j for j in range(1, min(int(s), horizon) + 1)) for s in steps]
    )
    predicted = logs[-1] + gains[:, None] * velocity
    predicted -= predicted.max(axis=1, keepdims=True)
    values = np.exp(predicted)
    values = values / values.sum(axis=1, keepdims=True) * history[-1].sum()
    # Preserve current values exactly; matters to numerical/physical comparisons.
    values[steps == 0] = history[-1]
    return values


def verify_forecast_validation(config, proof=None):
    import hashlib
    import json
    from pathlib import Path
    from .dvpa_config import ROOT

    cfg = config["feedback"]
    if cfg["curvature_forecast"] == "hold_current":
        return None
    if proof is None:
        proof = json.loads(
            (ROOT / cfg["forecast_validation"]).read_text(encoding="utf-8")
        )
    if (
        proof.get("status") != "completed"
        or proof.get("passed") is not True
        or proof.get("parameters") != parameters_from_config(cfg)
        or proof.get("public_data_sha256") != cfg["public_proxy_sha256"]
        or proof.get("forecast_module_sha256")
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        or proof.get("script_sha256")
        != hashlib.sha256(
            (ROOT / "scripts/check_dvpa_forecast.py").read_bytes()
        ).hexdigest()
    ):
        raise ValueError("Passed current causal forecast verification required")
    if set(proof["checks"]) != {"old_seed44", "proxy_seed42"}:
        raise ValueError("Both forecast confirmation histories required")
    for row in proof["checks"].values():
        one, pooled = row["one_step"], row["three_step_pool"]
        if (
            not one["forecast_regret"] < one["hold_regret"]
            or not one["forecast_clr_mse"] < one["hold_clr_mse"]
            or pooled["forecast_regret"] > pooled["hold_regret"]
        ):
            raise ValueError("Forecast did not beat persistence")
    for name, expected in proof["history_manifest"].items():
        path = (ROOT / name).resolve()
        if (
            not path.is_relative_to(ROOT)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Historical forecast verification inputs changed")
    return proof
