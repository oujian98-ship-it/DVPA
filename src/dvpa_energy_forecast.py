"""Causal ridge models for future public curvature, conditional on released energy."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


HORIZONS = (1, 3)
VARIANTS = ("B0_public", "B1_energy", "B2_legacy_direction")


def public_features(feedback, t, horizon):
    knots = feedback.curvature_log_mass
    recent = np.diff(knots[-4:])
    velocity = float(np.median(recent)) if len(recent) else 0.0
    forecast = feedback.pure_public_costs([min(t + horizon, feedback.protocol.rounds)], t)[0]
    current = feedback.pure_public_costs([t], t)[0]
    public = np.broadcast_to(
        [t / feedback.protocol.rounds, feedback.phase(t), feedback.phase(t + horizon), velocity],
        (len(feedback.names), 4),
    )
    return forecast, np.column_stack((np.log(current), np.log(forecast/current), public))


@dataclass(frozen=True)
class CurvaturePredictor:
    # Arrays have shape horizon x layer x feature, fit separately by layer.
    center: np.ndarray
    scale: np.ndarray
    slopes: np.ndarray
    intercept: np.ndarray
    residual_sd: np.ndarray

    @classmethod
    def fit(cls, examples, variant, ridge=12.0):
        centers, scales, slopes, intercepts, sds = [], [], [], [], []
        for horizon in HORIZONS:
            block = [e for e in examples if e["horizon"] == horizon]
            features = np.stack([e[variant] for e in block])
            targets = np.stack([np.log(e["target"]/e["offset"]) for e in block])
            center = features.mean(axis=0)
            scale = np.maximum(features.std(axis=0), 1e-6)
            x = (features - center)/scale
            intercept = targets.mean(axis=0)
            weights = []
            for layer in range(x.shape[1]):
                design = x[:, layer]
                cov = design.T @ design + ridge*np.eye(design.shape[1])
                weights.append(np.linalg.solve(cov, design.T @ (targets[:, layer] - intercept[layer])))
            weights = np.stack(weights)
            residuals = targets - intercept - np.sum(x*weights, axis=-1)
            centers.append(center)
            scales.append(scale)
            slopes.append(weights)
            intercepts.append(intercept)
            sds.append(np.maximum(residuals.std(axis=0, ddof=1), 1e-6))
        return cls(*(np.stack(a) for a in (centers, scales, slopes, intercepts, sds)))

    def distribution(self, horizon, offset, inputs):
        i = HORIZONS.index(horizon)
        x = np.clip((inputs-self.center[i])/self.scale[i], -5, 5)
        adjustment = self.intercept[i] + np.sum(x*self.slopes[i], axis=-1)
        return np.log(offset) + np.clip(adjustment, -3, 3), self.residual_sd[i]

    def predict(self, horizon, offset, inputs):
        location, spread = self.distribution(horizon, offset, inputs)
        bound = norm.ppf(.95) * spread
        return np.exp(location), np.exp(location-bound), np.exp(location+bound)

    def expected(self, horizon, offset, inputs):
        location, spread = self.distribution(horizon, offset, inputs)
        return np.exp(location + spread**2/2)

    def record(self):
        return {key: value.tolist() for key, value in vars(self).items()}

    @classmethod
    def from_record(cls, record):
        model = cls(**{k: np.array(v, dtype=float) for k, v in record.items()})
        if (model.center.shape != (2, 4, 8) or model.scale.shape != model.center.shape
                or model.slopes.shape != model.center.shape or model.intercept.shape != (2, 4)
                or model.residual_sd.shape != (2, 4)
                or not all(np.isfinite(a).all() for a in vars(model).values())
                or np.any(model.scale <= 0) or np.any(model.residual_sd <= 0)):
            raise ValueError("Invalid energy predictor record")
        return model
