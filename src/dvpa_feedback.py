"""DP-history-only, fixed-coordinate eight-dimensional working belief.

This Gaussian state model is an explicit utility approximation. It is NOT a
privacy assumption, an unbiased layer-importance estimator, or an exact model
of Poisson sampling noise.
"""

import hashlib
import math

import numpy as np
from numpy.polynomial.hermite import hermgauss
import torch

from .brdp_direction import DirectionState, direction_hash
from .dvpa_filter import digest


def psd(matrix, floor=0.0, ceiling=None):
    matrix = np.asarray(matrix, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite feedback covariance")
    vals, vecs = np.linalg.eigh((matrix + matrix.T) / 2)
    vals = np.maximum(vals, floor)
    if ceiling is not None:
        vals = np.minimum(vals, ceiling)
    return (vecs * vals) @ vecs.T


def release_hash(released, names):
    h = hashlib.sha256()
    if set(released) != set(names):
        raise ValueError("Released layer coverage mismatch")
    for n in names:
        x = released[n].detach().cpu().double().flatten()
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite released history")
        h.update(n.encode())
        h.update(x.contiguous().numpy().tobytes())
    return h.hexdigest()


class DPHistoryFeedback:
    def __init__(self, protocol):
        self.protocol = protocol
        self.cfg = protocol.config["feedback"]
        self.names = list(protocol.dimensions)
        self.direction = DirectionState(protocol.dimensions)
        self.basis = None
        self.basis_hash = "warmup:none"
        self.history_hash = digest("empty DP history")
        self.last_observed = 0
        self.prepared_round = 0
        layer = np.full((4, 4), self.cfg["layer_correlation"])
        np.fill_diagonal(layer, 1.0)
        phase = np.array(
            [[1.0, self.cfg["phase_correlation"]], [self.cfg["phase_correlation"], 1.0]]
        )
        self.correlation = np.kron(phase, layer)
        self.process_cov = self.cfg["process_std"] ** 2 * self.correlation
        self.mean = np.zeros(8)
        self.cov = self.cfg["prior_std"] ** 2 * self.correlation
        self.pred_mean = self.mean.copy()
        self.pred_cov = self.cov.copy()
        self.sampling_bounds = np.array(
            [
                self._sampling_bound(q)
                for q in (protocol.base["q_save"], [protocol.base["q_spend"]] * 3)
            ]
        )
        # L diag(Vbar) is a PSD upper bound; a diagonal of Vbar alone is not.
        self.extra = np.array([np.eye(4) * 4 * v for v in self.sampling_bounds])
        self.observation_rows = []
        self.public_coefficients = None
        self.public_coefficient_round = 0
        self.public_coefficient_history = []
        self.public_forecast_coefficients = None

    def set_public_coefficients(self, t, report):
        coefficients = np.array(report["coefficients"], dtype=float)
        if (
            self.cfg["utility_proxy"] != "public_curvature_signal"
            or self.prepared_round != t
            or self.last_observed != t - 1
            or self.public_coefficient_round == t
            or report["round"] != t
            or report["names"] != self.names
            or report["public_data_sha256"] != self.cfg["public_proxy_sha256"]
            or coefficients.shape != (4,)
            or not np.isfinite(coefficients).all()
            or (coefficients <= 0).any()
            or not np.isclose(
                coefficients.sum(), sum(self.protocol.dimensions.values()), rtol=1e-12
            )
        ):
            raise ValueError(
                "Public curvature must come from the preceding DP model and fixed public input"
            )
        self.public_coefficients = coefficients.copy()
        self.public_coefficient_round = t
        if self.cfg["curvature_forecast"] == "damped_log_trend":
            from .dvpa_forecast import forecast_coefficients, parameters_from_config

            if len(self.public_coefficient_history) != t - 1:
                raise ValueError(
                    "Public forecast history must be a complete causal prefix"
                )
            self.public_coefficient_history.append(coefficients.copy())
            self.public_forecast_coefficients = forecast_coefficients(
                self.public_coefficient_history,
                np.arange(self.protocol.rounds - t + 1),
                **parameters_from_config(self.cfg),
            )

    def public_forecast_record(self):
        if self.cfg["curvature_forecast"] == "hold_current":
            return {}
        from .dvpa_forecast import parameters_from_config

        return dict(
            forecast="causal damped relative-log trend; hold horizon endpoint",
            forecast_parameters=parameters_from_config(self.cfg),
            forecast_observed_rounds=len(self.public_coefficient_history),
            forecast_coefficients=self.public_forecast_coefficients.tolist(),
            forecast_scope="deterministic given past public curvature; virtual observations update original direction belief only",
        )

    def _sampling_bound(self, rates):
        q = np.asarray(rates)
        return float(
            np.sum(self.protocol.counts * q * (1 - q) * self.protocol.radii**2)
            / self.cfg["reference_clip"] ** 2
        )

    def phase(self, t):
        return int(t >= self.protocol.base["transition_round"])

    def indices(self, t):
        return np.arange(4) + 4 * self.phase(t)

    def prepare(self, t):
        if (
            type(t) is not int
            or t != self.last_observed + 1
            or self.prepared_round == t
        ):
            raise ValueError("Prediction requires exactly previous published history")
        if t == self.cfg["warmup_rounds"] + 1:
            old = self.direction.for_round(t)
            rng = np.random.default_rng(
                self.protocol.config["controller"]["public_seed"] + 9173
            )
            self.basis = {}
            for n in self.names:
                if old[n] is None:
                    value = torch.from_numpy(
                        rng.choice([-1.0, 1.0], size=self.protocol.dimensions[n])
                    )
                    old[n] = value / torch.linalg.vector_norm(value)
                self.basis[n] = old[n].clone()
            self.basis_hash = digest(
                {n: direction_hash(self.basis[n]) for n in self.names}
            )
            # Warmup chosen the basis; never reuse those outputs as independent
            # observations in the new coordinate system.
            self.mean = np.zeros(8)
            self.cov = self.cfg["prior_std"] ** 2 * self.correlation
        r = self.cfg["mean_reversion"]
        self.pred_mean = r * self.mean
        self.pred_cov = psd(r * r * self.cov + self.process_cov)
        self.prepared_round = t
        return self.pred_mean.copy(), self.pred_cov.copy()

    def dp_covariance(self, z, p):
        p = np.asarray(p, dtype=float)
        if (
            not math.isfinite(z)
            or z <= 0
            or p.shape != (4,)
            or not np.isfinite(p).all()
            or (p <= 0).any()
        ):
            raise ValueError("Invalid observation precision")
        return np.diag(1 / (self.cfg["reference_clip"] ** 2 * z * p))

    def observation_covariance(self, t, z, p):
        return self.extra[self.phase(t)] + self.dp_covariance(z, p)

    def posterior(self, t, z, p, observations=None, standard_normals=None):
        if self.prepared_round != t:
            raise ValueError("Predict before planning observations")
        ids = self.indices(t)
        R = self.observation_covariance(t, z, p)
        S = psd(self.pred_cov[np.ix_(ids, ids)] + R, 1e-12)
        gain = np.linalg.solve(S, self.pred_cov[ids, :]).T
        if standard_normals is not None:
            innovations = np.asarray(standard_normals) @ np.linalg.cholesky(S).T
        else:
            innovations = np.atleast_2d(observations) - self.pred_mean[ids]
        means = self.pred_mean + innovations @ gain.T
        H = np.eye(8)[ids]
        A = np.eye(8) - gain @ H
        covariance = psd(A @ self.pred_cov @ A.T + gain @ R @ gain.T)
        # A working-model information measure only, not privacy leakage.
        mutual_information = 0.5 * (np.linalg.slogdet(S)[1] - np.linalg.slogdet(R)[1])
        return means, covariance, float(mutual_information)

    def expected_costs(self, means, covariance, rounds, origin, quadrature_nodes=None):
        """E[K], never E[oracle minimum]. Batch x future rounds x four blocks."""
        means = np.atleast_2d(means)
        rounds = np.asarray(rounds, dtype=int)
        steps = rounds - origin
        if (steps < 0).any():
            raise ValueError("Cannot forecast backwards")
        r = self.cfg["mean_reversion"]
        powers = r**steps
        qfactor = (1 - powers**2) / (1 - r * r)
        projected_mean = means[:, None, :] * powers[None, :, None]
        var = (
            powers[:, None] ** 2 * np.diag(covariance)[None, :]
            + qfactor[:, None] * np.diag(self.process_cov)[None, :]
        )
        ids = np.array([self.indices(int(t)) for t in rounds])
        mu = np.take_along_axis(projected_mean, ids[None, :, :], axis=2)
        variance = np.take_along_axis(var, ids, axis=1)
        nodes, weights = hermgauss(
            self.protocol.config["controller"]["quadrature_nodes"]
            if quadrature_nodes is None
            else quadrature_nodes
        )
        points = (
            mu[..., None]
            + np.sqrt(2 * np.maximum(variance, 0))[None, :, :, None] * nodes
        )
        squared = points**2
        saturated = np.sum(
            (squared / (self.cfg["signal_scale"] ** 2 + squared)) * weights, axis=-1
        ) / math.sqrt(math.pi)
        importance = self.cfg["lambda0"] + self.cfg["lambda1"] * saturated
        layer_cost = np.array(list(self.protocol.dimensions.values()))
        if self.cfg["utility_proxy"] == "public_curvature_signal":
            if (
                self.public_coefficient_round != self.prepared_round
                or self.public_coefficients is None
            ):
                raise ValueError("Prepare public curvature before scoring actions")
            layer_cost = self.public_coefficients
        layer_costs = np.broadcast_to(layer_cost, (len(rounds), 4))
        if self.cfg["curvature_forecast"] == "damped_log_trend":
            if (
                origin != self.public_coefficient_round
                or self.public_forecast_coefficients is None
            ):
                raise ValueError("Future scoring requires the current causal forecast")
            layer_costs = self.public_forecast_coefficients[steps]
        coefficient = layer_costs * self.protocol.scales[rounds - 1, None] ** 2 / 4
        return importance * coefficient[None, :, :]

    def observe(self, t, released, certified):
        action = certified.action
        if (
            self.prepared_round != t
            or self.last_observed != t - 1
            or action.round_index != t
            or action.history_hash != self.history_hash
            or action.feedback_basis_hash != self.basis_hash
        ):
            raise ValueError("Feedback action/history/basis sequence mismatch")
        if set(released) != set(self.names) or any(
            released[n].numel() != self.protocol.dimensions[n] for n in self.names
        ):
            raise ValueError("Invalid released update shape")
        new_hash = release_hash(released, self.names)
        rows = []
        if self.basis is not None:
            scale = certified.source_plan.final_scale * self.cfg["reference_clip"] * 0.5
            x = np.array(
                [
                    float(
                        torch.dot(
                            self.basis[n], released[n].detach().cpu().double().flatten()
                        )
                    )
                    / scale
                    for n in self.names
                ]
            )
            p = np.array([v for _, v in action.layer_precision])
            means, cov, _ = self.posterior(t, action.precision, p, observations=x)
            ids = self.indices(t)
            residual = x - self.pred_mean[ids]
            old_extra = self.extra[self.phase(t)].copy()
            if self.cfg["extra_covariance"] == "dp_innovation":
                estimate = (
                    np.outer(residual, residual)
                    - self.pred_cov[np.ix_(ids, ids)]
                    - self.dp_covariance(action.precision, p)
                )
                v = 4 * self.sampling_bounds[self.phase(t)]
                # Public q=1 has zero sampling variance, but model mismatch can
                # remain; the explicit regularizer remains positive.
                ceiling = max(v, self.cfg["covariance_floor"])
                estimate = psd(estimate, self.cfg["covariance_floor"], ceiling)
                eta = self.cfg["innovation_ema"]
                shrink = self.cfg["covariance_shrinkage"]
                updated = (1 - eta) * old_extra + eta * estimate
                updated = (1 - shrink) * updated + shrink * np.eye(4) * ceiling
                self.extra[self.phase(t)] = psd(
                    updated, self.cfg["covariance_floor"], ceiling
                )
            self.mean = means[0]
            self.cov = cov
            for j, n in enumerate(self.names):
                rows.append(
                    dict(
                        round=t,
                        layer=n,
                        phase=self.phase(t),
                        projection=float(x[j]),
                        prediction=float(self.pred_mean[ids[j]]),
                        residual=float(residual[j]),
                        dp_variance=float(
                            self.dp_covariance(action.precision, p)[j, j]
                        ),
                        planning_extra_variance=float(old_extra[j, j]),
                        next_extra_variance=float(self.extra[self.phase(t), j, j]),
                        basis_hash=self.basis_hash,
                    )
                )
        else:
            self.mean = self.pred_mean.copy()
            self.cov = self.pred_cov.copy()
        self.direction.observe(t, released)
        self.history_hash = digest(
            dict(previous=self.history_hash, round=t, release=new_hash)
        )
        self.last_observed = t
        self.observation_rows.extend(rows)
        return rows

    def state_dict(self):
        state = dict(
            mean=self.mean.tolist(),
            covariance=self.cov.tolist(),
            extra_covariance=self.extra.tolist(),
            basis=self.basis,
            basis_hash=self.basis_hash,
            history_hash=self.history_hash,
            last_observed=self.last_observed,
            direction=self.direction.state_dict(),
        )
        if self.cfg["utility_proxy"] == "public_curvature_signal":
            state["public_proxy"] = dict(
                coefficients=None
                if self.public_coefficients is None
                else self.public_coefficients.tolist(),
                round=self.public_coefficient_round,
            )
            if self.cfg["curvature_forecast"] == "damped_log_trend":
                state["public_proxy"]["coefficient_history"] = [
                    v.tolist() for v in self.public_coefficient_history
                ]
                state["public_proxy"]["forecast_coefficients"] = (
                    None
                    if self.public_forecast_coefficients is None
                    else self.public_forecast_coefficients.tolist()
                )
        return state
