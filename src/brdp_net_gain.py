"""Public fixed-geometry net-gain checks; never changes training or clipping.

Gaussian reference uses direct SGM RDP on stable source orders >=1.1.
BR-DP uses the existing analytic density-sandwich / joint-inclusion bound.
Matching means BR-DP's every group guarantee is no weaker than Gaussian's;
group clip radii, Poisson q, time/layer weights and signal are identical.
Numerical bounds remain research floating-point, not interval certificates.
"""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from opacus.accountants.analysis.rdp import compute_rdp
from scipy.stats import norm

from .brdp_group_accountant import (
    epsilon_from_rdp,
    joint_round_rdp,
    source_orders,
    sampled_gaussian_rdp_upper,
)
from .brdp_noise import absolute_tail, moments, parameters, sample_scalar_noise


def stable_orders():
    # The removed near-one source orders give very poor delta conversion at
    # the registered epsilon<=100. Removing orders never weakens a guarantee.
    return source_orders()[source_orders() >= 1.1]


@dataclass
class Geometry:
    name: str
    q: np.ndarray  # rounds x groups
    radii: np.ndarray  # rounds x groups, fixed raw clipping radii
    scales: np.ndarray  # rounds; public final aggregation factor
    clip_weights: np.ndarray  # rounds x layers
    base_stds: np.ndarray  # rounds x layers, fixed allocation profile
    dimensions: tuple
    targets: np.ndarray
    warmup: int

    def __post_init__(self):
        self.q = np.asarray(self.q, dtype=float)
        self.radii = np.asarray(self.radii, dtype=float)
        self.scales = np.asarray(self.scales, dtype=float)
        self.clip_weights = np.asarray(self.clip_weights, dtype=float)
        self.base_stds = np.asarray(self.base_stds, dtype=float)
        self.targets = np.asarray(self.targets, dtype=float)
        t, g = self.q.shape
        k = len(self.dimensions)
        if (
            self.radii.shape != (t, g)
            or self.scales.shape != (t,)
            or self.clip_weights.shape != (t, k)
            or self.base_stds.shape != (t, k)
            or self.targets.shape != (g,)
        ):
            raise ValueError("Geometry shapes do not match")
        arrays = [
            self.q,
            self.radii,
            self.scales,
            self.clip_weights,
            self.base_stds,
            self.targets,
        ]
        if (
            any(not np.isfinite(a).all() or (a <= 0).any() for a in arrays)
            or (self.q > 1).any()
        ):
            raise ValueError("Require finite positive geometry and valid probabilities")
        if (
            not np.allclose(self.clip_weights.sum(axis=1), 1)
            or any(type(d) is not int or d < 1 for d in self.dimensions)
            or not 0 <= self.warmup < t
        ):
            raise ValueError("Invalid clip weights, dimensions or warmup")

    @property
    def coefficients(self):
        # ACTUAL squared joint whitened sensitivity before the common noise
        # multiplier; combines all layers before Poisson amplification.
        factor = np.sum(self.clip_weights / self.base_stds**2, axis=1)
        return self.radii**2 * self.scales[:, None] ** 2 * factor[:, None]


def static_geometry(name, q, rounds, epsilon, blocks=1, dimension=128, sensitivity=1.0):
    return Geometry(
        name,
        np.full((rounds, 1), q),
        np.full((rounds, 1), sensitivity),
        np.ones(rounds),
        np.full((rounds, blocks), 1 / blocks),
        np.ones((rounds, blocks)),
        (dimension,) * blocks,
        np.array([epsilon]),
        0,
    )


def saved_geometry(name, preflight, dimensions):
    plans = preflight["plans"]
    names = list(dimensions)
    return Geometry(
        name,
        np.array([p["group_rates"] for p in plans]),
        np.array([p["group_radii"] for p in plans]),
        np.array([p["final_scale"] for p in plans]),
        np.array([[dict(p["layer_clip_weights"])[g] for g in names] for p in plans]),
        np.array([[dict(p["layer_noise_std"])[g] for g in names] for p in plans]),
        tuple(int(dimensions[g]) for g in names),
        np.array(preflight["targets"]),
        3,
    )


@lru_cache(maxsize=32768)
def _gaussian_detail(q, squared, orders):
    if q == 1:
        return np.asarray(orders) * squared / 2, ()
    if squared < 1e-8:
        # Stable analytic upper bound at near-infinite-noise bracket points.
        # Direct fractional SGM expressions can cancel to a negative number.
        return sampled_gaussian_rdp_upper(q, squared, np.asarray(orders)), tuple(orders)
    v = np.asarray(
        compute_rdp(
            q=q, noise_multiplier=1 / math.sqrt(squared), steps=1, orders=orders
        ),
        dtype=float,
    )
    upper = sampled_gaussian_rdp_upper(q, squared, np.asarray(orders))
    # Some high fractional orders in the installed Opacus return negative RDP
    # even for ordinary q=.5,sigma=16. Never clip such values to zero. Use
    # the analytic log-moment convex upper bound for inconsistent entries.
    floor = np.floor(np.asarray(orders))
    lower = np.zeros_like(v)
    valid = floor >= 2
    lower[valid] = sampled_gaussian_rdp_upper(q, squared, floor[valid])
    tolerance = 1e-9 + 1e-9 * np.maximum(upper, 1.0)
    bad = (
        (~np.isfinite(v)) | (v < 0) | (v < lower - tolerance) | (v > upper + tolerance)
    )
    values = np.where(bad, upper, v)
    return values * (1 + 1e-12) + 1e-14, tuple(np.asarray(orders)[bad].tolist())


def _gaussian(q, squared, orders):
    return _gaussian_detail(q, squared, orders)[0]


class FixedGeometryAccountant:
    def __init__(self, geometry, orders=None):
        self.geometry = geometry
        self.orders = (
            stable_orders() if orders is None else np.asarray(orders, dtype=float)
        )
        self._phases = []
        a = geometry.coefficients
        for g in range(geometry.q.shape[1]):
            self._phases.append(
                Counter(
                    (
                        float(geometry.q[t, g]),
                        float(a[t, g]),
                        bool(t >= geometry.warmup),
                    )
                    for t in range(len(geometry.q))
                )
            )

    def costs(self, multiplier, r, tau, mode="brdp"):
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("Positive noise multiplier required")
        parameters(1.0, tau, r)
        if mode not in {"gaussian", "reference_bound", "brdp"}:
            raise ValueError("Unknown account mode")
        result = []
        blocks = len(self.geometry.dimensions)
        for phases in self._phases:
            total = np.zeros(len(self.orders))
            for (q, a, active), count in phases.items():
                if mode == "gaussian" or (mode == "brdp" and r == 0):
                    v = _gaussian(q, a / multiplier**2, tuple(self.orders))
                else:
                    rr = r if active and mode == "brdp" else 0.0
                    v = joint_round_rdp(
                        q, a / multiplier**2, self.orders, [(rr, tau)] * blocks
                    )
                total += count * v
            result.append(total)
        return np.asarray(result)

    def epsilons(self, multiplier, r, tau, delta, mode="brdp"):
        costs = self.costs(multiplier, r, tau, mode)
        converted = [
            epsilon_from_rdp(self.orders, v, delta, warn_boundary=False) for v in costs
        ]
        return np.array([v[0] for v in converted]), [v[1] for v in converted]

    def calibrate(self, targets, r, tau, delta, mode="brdp", reserve=1e-7):
        targets = np.asarray(targets, dtype=float)
        if (
            targets.shape != self.geometry.targets.shape
            or not np.isfinite(targets).all()
            or (targets <= 0).any()
        ):
            raise ValueError("Invalid group guarantee targets")
        desired = targets * (1 - reserve)
        floor, _ = self.epsilons(1e12, r, tau, delta, mode)
        if (floor > desired).any():
            return dict(
                status="BOUND_INFEASIBLE",
                bound_floor_epsilon=floor.tolist(),
                targets=targets.tolist(),
            )
        low, high = 1e-6, 1.0
        while (self.epsilons(low, r, tau, delta, mode)[0] <= desired).all():
            low /= 2
        while (self.epsilons(high, r, tau, delta, mode)[0] > desired).any():
            high *= 2
        for _ in range(38):
            mid = math.sqrt(low * high)
            if (self.epsilons(mid, r, tau, delta, mode)[0] <= desired).all():
                high = mid
            else:
                low = mid
        eps, alphas = self.epsilons(high, r, tau, delta, mode)
        if (eps > targets).any():
            raise ArithmeticError("Calibration exceeded common guarantees")
        fallback = set()
        if mode == "gaussian" or (mode == "brdp" and r == 0):
            for phases in self._phases:
                for q, a, _ in phases:
                    fallback.update(
                        _gaussian_detail(q, a / high**2, tuple(self.orders))[1]
                    )
        return dict(
            status="FEASIBLE",
            multiplier=high,
            epsilon=eps.tolist(),
            best_alpha=alphas,
            worst_cap_utilization=float(np.max(eps / targets)),
            gaussian_fractional_fallback_orders=sorted(fallback),
        )


def gain_metrics(scale_ratio, r, tau, dimension):
    """Exact ideal/perfect and misaligned direction MSE conditions, no training."""
    if (
        not math.isfinite(scale_ratio)
        or scale_ratio <= 0
        or type(dimension) is not int
        or dimension < 1
    ):
        raise ValueError("Invalid scale ratio/dimension")
    v = moments(1.0, tau, r)["variance"]
    s = scale_ratio**2
    perfect = s * v
    threshold = (1 - 1 / s) / (1 - v) if v < 1 else None
    alignment = (
        max(0.0, float(threshold)) if threshold is not None and threshold < 1 else None
    )
    result = dict(
        fixed_sigma_variance_factor=v,
        noise_std_ratio=scale_ratio,
        noise_variance_inflation=s,
        max_std_ratio_for_perfect_direction_gain=1 / math.sqrt(v),
        ideal_parallel_mse_ratio=perfect,
        energy_ratio=s * (dimension - 1 + v) / dimension,
        minimum_cosine_squared_for_direction_gain=alignment,
        even_perfect_alignment_fails=bool(perfect >= 1 - 1e-6),
    )
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result[f"mse_ratio_cos2_{a:g}"] = s * (1 - (1 - v) * a)
    for k in [1.0, 2.0, 3.0]:
        # Threshold in the SAME absolute units, k*sigma_G, for both laws.
        result[f"tail_ratio_at_{k:g}_gaussian_std"] = absolute_tail(
            k, scale_ratio, tau, r
        ) / (2 * norm.sf(k))
    return result


def profile_metrics(geometry, scale_ratio, r, tau):
    result = gain_metrics(scale_ratio, r, tau, max(geometry.dimensions))
    variances = geometry.base_stds**2
    factors = np.ones_like(variances)
    factors[geometry.warmup :] = moments(1.0, tau, r)["variance"]
    d = np.asarray(geometry.dimensions)
    result["trajectory_protected_axes_mse_ratio"] = float(
        scale_ratio**2 * np.sum(variances * factors) / np.sum(variances)
    )
    result["trajectory_total_energy_ratio"] = float(
        scale_ratio**2 * np.sum(variances * (d - 1 + factors)) / np.sum(variances * d)
    )
    result["trajectory_layer_direction_ratios"] = (
        scale_ratio**2 * np.sum(variances * factors, axis=0) / variances.sum(axis=0)
    ).tolist()
    return result


def public_mc_case(sigma_g, sigma_b, r, tau, dimension, n, seed=42):
    """PUBLIC independent synthetic draws; never replay training noise/RNG.

    Chi-square orthogonal energy avoids allocating dimension*n arrays. The
    true useful direction is public and has known squared cosine to the
    recycled axis. This is conditional geometry validation, not measurement
    of a private true gradient in FMNIST.
    """
    if n < 1000 or dimension < 1:
        raise ValueError("Require adequate samples and dimension")
    rng = np.random.default_rng(seed)
    b = sample_scalar_noise(sigma_b, tau, r, rng, n)
    g = rng.normal(0, sigma_g, n)
    orth_b = rng.normal(0, sigma_b, n) if dimension > 1 else np.zeros(n)
    orth_g = rng.normal(0, sigma_g, n) if dimension > 1 else np.zeros(n)
    target = moments(sigma_b, tau, r)["variance"]
    empirical = float(np.mean(b * b))
    se = float(np.std(b * b, ddof=1) / math.sqrt(n))
    result = dict(
        scope="public synthetic noise, no training data or released training noise",
        n=n,
        public_seed=seed,
        gaussian_sigma=sigma_g,
        brdp_sigma=sigma_b,
        theoretical_parallel_variance=target,
        empirical_parallel_mse=empirical,
        mse_standard_error=se,
        moment_check_passed=bool(abs(empirical - target) <= 6 * se),
        mean=float(b.mean()),
    )
    for a in [0.0, 0.5, 1.0] if dimension > 1 else [1.0]:
        pb = math.sqrt(a) * b + math.sqrt(1 - a) * orth_b
        pg = math.sqrt(a) * g + math.sqrt(1 - a) * orth_g
        result[f"empirical_mse_ratio_cos2_{a:g}"] = float(
            np.mean(pb * pb) / np.mean(pg * pg)
        )
        result[f"theoretical_mse_ratio_cos2_{a:g}"] = (sigma_b / sigma_g) ** 2 * (
            1 - (1 - moments(1.0, tau, r)["variance"]) * a
        )
    for k in [1, 2, 3]:
        probability = float(np.mean(np.abs(b) > k * sigma_g))
        expected = absolute_tail(k * sigma_g, sigma_b, tau, r)
        se_tail = math.sqrt(max(expected * (1 - expected), 1 / n) / n)
        result[f"tail_{k}_empirical"] = probability
        result[f"tail_{k}_theory"] = expected
        result[f"tail_{k}_check_passed"] = bool(
            abs(probability - expected) <= 6 * se_tail + 1 / n
        )
    eb = b * b + orth_b * orth_b
    eg = g * g + orth_g * orth_g
    if dimension > 2:
        eb += sigma_b**2 * rng.chisquare(dimension - 2, n)
        eg += sigma_g**2 * rng.chisquare(dimension - 2, n)
    expected_energy = target + (dimension - 1) * sigma_b**2
    energy_se = float(np.std(eb, ddof=1) / math.sqrt(n))
    result.update(
        empirical_energy_ratio=float(eb.mean() / eg.mean()),
        theoretical_energy_ratio=expected_energy / (dimension * sigma_g**2),
        energy_check_passed=bool(
            abs(float(eb.mean()) - expected_energy) <= 6 * energy_se
        ),
    )
    return result
