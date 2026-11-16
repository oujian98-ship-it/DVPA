"""Grouped Poisson DP-FL accounting for soft-recycled Gaussian block noise.

For reference sampled-Gaussian laws P0,Q0, soft recycling gives
 m/Z <= P/P0,Q/Q0 <= 1/Z (products across independent blocks).
Thus D_alpha(P||Q) <= SGM_alpha + sum[-log(m)-log(Z)/(alpha-1)].
An independent valid bound reveals the Bernoulli inclusion indicator and uses
joint convexity: log(1-q+q*exp((alpha-1)*base_RDP))/(alpha-1).
Take the smaller upper bound ONCE for the whole joint round, never per block.

Scope: add/remove a bounded contribution in a fixed client slot; public fixed
denominator; fresh independent noise; all planning deterministic before training.
Fractional reference SGM moments use convex interpolation of integer log moments.
Only generic RDP->epsilon conversion is delegated to Opacus.
"""

from functools import lru_cache
import math
import warnings

import numpy as np
from opacus.accountants.analysis.rdp import get_privacy_spent
from scipy.special import gammaln, logsumexp

from .brdp_noise import parameters

ACCOUNTANT_VERSION = "poisson_gaussian_reference_soft_density_v1"


def source_orders():
    return np.asarray(
        [1 + x / 1e10 for x in range(1, 100)]
        + [1 + x / 1e7 for x in range(1, 100)]
        + [1 + x / 1e4 for x in range(1, 100)]
        + [1 + x / 10 for x in range(1, 9)]
        + [x / 2 for x in range(4, 300)],
        dtype=np.float64,
    )


@lru_cache(maxsize=16)
def _integer_grid(maximum):
    n = np.arange(2, maximum + 1, dtype=float)[:, None]
    j = np.arange(maximum + 1, dtype=float)[None, :]
    valid = j <= n
    log_comb = np.full(np.broadcast_shapes(n.shape, j.shape), -np.inf)
    nn, jj = np.broadcast_arrays(n, j)
    log_comb[valid] = (
        gammaln(nn[valid] + 1)
        - gammaln(jj[valid] + 1)
        - gammaln(nn[valid] - jj[valid] + 1)
    )
    return n, j, valid, log_comb


def sampled_gaussian_rdp_upper(q, whitened_sensitivity_sq, orders):
    """Analytic integer SGM moments; convex log-moment interpolation above them."""
    orders = np.asarray(orders, dtype=np.float64)
    if orders.ndim != 1 or not np.isfinite(orders).all() or (orders <= 1).any():
        raise ValueError("Require finite RDP orders > 1")
    if (
        not math.isfinite(q)
        or not 0 <= q <= 1
        or not math.isfinite(whitened_sensitivity_sq)
        or whitened_sensitivity_sq < 0
    ):
        raise ValueError("Invalid sampling rate/sensitivity")
    if q == 0 or whitened_sensitivity_sq == 0:
        return np.zeros_like(orders)
    if q == 1:
        return orders * whitened_sensitivity_sq / 2
    maximum = int(np.ceil(orders.max()))
    n, j, valid, log_comb = _integer_grid(maximum)
    terms = (
        log_comb
        + j * math.log(q)
        + (n - j) * math.log1p(-q)
        + (j * j - j) * whitened_sensitivity_sq / 2
    )
    integer_moments = np.zeros(maximum + 1)
    integer_moments[2:] = logsumexp(np.where(valid, terms, -np.inf), axis=1)
    lower = np.floor(orders).astype(int)
    upper = np.ceil(orders).astype(int)
    fraction = orders - lower
    moments = (1 - fraction) * integer_moments[lower] + fraction * integer_moments[
        upper
    ]
    values = moments / (orders - 1)
    # Research numerical safety margin; not a hardened floating-point proof.
    return np.maximum(0.0, values) * (1 + 1e-12) + 1e-14


def recycling_penalty(orders, mechanisms):
    orders = np.asarray(orders)
    result = np.zeros_like(orders, dtype=float)
    for r, tau in mechanisms:
        _, _, r, _, log_z = parameters(1.0, tau, r)
        result += -math.log1p(-r) - log_z / (orders - 1)
    return result


def joint_round_rdp(q, whitened_sensitivity_sq, orders, mechanisms):
    orders = np.asarray(orders, dtype=float)
    reference = sampled_gaussian_rdp_upper(q, whitened_sensitivity_sq, orders)
    if q == 0 or whitened_sensitivity_sq == 0:
        return np.zeros_like(orders)
    penalty = recycling_penalty(orders, mechanisms)
    sandwich = reference + penalty
    base = orders * whitened_sensitivity_sq / 2 + penalty
    convex = np.logaddexp(
        math.log1p(-q) if q < 1 else -np.inf, math.log(q) + (orders - 1) * base
    ) / (orders - 1)
    value = np.minimum(sandwich, convex)
    return np.maximum(value, 0.0) * (1 + 1e-12) + 1e-14


def epsilon_from_rdp(orders, costs, delta, warn_boundary=True):
    costs = np.asarray(costs, dtype=float)
    if (
        costs.shape != np.asarray(orders).shape
        or not np.isfinite(costs).all()
        or (costs < 0).any()
    ):
        raise ValueError("Invalid RDP vector")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0,1)")
    if not costs.any():
        return 0.0, float(np.asarray(orders)[0])
    with warnings.catch_warnings():
        if not warn_boundary:
            # Only suppress bracketing-point warnings during PUBLIC calibration.
            # Final guarantees retain the normal boundary warning.
            warnings.filterwarnings("ignore", message="Optimal order is the .* alpha.*")
        eps, alpha = get_privacy_spent(orders=orders, rdp=costs, delta=delta)
    return max(0.0, float(eps)), float(alpha)


def round_group_costs(
    orders,
    group_radii,
    final_scale,
    layer_clip_weights,
    layer_noise_std,
    rates,
    mechanisms,
):
    """Recompute cost from ACTUAL clip/noise units, not nominal budget labels."""
    if set(layer_clip_weights) != set(layer_noise_std) or set(mechanisms) != set(
        layer_noise_std
    ):
        raise ValueError("Layer coverage mismatch")
    if not math.isfinite(final_scale) or final_scale <= 0:
        raise ValueError("Invalid lr*rate/denominator scale")
    squared = []
    shape_parameters = list(mechanisms.values())
    for radius, q in zip(group_radii, rates, strict=True):
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Invalid group clip radius")
        norm_sq = 0.0
        for layer, weight in layer_clip_weights.items():
            std = layer_noise_std[layer]
            if (
                not math.isfinite(std)
                or std <= 0
                or not math.isfinite(weight)
                or weight <= 0
            ):
                raise ValueError("Active layer requires positive clip weight and noise")
            norm_sq += (radius * math.sqrt(weight) * final_scale / std) ** 2
        squared.append(joint_round_rdp(float(q), norm_sq, orders, shape_parameters))
    return np.asarray(squared)


class GroupPrivacyLedger:
    """Precommitted uniform envelopes make multi-order prefix guarantees valid.

    Direction-dependent Gaussian fallback can LOWER actual path charges. The
    minimum over that realized vector is labelled a diagnostic; guarantees use
    the deterministic envelope covering every possible fallback history.
    """

    def __init__(self, orders, targets, delta, envelope):
        self.orders = np.asarray(orders, dtype=float)
        self.targets = np.asarray(targets, dtype=float)
        self.delta = delta
        self.envelope = np.asarray(envelope, dtype=float)
        if self.envelope.ndim != 3 or self.envelope.shape[1:] != (
            len(targets),
            len(orders),
        ):
            raise ValueError("Envelope must be rounds x groups x orders")
        if not np.isfinite(self.envelope).all() or (self.envelope < 0).any():
            raise ValueError("Invalid uniform privacy envelope")
        self.actual = np.zeros((len(targets), len(orders)))
        self.uniform = np.zeros_like(self.actual)
        self.history = []

    def accept(self, t, actual_increment):
        if t != len(self.history) + 1 or t > len(self.envelope):
            raise ValueError("Group releases must be sequential")
        costs = np.asarray(actual_increment, dtype=float)
        bound = self.envelope[t - 1]
        if (
            costs.shape != bound.shape
            or not np.isfinite(costs).all()
            or (costs < 0).any()
        ):
            raise ValueError("Invalid actual mechanism costs")
        if (costs > bound * (1 + 1e-9) + 1e-11).any():
            raise RuntimeError("Actual mechanism exceeds precommitted RDP envelope")
        uniform = self.uniform + bound
        actual = self.actual + costs
        rows = []
        for g, target in enumerate(self.targets):
            guarantee, guaranteed_alpha = epsilon_from_rdp(
                self.orders, uniform[g], self.delta
            )
            path, path_alpha = epsilon_from_rdp(self.orders, actual[g], self.delta)
            if guarantee > target:
                raise RuntimeError("Group privacy target exceeded before release")
            anchor = int(np.argmin(np.abs(self.orders - guaranteed_alpha)))
            rows.append(
                dict(
                    round=t,
                    group=g,
                    epsilon_target=float(target),
                    delta=self.delta,
                    epsilon_guarantee=guarantee,
                    epsilon_actual_path_accounting=path,
                    actual_path_value_is_global_guarantee=False,
                    guarantee_alpha=guaranteed_alpha,
                    path_best_alpha=path_alpha,
                    actual_rdp_at_guarantee_alpha=float(actual[g, anchor]),
                    uniform_rdp_at_guarantee_alpha=float(uniform[g, anchor]),
                )
            )
        self.actual, self.uniform = actual, uniform
        self.history.extend([rows])
        return rows
