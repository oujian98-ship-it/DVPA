"""Prospective, fixed-order per-group Renyi filtering for adaptive Gaussian FL.

The certificate covers ALL histories, using one public order/cap per group.
Realized charges are diagnostics, not an ex-post tighter DP guarantee.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
import copy
import hashlib
import json
import math

import numpy as np
from scipy.special import gammaln, logsumexp

from .brdp_source_protocol import SourceRoundPlan

VERSION = "dvpa_fixed_order_joint_sgm_filter_v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@lru_cache(maxsize=512)
def _terms(alpha, q):
    if type(alpha) is not int or alpha < 2 or not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("Invalid public SGM order/rate")
    j = np.arange(alpha + 1, dtype=float)
    if q in (0, 1):
        return j, None
    return j, (
        gammaln(alpha + 1)
        - gammaln(j + 1)
        - gammaln(alpha - j + 1)
        + j * math.log(q)
        + (alpha - j) * math.log1p(-q)
    )


def sgm_cost(alpha, q, squared):
    """Vectorized integer moments, same safety margin as the reference primitive."""
    if type(alpha) is not int or alpha < 2:
        raise ValueError("Integer Renyi order required")
    squared = np.asarray(squared, dtype=float)
    if not np.isfinite(squared).all() or (squared < 0).any():
        raise ValueError("Invalid whitened sensitivity")
    j, logs = _terms(int(alpha), float(q))
    if q == 0:
        return np.zeros_like(squared)
    if q == 1:
        return alpha * squared / 2
    values = logsumexp(logs + squared[..., None] * j * (j - 1) / 2, axis=-1) / (
        alpha - 1
    )
    return np.where(squared == 0, 0.0, np.maximum(values, 0.0) * (1 + 1e-12) + 1e-14)


@dataclass(frozen=True)
class PrecisionAction:
    round_index: int
    precision: float
    layer_precision: tuple
    feedback_basis_hash: str
    history_hash: str

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CertifiedPlan:
    action: PrecisionAction
    source_plan: SourceRoundPlan
    group_rdp_increment: tuple
    plan_hash: str

    def as_dict(self):
        return asdict(self)


def to_source_plan(action, protocol):
    t = action.round_index
    if type(t) is not int or not 1 <= t <= protocol.rounds:
        raise ValueError("Invalid action round")
    pairs = action.layer_precision
    if [n for n, _ in pairs] != list(protocol.dimensions):
        raise ValueError("Layer precision coverage/order mismatch")
    p = np.asarray([v for _, v in pairs], dtype=float)
    z = action.precision
    if (
        isinstance(z, bool)
        or not math.isfinite(z)
        or z <= 0
        or not np.isfinite(p).all()
        or (p < protocol.p_min - 1e-14).any()
        or not math.isclose(float(p.sum()), 1.0, abs_tol=1e-12)
    ):
        raise ValueError("Invalid physical precision action")
    scale = protocol.scales[t - 1]
    omega = 1 / len(p)
    return SourceRoundPlan(
        t,
        tuple(protocol.rates[t - 1]),
        float(protocol.denominators[t - 1]),
        tuple(protocol.radii),
        float(scale),
        None,
        tuple((n, omega) for n in protocol.dimensions),
        pairs,
        tuple((n, float(scale * math.sqrt(omega / (z * v)))) for n, v in pairs),
        tuple((n, 0.0, 1.0) for n in protocol.dimensions),
    )


class GroupRDPFilter:
    def __init__(self, protocol):
        self.protocol = protocol
        self.spent = np.zeros(3)
        self.receipts = []

    def remaining(self):
        return self.protocol.caps.copy() - self.spent

    def certify(self, action, source_plan=None):
        """Validate actual noise/clip/sampling fields; never trust a reported cost."""
        expected = to_source_plan(action, self.protocol)
        plan = expected if source_plan is None else source_plan
        if digest(plan.as_dict()) != digest(expected.as_dict()):
            raise ValueError("Actual release parameters differ from precision action")
        weights = dict(plan.layer_clip_weights)
        stds = dict(plan.layer_noise_std)
        # Reconstruct physical precision and proportions from the actual sigma.
        inv = np.array(
            [plan.final_scale**2 * weights[n] / stds[n] ** 2 for n in weights]
        )
        z = float(inv.sum())
        if not np.isclose(z, action.precision, rtol=1e-12, atol=0.0):
            raise ValueError("Precision/noise mismatch")
        if not np.allclose(
            inv / z, [v for _, v in action.layer_precision], rtol=1e-12, atol=1e-14
        ):
            raise ValueError("Layer/noise mismatch")
        costs = []
        for g, (radius, q, alpha) in enumerate(
            zip(plan.group_radii, plan.group_rates, self.protocol.orders, strict=True)
        ):
            squared = sum(
                (plan.final_scale * radius * math.sqrt(weights[n]) / stds[n]) ** 2
                for n in weights
            )
            cost = float(sgm_cost(int(alpha), q, squared))
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("Invalid conditional privacy cost")
            costs.append(cost)
        payload = dict(
            action=action.as_dict(),
            source_plan=plan.as_dict(),
            group_rdp_increment=costs,
        )
        return CertifiedPlan(action, plan, tuple(costs), digest(payload))

    def can_reserve(self, certified):
        try:
            fresh = self.certify(certified.action, certified.source_plan)
        except (ValueError, OverflowError):
            return False
        t = certified.action.round_index
        if (
            fresh.plan_hash != certified.plan_hash
            or fresh.group_rdp_increment != certified.group_rdp_increment
        ):
            return False
        if t != len(self.receipts) + 1:
            return False
        return bool(
            np.all(
                self.spent + np.array(fresh.group_rdp_increment) <= self.protocol.caps
            )
        )

    def reserve(self, certified):
        if not self.can_reserve(certified):
            raise RuntimeError(
                "Privacy filter rejected before accessing current updates"
            )
        new_spent = self.spent + np.array(certified.group_rdp_increment)
        receipt = dict(
            round=certified.action.round_index,
            plan_hash=certified.plan_hash,
            increment=list(certified.group_rdp_increment),
            spent=new_spent.tolist(),
            previous_receipt_hash=self.receipts[-1]["receipt_hash"]
            if self.receipts
            else "genesis",
        )
        receipt["receipt_hash"] = digest(receipt)
        self.spent = new_spent
        self.receipts.append(receipt)
        return copy.deepcopy(receipt)

    def rows(self):
        t = len(self.receipts)
        conversion = np.log(1 / self.protocol.base["delta"]) / (
            self.protocol.orders - 1
        )
        return [
            dict(
                round=t,
                group=g,
                alpha=int(self.protocol.orders[g]),
                rdp_cap=float(self.protocol.caps[g]),
                rdp_spent=float(self.spent[g]),
                rdp_remaining=float(self.remaining()[g]),
                rdp_increment=float(self.receipts[-1]["increment"][g]) if t else 0.0,
                epsilon_guarantee=float(self.protocol.base["epsilons"][g])
                if t
                else 0.0,
                delta=self.protocol.base["delta"],
                epsilon_path_diagnostic=float(self.spent[g] + conversion[g])
                if t
                else 0.0,
                is_global_guarantee=False,
            )
            for g in range(3)
        ]

    def state_dict(self):
        return dict(
            version=VERSION,
            spent=self.spent.tolist(),
            receipts=copy.deepcopy(self.receipts),
        )
