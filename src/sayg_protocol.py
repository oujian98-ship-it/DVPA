"""Execute author Scheme2 spending methods with an independent release ledger.

The source epsilon-remainder recurrence, epochs=remaining_rounds Opacus API
semantics, sampling phases and harmonic clip calibration are preserved. Only
the clipping geometry and final aggregate noise delivery use the shared corrected
protocol. This is a documented source-method adaptation, not a claim that every
source implementation detail is a mathematically required paper specification.
"""

import copy
from functools import lru_cache
import hashlib
import math
import warnings

import numpy as np

from .brdp_group_accountant import (
    epsilon_from_rdp,
    source_orders,
    sampled_gaussian_rdp_upper,
)
from .brdp_source_protocol import SourceRoundPlan
from .original_scheduler import groups
from .upstream_runner import (
    SOURCE,
    build_source_classes,
    checked_source_noise,
    cached_rdp,
)


def gaussian_round_costs(
    orders, group_radii, final_scale, clip_weights, stds, rates, shapes
):
    """Gaussian reference accounting, once for the JOINT whitened update.

    Never use this on soft-recycled noise. Orders >=2 are evaluated directly.
    Orders below 2 use D_alpha <= D_2: the source contains orders just 1e-10
    above 1, where direct evaluation suffers cancellation under harmless scale
    rounding. The monotone bound avoids that instability while preserving the
    tighter direct computation for fractional orders >=2.
    """
    if (
        set(clip_weights) != set(stds)
        or set(shapes) != set(stds)
        or any(r != 0 for r, _ in shapes.values())
    ):
        raise ValueError(
            "Gaussian accountant requires Gaussian noise and complete layers"
        )
    rows = []
    for radius, q in zip(group_radii, rates, strict=True):
        squared = math.fsum(
            (radius * math.sqrt(clip_weights[g]) * final_scale / stds[g]) ** 2
            for g in stds
        )
        if not math.isfinite(squared) or squared <= 0:
            raise ValueError("Invalid Gaussian whitened sensitivity")
        value = np.array(
            cached_rdp(float(q), 1 / math.sqrt(squared), 1, tuple(orders)),
            dtype=float,
            copy=True,
        )
        low_order_bound = sampled_gaussian_rdp_upper(
            float(q), squared, np.array([2.0])
        )[0]
        value = np.where(np.asarray(orders) < 2.0, low_order_bound, value)
        # Opacus' direct fractional-order calculation can lose numerical
        # stability at high noise and large alpha (including negative RDP
        # values). Replace only invalid entries with the independently
        # computed log-moment/convex-interpolation upper bound.
        invalid = ~np.isfinite(value) | (value < 0)
        if invalid.any():
            stable_upper = sampled_gaussian_rdp_upper(
                float(q), squared, np.asarray(orders, dtype=float)
            )
            value[invalid] = stable_upper[invalid]
        if not np.isfinite(value).all() or (value < 0).any():
            raise ValueError("Invalid Gaussian RDP vector after stable fallback")
        rows.append(value * (1 + 1e-12) + 1e-14)
    return np.asarray(rows)


@lru_cache(maxsize=12)
def _source_trace(rounds, budgets, delta, q_save, q_spend, transition, source_digest):
    # AST-loaded classes have their own globals. No upstream file or shared
    # module namespace is modified, and no private training data is consulted.
    client_class, _, _ = build_source_classes("time_adaptive")
    orders = tuple(source_orders())

    def solve(**kw):
        # Only PUBLIC bracket-search warnings are suppressed. Actual prefix and
        # final RDP conversions below retain their boundary-order warnings.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Optimal order is the .* alpha.*")
            return checked_source_noise(tuple(sorted(kw.items())), orders)

    client_class.set_local_sigma_and_sample_rate.__globals__["get_noise_multiplier"] = (
        solve
    )
    clients = []
    for target in budgets:
        client = object.__new__(client_class)
        client.rounds = rounds
        client.dp_eps_budget = target
        client.dp_eps_spent = 0.0
        client.delta = delta
        client.alphas = list(orders)
        client.rdp_eps_spent = np.zeros(len(orders))
        clients.append(client)
    rows = []
    for t in range(1, rounds + 1):
        remaining = [c.dp_eps_budget - c.dp_eps_spent for c in clients]
        sigmas = [
            float(c.set_local_sigma_and_sample_rate(q_spend, t - 1)) for c in clients
        ]
        rates = list(q_save) if t < transition else [q_spend] * 3
        for c, q in zip(clients, rates, strict=True):
            c.rdp_accountant(sample_rate=q)
        rows.append(
            dict(
                round=t,
                remaining_epsilon_before=remaining,
                local_sigma=sigmas,
                group_rates=rates,
                source_epsilon_after=[float(c.dp_eps_spent) for c in clients],
                source_best_alpha=[float(c.best_alpha) for c in clients],
                get_noise_epochs=rounds - t + 1,
                get_noise_implied_steps=int((rounds - t + 1) / q_spend),
            )
        )
    return rows


def source_trace(cfg):
    path = SOURCE / "src/clients/client_scheme2_time_adaptive_dpfl.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = _source_trace(
        cfg["rounds"],
        tuple(cfg["epsilons"]),
        cfg["delta"],
        tuple(cfg["q_save"]),
        cfg["q_spend"],
        cfg["transition_round"],
        digest,
    )
    return copy.deepcopy(rows), digest


@lru_cache(maxsize=12)
def _idp_trace(rounds, budgets, delta, q, source_digest):
    client_class, _, _ = build_source_classes("idp_fedavg")
    orders = tuple(source_orders())

    def solve(**kw):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Optimal order is the .* alpha.*")
            return checked_source_noise(tuple(sorted(kw.items())), orders)

    client_class.set_local_sigma_and_sample_rate.__globals__["get_noise_multiplier"] = (
        solve
    )
    clients = []
    for target in budgets:
        c = object.__new__(client_class)
        c.rounds = rounds
        c.dp_eps_budget = target
        c.dp_eps_spent = 0.0
        c.delta = delta
        c.alphas = list(orders)
        c.rdp_eps_spent = np.zeros(len(orders))
        c.set_local_sigma_and_sample_rate(q)
        clients.append(c)
    rows = []
    for t in range(1, rounds + 1):
        remaining = [c.dp_eps_budget - c.dp_eps_spent for c in clients]
        for c in clients:
            c.rdp_accountant()
        rows.append(
            dict(
                round=t,
                remaining_epsilon_before=remaining,
                local_sigma=[float(c.local_noise_multiplier) for c in clients],
                group_rates=[q] * 3,
                source_epsilon_after=[float(c.dp_eps_spent) for c in clients],
                source_best_alpha=[float(c.best_alpha) for c in clients],
                get_noise_epochs=rounds,
                get_noise_implied_steps=int(rounds / q),
            )
        )
    return rows


def idp_trace(cfg):
    path = SOURCE / "src/clients/client_scheme1_baseline.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return copy.deepcopy(
        _idp_trace(
            cfg["rounds"], tuple(cfg["epsilons"]), cfg["delta"], cfg["q_spend"], digest
        )
    ), digest


def source_preflight(cfg, dimensions):
    names = list(dimensions)
    if set(names) != set(cfg["layer_decay"]):
        raise ValueError("Source layer registry mismatch")
    is_idp = cfg["method"] == "IDP_FEDAVG"
    trace, digest = idp_trace(cfg) if is_idp else source_trace(cfg)
    orders = source_orders()
    counts = np.bincount(groups(cfg["clients"], cfg["group_proportions"]), minlength=3)
    omega = dict.fromkeys(names, 1 / len(names))
    plans, costs = [], []
    for row in trace:
        sigmas = np.asarray(row["local_sigma"])
        global_sigma = 1 / float(np.dot(counts / counts.sum(), 1 / sigmas))
        radii = cfg["clip"] * global_sigma / sigmas
        rates = row["group_rates"]
        denominator = float(np.dot(counts, rates))
        scale = cfg["lr"] * cfg["rate"] / denominator
        stds = dict.fromkeys(names, cfg["clip"] * global_sigma * scale)
        plan = SourceRoundPlan(
            row["round"],
            tuple(rates),
            denominator,
            tuple(radii.tolist()),
            scale,
            None,
            tuple(omega.items()),
            tuple(omega.items()),
            tuple(stds.items()),
            tuple((name, 0.0, cfg["tau"]) for name in names),
        )
        plans.append(plan)
        costs.append(
            gaussian_round_costs(
                orders,
                plan.group_radii,
                scale,
                omega,
                stds,
                rates,
                {name: (0.0, cfg["tau"]) for name in names},
            )
        )
    envelope = np.asarray(costs)
    final = [
        epsilon_from_rdp(orders, envelope[:, g].sum(axis=0), cfg["delta"])[0]
        for g in range(3)
    ]
    if any(used > cap for used, cap in zip(final, cfg["epsilons"], strict=True)):
        raise RuntimeError("Actual corrected source release exceeds privacy caps")
    first_sigmas = np.asarray(trace[0]["local_sigma"])
    kappa = 1 / float(np.dot(counts / counts.sum(), 1 / first_sigmas))
    return dict(
        orders=orders,
        counts=counts,
        betas=1 / first_sigmas,
        kappa=kappa,
        group_radii=np.asarray(plans[0].group_radii),
        plans=plans,
        envelope=envelope,
        final_epsilon=final,
        schedule_metadata=dict(
            method="source_scheme1_constant_individualized_gaussian"
            if is_idp
            else "source_scheme2_epsilon_remainder_spend_as_you_go",
            source_client_sha256=digest,
            source_trace=trace,
            correction="joint weight/bias clipping and independent actual aggregate ledger",
            calibration="preserve source one-time epochs=rounds and validate same expanded orders"
            if is_idp
            else "preserve source epochs=remaining_rounds and validate same expanded orders",
            noise_delivery="equivalent aggregate Gaussian; trusted server aggregate observer",
            time_fraction="not applicable: source sigmas and clip radii vary by group/round",
        ),
    )
