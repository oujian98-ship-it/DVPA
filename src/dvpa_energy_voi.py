"""Fixed-z comparisons with counterfactual observations and independent MC."""

import hashlib
import itertools
import json

import numpy as np
from scipy.stats import chi2

from .dvpa_config import ROOT
from .dvpa_convex import ConvexPrecision
from .dvpa_energy_feedback import observation_geometry, virtual_energies
from .dvpa_energy_forecast import CurvaturePredictor, HORIZONS, public_features
from .dvpa_filter import GroupRDPFilter, PrecisionAction
from .dvpa_ranking import confirm_energy_gain


def gate_scope(config):
    base = config["base"]
    return dict(base={key: val for key, val in base.items() if key not in {"device", "seed", "data_root"}},
                public_proxy_sha256=config["feedback"]["public_proxy_sha256"],
                public_proxy_probes=config["feedback"]["public_proxy_probes"],
                curvature_parameters={key: config["feedback"][key] for key in (
                    "curvature_mode", "curvature_forecast", "forecast_horizon", "forecast_damping",
                    "forecast_ema", "forecast_clip", "energy_state", "warmup_rounds")},
                privacy=config["privacy"], p_min=config["controller"]["p_min"])


def load_energy_gate(config, snapshot=None):
    settings = config["controller"]
    path = (ROOT / settings["energy_gate_file"]).resolve() if snapshot is None else snapshot
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != settings["energy_gate_sha256"]:
        raise ValueError("Energy VOI gate hash mismatch")
    record = json.loads(data)
    if (record.get("version") != 1 or record.get("passed") != {"A": True, "B": True, "C": True}
            or record.get("scope") != gate_scope(config) or len(set(record.get("confirming_seeds", []))) < 2):
        raise ValueError("Energy VOI gate did not pass for this protocol")
    return CurvaturePredictor.from_record(record["predictor"]), data


def layer_candidates(p, fixed, floor, priorities):
    values = [("deterministic", p), ("uniform", np.full(len(p), 1/len(p))), ("dimension", fixed)]
    pairs = list(itertools.permutations(range(len(p)), 2))
    pairs.sort(key=lambda pair: -max(priorities[pair[0]], priorities[pair[1]]))
    for i, j in pairs:
        for transfer in (.02, .05, .10):
            delta = min(transfer, p[i] - floor)
            if delta <= 1e-14:
                continue
            shifted = p.copy()
            shifted[i] = max(p[i] - delta, floor)
            shifted[j] += p[i] - shifted[i]
            values.append((f"{i}->{j}:{transfer:g}", shifted))
    unique = {}
    for label, vector in values:
        unique.setdefault(tuple(np.round(vector, 13)), (label, vector))
    return list(unique.values())


def mc_draws(seed, t, dimensions, size, stream):
    rng = np.random.default_rng([seed, t, 3791, stream])
    u = rng.uniform(size=(size//2, len(dimensions)))
    v = rng.uniform(size=u.shape)
    n = rng.standard_normal(u.shape)
    return (np.concatenate([u, 1-u]), np.concatenate([n, -n]),
            chi2.ppf(np.concatenate([v, 1-v]), np.asarray(dimensions)-1))


def forecast_energy_recourses(energy, posterior, feedback, t, costs, predictor):
    features = energy.features(posterior)
    base = {}
    for horizon in HORIZONS:
        offset, public = public_features(feedback, t, horizon)
        stacked = np.concatenate([np.broadcast_to(public, (*features.shape[:-1], public.shape[-1])),
                                  features], axis=-1)
        base[horizon] = np.log(predictor.expected(horizon, offset, stacked)/offset)
    # Interpolate t+2 and damp the learned log change beyond the validated horizon.
    adjustments = [base[1] if h == 1 else ((base[1]+base[3])/2 if h == 2
                   else base[3]*.75**(h-3)) for h in range(1, len(costs)+1)]
    return costs * np.exp(np.stack(adjustments, axis=1))


def compare_candidates(controller, t, z, options, remaining, coefficients, feedback,
                       forecasts, *, weight=1.0, samples=16, confirmation_batches=6):
    protocol = controller.protocol
    future = np.arange(t+1, protocol.rounds+1)
    config = controller.cfg
    result = dict(round=t, z=z, status="no_future", candidates=[], solver_fallbacks=0)
    if not len(future):
        return 0, result
    ledger = GroupRDPFilter(protocol)
    future_floor = protocol.charges(future, protocol.z_floor).sum(axis=0)
    cost = protocol.charges(t, z)
    if z < protocol.z_floor or np.any(cost+future_floor > remaining):
        raise ValueError("Deterministic benchmark cannot fund its suffix floor")
    suffix = ConvexPrecision(protocol, future, remaining - cost)
    tail = suffix.value(coefficients[1:])
    result["tail_loss"] = tail
    candidates = []
    for label, p in options:
        action = PrecisionAction(t, z, tuple(zip(protocol.dimensions, map(float, p), strict=True)),
                                 feedback.basis_hash, feedback.history_hash)
        cert = ledger.certify(action)
        feasible = bool(np.all(np.array(cert.group_rdp_increment) + future_floor <= remaining))
        if not np.allclose(cert.group_rdp_increment, cost, atol=1e-13, rtol=1e-12):
            raise ValueError("Fixed-z candidate changed the certified RDP")
        candidates.append(dict(candidate=label, p=p.tolist(), feasible=feasible,
                               group_rdp_increment=list(cert.group_rdp_increment),
                               current_loss=float(np.sum(coefficients[0]/p)/z),
                               voi_proposal=None, proposal_gain=None, voi_confirm=None,
                               confirmation_gain=None, confirmation_se=None, confirmation_lower=None))
    result["candidates"] = candidates
    loss = np.array([r["current_loss"] for r in candidates])
    delta = loss - loss[0]
    draws = [mc_draws(config["public_seed"], t, list(protocol.dimensions.values()), samples, stream)
             for stream in range(confirmation_batches+1)]

    def value(i, stream):
        distribution = forecasts(np.asarray(candidates[i]["p"]), draws[stream])
        # Batchwise Jensen gap removes MC drift in the mean from the action comparison.
        # Values preserve the same remaining RDP and the same loss units for all p.
        if not np.isfinite(distribution).all() or np.any(distribution <= 0):
            raise FloatingPointError("Invalid virtual curvature")
        if np.all(distribution == distribution[0]):
            return 0.0
        return suffix.value(distribution.mean(axis=0)) - np.mean([suffix.value(k) for k in distribution])

    try:
        proposal = np.array([value(i, 0) if row["feasible"] else 0.0 for i, row in enumerate(candidates)])
        allowed_delta = np.where([r["feasible"] for r in candidates], delta, np.inf)

        def confirm(winner):
            baseline, candidate = [np.array([value(i, s) for s in range(1, len(draws))]) for i in (0, winner)]
            candidates[0]["voi_confirm"] = float(baseline.mean())
            candidates[winner]["voi_confirm"] = float(candidate.mean())
            return weight*baseline, weight*candidate

        selected, winner, gains, mean, error, lower = confirm_energy_gain(
            weight*proposal, confirm, allowed_delta, config["ranking_gap_factor"], config["local_min_decrease"])
        for i, row in enumerate(candidates):
            row.update(voi_proposal=float(proposal[i]), proposal_gain=float(gains[i]) if row["feasible"] else None,
                       current_loss_delta=float(delta[i]), j0=loss[i]+tail)
        candidates[winner].update(confirmation_gain=mean, confirmation_se=error, confirmation_lower=lower)
        result.update(proposal_index=winner, status="changed" if selected else "unconfirmed")
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError, ValueError):
        selected = 0
        result.update(status="numerical_fallback", solver_fallbacks=1)
    if suffix.fallbacks:
        selected = 0
        result["status"] = "solver_fallback"
        result["solver_fallbacks"] += suffix.fallbacks
    result["selected_index"] = selected
    return selected, result


def energy_options(controller, t, p, coefficients, energy):
    prior = energy.predict(t)
    _, uncertainty = energy.moments(prior)
    # Sensitivity of future variance loss to a change in share at the baseline.
    sensitivity = coefficients[1:].mean(axis=0)/p**2
    priority = uncertainty/uncertainty.max() * sensitivity/sensitivity.max()
    return layer_candidates(p, controller.static_layer_weights(), controller.protocol.p_min, priority)


def choose_energy(controller, feedback, t, z, p, remaining, coefficients, predictor,
                  baseline_fallbacks=0, *, energy=None, weight=1.0):
    if baseline_fallbacks or t == controller.protocol.rounds:
        return 0, dict(round=t, z=z, status="no_future" if t == controller.protocol.rounds else "baseline_fallback",
                       candidates=[], solver_fallbacks=0)
    energy = feedback.energy if energy is None else energy
    prior = energy.predict(t)
    options = energy_options(controller, t, p, coefficients, energy)
    protocol = controller.protocol
    certifier = GroupRDPFilter(protocol)

    def forecast(p, draws):
        action = PrecisionAction(t, z, tuple(zip(protocol.dimensions, map(float, p), strict=True)),
                                 feedback.basis_hash, feedback.history_hash)
        _, _, nu2 = observation_geometry(certifier.certify(action).source_plan)
        obs = virtual_energies(energy, prior, nu2, *draws)
        posterior = energy.update(prior, obs, nu2)
        return forecast_energy_recourses(energy, posterior, feedback, t, coefficients[1:], predictor)

    return compare_candidates(controller, t, z, options, remaining, coefficients, feedback, forecast,
                              weight=weight, samples=controller.cfg["virtual_observations"])
