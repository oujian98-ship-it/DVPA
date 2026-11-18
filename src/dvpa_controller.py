"""Finite full-tail, one-observation lookahead with action-dependent noise."""

import itertools
import math
import time

import numpy as np

from .dvpa_filter import PrecisionAction


def spatial_optimum(costs, floor):
    """Vectorized water filling, including active lower bounds."""
    k = np.asarray(costs, dtype=float)
    if not np.isfinite(k).all() or (k <= 0).any() or not 0 < floor < 1 / k.shape[-1]:
        raise ValueError("Positive finite layer costs and feasible floor required")
    roots = np.sqrt(k)
    fixed = np.zeros_like(k, dtype=bool)
    p = np.zeros_like(k)
    for _ in range(k.shape[-1]):
        available = 1 - floor * fixed.sum(axis=-1, keepdims=True)
        free = np.where(fixed, 0.0, roots)
        p = np.where(
            fixed,
            floor,
            available * free / np.maximum(free.sum(axis=-1, keepdims=True), 1e-300),
        )
        newly = (p < floor) & ~fixed
        if not newly.any():
            break
        fixed |= newly
    return p


def allocation_risk(costs, floor):
    k = np.asarray(costs)
    return np.sum(k / spatial_optimum(k, floor), axis=-1)


def _compositions(total, parts):
    if parts == 1:
        yield (total,)
    else:
        for v in range(total + 1):
            for tail in _compositions(total - v, parts - 1):
                yield (v,) + tail


class DecisionValueController:
    def __init__(self, protocol, energy_predictor=None):
        self.protocol = protocol
        self.cfg = protocol.config["controller"]
        if self.cfg["voi_explore_p_only"] != (energy_predictor is not None):
            raise ValueError("Online energy VOI requires the audited offline predictor")
        self.energy_predictor = energy_predictor
        self.last_energy_ranking = None
        self.last_energy_mc_seconds = 0.0
        units = self.cfg["simplex_units"]
        floor = protocol.p_min
        self.grid = [
            floor + (1 - 4 * floor) * np.array(x) / units
            for x in _compositions(units, 4)
        ]

    def static_layer_weights(self):
        return spatial_optimum(list(self.protocol.dimensions.values()), self.protocol.p_min)

    def layer_weights(self, costs):
        policy = self.cfg["layer_policy"]
        if policy == "p_dim" or self.cfg["mode"] == "time_only":
            return np.broadcast_to(self.static_layer_weights(), np.shape(costs))
        optimum = spatial_optimum(costs, self.protocol.p_min)
        if policy == "p_mix":
            mix = self.cfg["layer_mix"]
            return (1-mix)*self.static_layer_weights() + mix*optimum
        return optimum

    def spatial_candidates(self, current_costs):
        if self.cfg["mode"] == "time_only":
            return [self.static_layer_weights()]
        optimum = spatial_optimum(current_costs, self.protocol.p_min)
        vectors = [optimum, np.full(4, 0.25), *self.grid]
        for i, j in itertools.permutations(range(4), 2):
            delta = min(self.cfg["pair_transfer"], optimum[i] - self.protocol.p_min)
            if delta > 1e-14:
                p = optimum.copy()
                p[i] -= delta
                p[j] += delta
                vectors.append(p)
        result = []
        seen = set()
        for p in vectors:
            key = tuple(np.round(p, 14))
            if key not in seen:
                seen.add(key)
                result.append(p)
        return result

    def tail_plans(self, t, remaining):
        """Each row is a separately feasible entire remaining time schedule."""
        future = np.arange(t + 1, self.protocol.rounds + 1)
        if not len(future):
            return future, np.empty((1, 0))
        if self.cfg["mode"] == "layer_only":
            fixed = np.full((1, len(future)), self.protocol.z0)
            if np.any(self.protocol.charges(future, fixed[0]).sum(axis=0) > remaining):
                raise RuntimeError("Fixed STATIC schedule exceeds remaining RDP")
            return future, fixed
        floor = self.protocol.z_floor
        baseline = self.protocol.charges(future, floor).sum(axis=0)
        if np.any(baseline > remaining):
            raise RuntimeError("Candidate fails minimum future reserve")
        shapes = []
        n = len(future)
        for name in self.cfg["tail_profiles"]:
            if name == "uniform":
                v = np.ones(n)
            elif name == "front_loaded":
                v = np.linspace(2.0, 0.5, n)
            elif name == "back_loaded":
                v = np.linspace(0.5, 2.0, n)
            elif name == "source_phase":
                v = np.where(future < self.protocol.base["transition_round"], 1.0, 2.0)
            else:
                v = (
                    1
                    / (
                        self.protocol.denominators[future - 1]
                        / self.protocol.base["clients"]
                    )
                    ** 2
                )
            shapes.append(v / np.mean(v))
        shapes = np.array(shapes)
        low = np.zeros(len(shapes))
        high = np.full(len(shapes), max(self.protocol.reference_z) * 4)
        limit = remaining * (1 - self.protocol.config["privacy"]["numerical_reserve"])
        # Baseline feasibility is preserved even arbitrarily close to a floor.
        limit = np.maximum(limit, baseline)
        for _ in range(60):
            ok = np.all(
                self.protocol.charges(
                    future[None, :], floor + high[:, None] * shapes
                ).sum(axis=1)
                <= limit,
                axis=1,
            )
            if not ok.any():
                break
            high = np.where(ok, high * 2, high)
        else:
            raise RuntimeError("Tail calibration could not bracket costs")
        for _ in range(46):
            mid = (low + high) / 2
            ok = np.all(
                self.protocol.charges(
                    future[None, :], floor + mid[:, None] * shapes
                ).sum(axis=1)
                <= limit,
                axis=1,
            )
            low = np.where(ok, mid, low)
            high = np.where(ok, high, mid)
        return future, floor + low[:, None] * shapes

    def tail_value(self, feedback, means, covariance, future, plans, t):
        if not len(future):
            return np.zeros(len(np.atleast_2d(means))), np.zeros(
                len(np.atleast_2d(means)), dtype=int
            )
        k = feedback.expected_costs(means, covariance, future, origin=t)
        return self.tail_value_from_costs(k, plans)

    def tail_value_from_costs(self, k, plans):
        if self.cfg["mode"] == "time_only":
            risks = np.sum(k / self.static_layer_weights(), axis=-1)
        else:
            risks = allocation_risk(k, self.protocol.p_min)
        costs = risks @ (1 / plans).T
        winner = np.argmin(costs, axis=1)
        return costs[np.arange(len(costs)), winner], winner

    def information_estimate(self, posterior_costs, plans, frozen):
        """Concave Jensen gap with a sample-mean control variate.

        E[posterior K] equals prior K in the exact working model. Finite nested
        MC/GH need not preserve that identity. Evaluating W at the SAME empirical
        mean removes the spurious first-order tower discrepancy. This is a
        finite-sample biased numerical estimator, consistent as both MC and
        quadrature accuracy increase under the working model, not a new bound.
        Both the uncorrected estimate and correction are logged.
        """
        values, winners = self.tail_value_from_costs(posterior_costs, plans)
        empirical, _ = self.tail_value_from_costs(
            posterior_costs.mean(axis=0, keepdims=True), plans
        )
        voi = float(empirical[0] - values.mean())
        if voi < -1e-9 * max(1.0, abs(float(empirical[0]))):
            raise RuntimeError("Concave allocation-risk Jensen check failed")
        voi = max(0.0, voi)  # Only cancel roundoff after the analytic Jensen check.
        raw = float(frozen - values.mean())
        correction = float(frozen - empirical[0])
        return float(frozen - voi), voi, raw, correction, winners

    def choose(self, t, feedback, remaining):
        started = time.perf_counter()
        if feedback.prepared_round != t or feedback.last_observed != t - 1:
            raise ValueError("Only previous DP history may enter current action")
        if self.energy_predictor is not None:
            self.last_energy_ranking = dict(round=t, status="warmup", candidates=[])
            self.last_energy_mc_seconds = 0.0
        remaining = np.asarray(remaining, dtype=float)
        current = feedback.expected_costs(
            feedback.pred_mean, feedback.pred_cov, [t], origin=t
        )[0, 0]
        warm = t <= self.protocol.config["feedback"]["warmup_rounds"]
        mode = self.cfg["mode"]

        def action(z, p):
            return PrecisionAction(
                t,
                float(z),
                tuple(zip(self.protocol.dimensions, map(float, p), strict=True)),
                feedback.basis_hash,
                feedback.history_hash,
            )

        fixed = self.cfg["fixed_precision_schedule"]
        if warm or mode == "static" or fixed is not None:
            z = self.protocol.z0 if fixed is None else fixed[t - 1]
            p = self.static_layer_weights()
            if fixed is not None and not warm:
                current = feedback.expected_costs(
                    feedback.pred_mean, feedback.pred_cov, [t], origin=t,
                    quadrature_nodes=self.cfg["ranking_quadrature_nodes"],
                )[0, 0]
                if self.cfg["layer_policy"] is None:
                    gamma = self.cfg["layer_gamma"]
                    dimensions = np.asarray(list(self.protocol.dimensions.values()), dtype=float)
                    p = spatial_optimum(dimensions**(1-gamma) * current**gamma, self.protocol.p_min)
                else:
                    p = self.layer_weights(current)
            chosen = action(z, p)
            log = dict(
                round=t,
                mode=mode,
                warmup=warm,
                candidates=1,
                current_loss=float(np.sum(current / p) / z),
                tail_loss=0.0,
                information_value=0.0,
                information_value_weight=float(self.cfg["information_value_weight"]),
                weighted_information_value=0.0,
                unweighted_objective=float(np.sum(current / p) / z),
                raw_information_value=0.0,
                tower_correction=0.0,
                working_mutual_information=0.0,
                objective=float(np.sum(current / p) / z),
                selected_tail_profile="public_uniform" if fixed is None else "fixed_schedule",
                fallback_reason="",
                controller_seconds=time.perf_counter() - started,
            )
            if self.cfg["tail_solver"] == "convex":
                log.update(convex_fallbacks=0, convex_voi_changed=False,
                           convex_voi_margin=0.0, convex_reference_z=z)
            if self.cfg["ranking_enabled"]:
                from .dvpa_ranking import blank_log

                log.update(blank_log("warmup" if warm else "static"))
                self.last_ranking = dict(
                    round=t,
                    status=log["ranking_status"],
                    selected_action=chosen.as_dict(),
                )
            if self.cfg["local_search_enabled"]:
                from .dvpa_local_search import blank_local

                log.update(blank_local("warmup" if warm else "static"))
                self.last_local_search = dict(
                    round=t,
                    status=log["local_status"],
                    selected_action=chosen.as_dict(),
                )
            return chosen, log
        if self.cfg["tail_solver"] == "convex":
            from .dvpa_convex import choose_convex

            return choose_convex(self, t, feedback, remaining, started)
        if self.cfg["ranking_enabled"]:
            from .dvpa_ranking import choose_stable

            chosen, log = choose_stable(self, t, feedback, remaining, started)
            if self.cfg["local_search_enabled"]:
                from .dvpa_local_search import refine

                return refine(self, t, feedback, remaining, chosen, log, started)
            return chosen, log
        actions, tails = self.candidate_context(t, feedback, remaining)
        scores = self.score_actions(t, feedback, actions, tails)
        winner = min(range(len(actions)), key=lambda i: scores[i]["objective"])
        log = dict(scores[winner])
        log.pop("frozen_objective")
        log.update(
            candidates=len(actions), controller_seconds=time.perf_counter() - started
        )
        return actions[winner], log

    def candidate_context(self, t, feedback, remaining):
        """Keep the v1 physical candidate set fixed across all numerical repeats."""
        current = feedback.expected_costs(
            feedback.pred_mean, feedback.pred_cov, [t], origin=t
        )[0, 0]
        rest = np.arange(t + 1, self.protocol.rounds + 1)
        fixed_time = self.cfg["mode"] == "layer_only"
        reserve = (
            self.protocol.charges(rest, self.protocol.z0 if fixed_time else self.protocol.z_floor).sum(axis=0)
            if len(rest)
            else np.zeros(3)
        )
        if fixed_time:
            z_candidates = [self.protocol.z0]
        else:
            zbar = self.protocol.uniform_feasible(
                np.arange(t, self.protocol.rounds + 1), remaining
            )
            z_candidates = sorted(
                {self.protocol.z_floor, *(float(zbar * v) for v in self.cfg["time_multipliers"])}
            )
        actions = []
        tails = {}
        for z in z_candidates:
            if z < self.protocol.z_floor * (1 - 1e-12):
                continue
            cost = self.protocol.charges(t, z)
            if np.any(cost + reserve > remaining):
                continue
            tails[z] = self.tail_plans(t, remaining - cost)
            for p in self.spatial_candidates(current):
                actions.append(
                    PrecisionAction(
                        t,
                        float(z),
                        tuple(
                            zip(self.protocol.dimensions, map(float, p), strict=True)
                        ),
                        feedback.basis_hash,
                        feedback.history_hash,
                    )
                )
        if not actions:
            raise RuntimeError("No feasible candidate; stop before training")
        return actions, tails

    def score_actions(
        self, t, feedback, actions, tails, samples=None, nodes=None, repeat=0, stream=0
    ):
        """Paired common-random-number evaluations; repeat streams are public."""
        mode = self.cfg["mode"]
        count = self.cfg["virtual_observations"] if samples is None else samples
        current = feedback.expected_costs(
            feedback.pred_mean, feedback.pred_cov, [t], origin=t, quadrature_nodes=nodes
        )[0, 0]
        # Preserve exactly the old stream when repeat=stream=0.
        seed = (
            [self.cfg["public_seed"], t]
            if repeat == stream == 0
            else [self.cfg["public_seed"], t, stream, repeat, 1907]
        )
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        normals = rng.standard_normal((count // 2, 4))
        normals = np.concatenate([normals, -normals])
        frozen_cache = {}
        frozen_cache_winner = {}
        logs = []
        for action in actions:
            z = action.precision
            p = np.array([v for _, v in action.layer_precision])
            future, plans = tails[z]
            if z not in frozen_cache:
                if len(future):
                    k = feedback.expected_costs(
                        feedback.pred_mean,
                        feedback.pred_cov,
                        future,
                        origin=t,
                        quadrature_nodes=nodes,
                    )
                    values, winners = self.tail_value_from_costs(k, plans)
                    frozen_cache[z] = float(values[0])
                    frozen_cache_winner[z] = int(winners[0])
                else:
                    frozen_cache[z] = 0.0
            frozen = frozen_cache[z]
            now = float(np.sum(current / p) / z)
            if len(future) and mode in {"time_only", "layer_only"}:
                after = frozen
                voi = mi = raw = correction = 0.0
                objective = now + frozen
                selected_tail = (
                    "public_uniform" if mode == "layer_only"
                    else self.cfg["tail_profiles"][frozen_cache_winner[z]]
                )
            elif len(future):
                means, cov, mi = feedback.posterior(t, z, p, standard_normals=normals)
                posterior_costs = feedback.expected_costs(
                    means, cov, future, origin=t, quadrature_nodes=nodes
                )
                after, voi, raw, correction, winners = self.information_estimate(
                    posterior_costs, plans, frozen
                )
                objective = now + (
                    frozen if mode in {"frozen_info", "entropy"} else after
                )
                # Weight 1 retains the original arithmetic exactly. Other
                # weights are an explicit information-preference ablation,
                # not a calibrated expected total distortion.
                if mode == "decision_value" and self.cfg["information_value_weight"] != 1:
                    objective -= (self.cfg["information_value_weight"] - 1) * voi
                if mode == "entropy":
                    objective -= self.cfg["entropy_weight"] * mi
                selected_tail = self.cfg["tail_profiles"][
                    int(np.bincount(winners, minlength=len(plans)).argmax())
                ]
            else:
                after = voi = mi = raw = correction = 0.0
                objective = now
                selected_tail = "none"
            if not all(math.isfinite(v) for v in (objective, voi, after, now, mi)):
                raise ValueError("Nonfinite utility cannot select a private query")
            logs.append(
                dict(
                    round=t,
                    mode=mode,
                    warmup=False,
                    current_loss=now,
                    tail_loss=after,
                    information_value=voi,
                    information_value_weight=float(self.cfg["information_value_weight"]),
                    weighted_information_value=float(self.cfg["information_value_weight"] * voi),
                    unweighted_objective=float(now + after),
                    raw_information_value=raw,
                    tower_correction=correction,
                    working_mutual_information=mi,
                    objective=objective,
                    selected_tail_profile=selected_tail,
                    fallback_reason="",
                    frozen_objective=now + frozen,
                ),
            )
        return logs
