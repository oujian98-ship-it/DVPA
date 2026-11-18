"""Continuous precision planning. Final releases still use the original RDP filter."""

import math
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, xlog1py, xlogy

from .dvpa_controller import allocation_risk, spatial_optimum
from .dvpa_filter import PrecisionAction


class ConvexPrecision:
    """Minimize weighted Gaussian variance subject to all three group balances."""

    def __init__(self, protocol, rounds, remaining):
        self.protocol = protocol
        self.rounds = np.asarray(rounds)
        self.scale = protocol.z0
        self.floor = protocol.z_floor / self.scale
        self.remaining = np.asarray(remaining)
        self.floor_plan = np.full(len(rounds), self.floor)
        floor_cost = self.exact(self.floor_plan)
        if np.any(floor_cost > self.remaining):
            raise ValueError("Insufficient balance for floor precision on suffix")
        reserve = protocol.config["privacy"]["numerical_reserve"]
        self.limit = np.maximum(self.remaining * (1 - reserve), floor_cost)
        self.units = np.maximum(self.limit, 1e-12)
        rates = protocol.rates[self.rounds - 1].T
        orders = np.asarray(protocol.orders)
        k = np.arange(orders.max() + 1)
        log_comb = np.array([[math.log(math.comb(a, i)) if i <= a else -np.inf
                              for i in k] for a in orders])
        valid = k <= orders[:, None]
        q = rates[:, :, None]
        self.weights = log_comb[:, None, :] + xlogy(k, q) + xlog1py(
            np.maximum(orders[:, None, None] - k, 0), -q
        )
        self.weights = np.where(valid[:, None, :], self.weights, -np.inf)
        self.slopes = protocol.radii[:, None, None] ** 2 * self.scale * k * (k - 1) / 2
        self.orders = (orders - 1)[:, None]
        self.nontrivial = rates > 0
        self.margin = np.where((rates > 0) & (rates < 1), 1 + 1e-12, 1.0)
        self.offset = np.where((rates > 0) & (rates < 1), 1e-14, 0.0)
        self.last_x = None
        self.last_cost_jac = None
        self.fallbacks = 0
        self.uniform = self._uniform()

    def exact(self, z):
        return self.protocol.charges(self.rounds, z * self.scale).sum(axis=0)

    def costs(self, z):
        if self.last_x is None or not np.array_equal(self.last_x, z):
            matrix = self.weights + self.slopes * z[None, :, None]
            log_norm = logsumexp(matrix, axis=-1)
            rdps = np.where(self.nontrivial, log_norm / self.orders * self.margin + self.offset, 0)
            derivative = (
                np.exp(matrix - log_norm[..., None]) * self.slopes
            ).sum(axis=-1) / self.orders * self.margin
            self.last_x = z.copy()
            self.last_cost_jac = rdps.sum(axis=1), derivative
        return self.last_cost_jac

    def _uniform(self):
        low, high = self.floor, max(1.0, self.floor * 2)
        while np.all(self.costs(np.full(len(self.rounds), high))[0] <= self.limit):
            high *= 2
            if high > 1e8:
                raise ValueError("Convex planner requires at least one finite binding budget")
        for _ in range(40):
            mid = (low + high) / 2
            if np.all(self.costs(np.full(len(self.rounds), mid))[0] <= self.limit):
                low = mid
            else:
                high = mid
        return self._feasible(np.full(len(self.rounds), low))

    def _feasible(self, candidate):
        if np.all(self.exact(candidate) <= self.remaining):
            return candidate
        low, high = 0.0, 1.0
        for _ in range(36):
            mid = (low + high) / 2
            point = self.floor_plan + mid * (candidate - self.floor_plan)
            if np.all(self.exact(point) <= self.remaining):
                low = mid
            else:
                high = mid
        return self.floor_plan + low * (candidate - self.floor_plan)

    def solve(self, risks):
        risks = np.asarray(risks, dtype=float)
        if risks.shape != self.rounds.shape or (risks <= 0).any() or not np.isfinite(risks).all():
            raise ValueError("Require positive finite risk for each planned round")
        if len(risks) == 1 or np.array_equal(self.uniform, self.floor_plan):
            return self.uniform * self.scale
        weights = risks / risks.sum()
        result = minimize(
            lambda z: np.dot(weights, 1 / z), self.uniform, jac=lambda z: -weights / z**2,
            method="SLSQP", bounds=[(self.floor, None)] * len(risks),
            constraints=[dict(type="ineq",
                              fun=lambda z: (self.limit - self.costs(z)[0]) / self.units,
                              jac=lambda z: -self.costs(z)[1] / self.units[:, None])],
            options=dict(ftol=1e-10, maxiter=80),
        )
        if not result.success or not np.isfinite(result.x).all():
            self.fallbacks += 1
            return self.uniform * self.scale
        candidate = self._feasible(np.maximum(result.x, self.floor))
        if np.dot(weights, 1 / candidate) > np.dot(weights, 1 / self.uniform) + 1e-10:
            self.fallbacks += 1
            return self.uniform * self.scale
        return candidate * self.scale

    def value(self, coefficients):
        risks = allocation_risk(coefficients, self.protocol.p_min)
        return float(np.dot(risks, 1/self.solve(risks)))


def _risks(controller, k):
    if controller.cfg["layer_policy"] is not None:
        return np.sum(k/controller.layer_weights(k), axis=-1)
    if controller.cfg["mode"] == "time_only":
        return np.sum(k / controller.static_layer_weights(), axis=-1)
    return allocation_risk(k, controller.protocol.p_min)


def _layer_options(controller, feedback, current, t):
    if controller.cfg["layer_policy"] is not None:
        return [controller.layer_weights(current)]
    if controller.cfg["mode"] == "time_only":
        return [controller.static_layer_weights()]
    p = spatial_optimum(current, controller.protocol.p_min)
    free = p - controller.protocol.p_min
    variance = np.diag(feedback.pred_cov)[feedback.indices(t)]
    options = [p]
    for i in np.argsort(-variance, kind="stable")[:2]:
        other = free.copy()
        other[i] = 0.0
        total = other.sum()
        if total > 1e-6:
            delta = min(0.03, total / 2)
            shifted = p - delta * other / total
            shifted[i] += delta
            options.append(shifted)
    return options


def choose_convex(controller, t, feedback, remaining, started):
    protocol = controller.protocol
    mode = controller.cfg["mode"]
    rounds = np.arange(t, protocol.rounds + 1)
    coefficients = feedback.expected_costs(
        feedback.pred_mean, feedback.pred_cov, rounds, origin=t,
        quadrature_nodes=controller.cfg["ranking_quadrature_nodes"],
    )[0]
    planner = ConvexPrecision(protocol, rounds, remaining)
    risks = _risks(controller, coefficients)
    schedule = planner.solve(risks)
    layers = _layer_options(controller, feedback, coefficients[0], t)
    z, p = float(schedule[0]), layers[0]
    current = float(np.dot(coefficients[0], 1 / p) / z)
    tail = float(np.dot(risks[1:], 1 / schedule[1:]))
    chosen = (z, p, current, tail, 0.0)
    weight = controller.cfg["information_value_weight"]
    attempts = 1
    fallbacks = planner.fallbacks
    changed = False
    margin = 0.0
    if controller.energy_predictor is not None:
        from .dvpa_energy_voi import choose_energy

        mc_started = perf_counter()
        selected, record = choose_energy(controller, feedback, t, z, p, remaining, coefficients,
                                         controller.energy_predictor, planner.fallbacks)
        controller.last_energy_mc_seconds = perf_counter() - mc_started if record["candidates"] else 0.0
        controller.last_energy_ranking = record
        if record["candidates"]:
            result = record["candidates"][selected]
            chosen = (z, np.asarray(result["p"]), result["current_loss"], record["tail_loss"],
                      result["voi_confirm"] if result["voi_confirm"] is not None else (result["voi_proposal"] or 0.0))
            changed = selected != 0
            margin = result["confirmation_lower"] or 0.0
        attempts = max(1, len(record["candidates"]))
        fallbacks += record["solver_fallbacks"]
    if mode == "decision_value" and weight > 0 and len(rounds) > 1:
        reserve = protocol.charges(rounds[1:], protocol.z_floor).sum(axis=0)
        # Independent public streams for proposal and confirmation. Within each
        # stream all actions receive the same antithetic standard-normal draws.
        draws = []
        for stream in (1, 2, 3, 4):
            rng = np.random.default_rng([controller.cfg["public_seed"], t, stream, 3841])
            n = rng.standard_normal((8, len(layers[0])))
            draws.append(np.concatenate([n, -n]))
        candidates = []
        for alt_z in (z, z * 0.97, z * 1.03):
            left = remaining - protocol.charges(t, alt_z)
            if alt_z < protocol.z_floor or np.any(left < reserve):
                continue
            suffix = ConvexPrecision(protocol, rounds[1:], left)
            prior = float(np.dot(risks[1:], 1 / suffix.solve(risks[1:])))
            for alt_p in layers:
                now = float(np.dot(coefficients[0], 1 / alt_p) / alt_z)
                estimates = []
                for noise in draws:
                    means, cov, _ = feedback.posterior(t, alt_z, alt_p, standard_normals=noise)
                    costs = feedback.expected_costs(
                        means, cov, rounds[1:], origin=t,
                        quadrature_nodes=controller.cfg["ranking_quadrature_nodes"],
                    )
                    posterior_risks = _risks(controller, costs)
                    values = [float(np.dot(k, 1 / suffix.solve(k))) for k in posterior_risks]
                    mean_risks = _risks(controller, costs.mean(axis=0))
                    empirical = float(np.dot(mean_risks, 1 / suffix.solve(mean_risks)))
                    estimates.append(max(0.0, empirical - float(np.mean(values))))
                candidates.append((alt_z, alt_p, now, prior, estimates))
            fallbacks += suffix.fallbacks
        attempts = len(candidates)
        if candidates and not fallbacks:
            proposal = min(candidates, key=lambda x: x[2] + x[3] - weight * x[4][0])
            benchmark = candidates[0]
            proposal_diff = np.array(benchmark[4][1:]) - np.array(proposal[4][1:])
            gains = benchmark[2] + benchmark[3] - proposal[2] - proposal[3] - weight * proposal_diff
            margin = float(gains.mean() - controller.cfg["ranking_gap_factor"] *
                           gains.std(ddof=1) / np.sqrt(len(gains)))
            if proposal is not benchmark and margin > controller.cfg["local_min_decrease"]:
                z1, p1, now, prior, estimates = proposal
                chosen = (z1, p1, now, prior, float(np.mean(estimates[1:])))
                changed = True
            else:
                chosen = (*chosen[:4], float(np.mean(benchmark[4][1:])))

    z, p, current, tail, voi = chosen
    reserve = protocol.charges(rounds[1:], protocol.z_floor).sum(axis=0)
    if np.any(protocol.charges(t, z) + reserve > remaining):
        raise RuntimeError("Convex action must preserve the suffix floor for all groups")
    action = PrecisionAction(t, z, tuple(zip(protocol.dimensions, map(float, p), strict=True)),
                             feedback.basis_hash, feedback.history_hash)
    return action, dict(
        round=t, mode=mode, warmup=False, current_loss=current, tail_loss=tail - voi,
        information_value=voi, information_value_weight=weight, weighted_information_value=weight*voi,
        unweighted_objective=current+tail-voi, raw_information_value=0.0, tower_correction=0.0,
        working_mutual_information=0.0, objective=current+tail-weight*voi,
        selected_tail_profile="convex", fallback_reason="solver_fallback" if fallbacks else "",
        candidates=attempts, controller_seconds=perf_counter()-started,
        convex_fallbacks=fallbacks, convex_voi_changed=changed, convex_voi_margin=margin,
        convex_reference_z=float(schedule[0]),
    )
