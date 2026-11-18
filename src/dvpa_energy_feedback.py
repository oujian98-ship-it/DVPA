"""Causal layer energy from a certified, published joint Gaussian update."""

import math

import numpy as np
from scipy.special import logsumexp, ndtr


ENERGY_STATE = "log_ar1_grid_v1"


def energy_variance(dimension, variance, energy):
    return 2 * dimension * variance**2 + 4 * variance * energy


def observation_geometry(plan):
    # The source plan already divides the raw sum by its expected count.
    normalizer = plan.final_scale * plan.denominator
    sigmas = np.array([sigma for _, sigma in plan.layer_noise_std])
    if normalizer <= 0 or not np.isfinite(normalizer) or not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
        raise ValueError("Invalid certified Gaussian observation geometry")
    return normalizer, sigmas**2, (sigmas / normalizer)**2


def energy_observations(released, certified, dimensions, seed):
    plan = certified.source_plan
    action = certified.action
    normalizer, sigma2, nu2 = observation_geometry(plan)
    if set(released) != set(dimensions):
        raise ValueError("Released layer coverage mismatch")
    rows = []
    for index, (name, size) in enumerate(dimensions.items()):
        vector = released[name].detach().cpu().double().flatten().numpy()
        if vector.size != size or not np.isfinite(vector).all():
            raise ValueError("Invalid published layer")
        raw_norm = float(np.einsum("i,i->", vector, vector))
        raw_energy = raw_norm - size * sigma2[index]
        energy = raw_energy / normalizer**2
        sd = math.sqrt(energy_variance(size, nu2[index], max(0.0, energy)))
        rows.append(dict(
            round=action.round_index, t=action.round_index, seed=seed, layer=name,
            d_l=size, q_schedule=",".join(map(str, plan.group_rates)), D_t=plan.denominator,
            s_t=plan.final_scale, z_t=action.precision, p_tl=dict(action.layer_precision)[name],
            sigma2=float(sigma2[index]), nu2=float(nu2[index]), norm2_Y=raw_norm,
            e_hat_raw=float(raw_energy), norm2_U=raw_norm / normalizer**2,
            e_hat=float(energy), e_hat_per_param=float(energy / size),
            negative=bool(energy < 0), noise_only_sd=math.sqrt(2 * size) * nu2[index],
            plug_in_cv=sd / energy if energy > 0 else "",
        ))
    return rows


class LayerEnergyState:
    """Independent per-layer log-AR(1) beliefs on a bounded quadrature grid.

    The Gaussian observation approximation uses both exact conditional moments
    of debiased energy. Coverage is checked offline against noncentral chi-square.
    """

    def __init__(self, protocol):
        self.dimensions = np.asarray(list(protocol.dimensions.values()))
        scale = float(max(protocol.radii))**2
        # Public prior and transition. The switch gets extra process variance.
        self.log_grid = np.log(scale) + np.linspace(-22.0, 2.0, 97)
        self.grid = np.exp(self.log_grid)
        center = np.log(scale) - 6.0
        weights = np.exp(-0.5 * ((self.log_grid - center) / 3.0)**2)
        self.posterior = np.tile(weights / weights.sum(), (len(self.dimensions), 1))
        centers = center + 0.90 * (self.log_grid - center)
        self.transitions = []
        for process_sd in (0.75, 1.25):
            weights = np.exp(-0.5 * ((self.log_grid - centers[:, None]) / process_sd)**2)
            self.transitions.append(weights / weights.sum(axis=-1, keepdims=True))
        self.switch_round = protocol.base["transition_round"]
        self.round = 0

    def predict(self, t):
        if t != self.round + 1:
            raise ValueError("Energy prediction requires precisely the previous DP prefix")
        return self.posterior @ self.transitions[int(t == self.switch_round)]

    def update(self, prior, observations, nu2):
        variance = energy_variance(self.dimensions[:, None], np.asarray(nu2)[:, None], self.grid)
        residual = np.asarray(observations)[..., None] - self.grid
        likelihood = -0.5 * (np.log(variance) + residual**2 / variance)
        with np.errstate(divide="ignore"):
            weights = np.log(prior) + likelihood
        normalizer = logsumexp(weights, axis=-1, keepdims=True)
        if not np.isfinite(normalizer).all():
            raise ValueError("Energy likelihood could not be normalized")
        return np.exp(weights - normalizer)

    def moments(self, posterior):
        mean = posterior @ self.grid
        sd = np.sqrt(np.sum(posterior * (self.grid - mean[..., None])**2, axis=-1))
        return mean, sd

    def features(self, posterior):
        # Bound feature scale using posterior log moments, never log(e_hat).
        mean = posterior @ self.log_grid
        sd = np.sqrt(np.sum(posterior * (self.log_grid - mean[..., None])**2, axis=-1))
        return np.stack([mean, sd], axis=-1)

    def observe(self, t, rows):
        prior = self.predict(t)
        values = np.array([row["e_hat"] for row in rows])
        nu2 = np.array([row["nu2"] for row in rows])
        variance = energy_variance(self.dimensions[:, None], nu2[:, None], self.grid)
        pit = np.sum(prior * ndtr((values[:, None] - self.grid) / np.sqrt(variance)), axis=-1)
        updated = self.update(prior, values, nu2)
        means, sds = self.moments(updated)
        for i, row in enumerate(rows):
            row.update(chosen_state_posterior=ENERGY_STATE, posterior_energy=float(means[i]),
                       posterior_sd=float(sds[i]), predictive_pit=float(pit[i]))
        self.posterior = updated
        self.round = t

    def state_dict(self):
        return dict(version=ENERGY_STATE, round=self.round, posterior=self.posterior.tolist())


def virtual_energies(state, prior, nu2, uniforms, normal, chi_square):
    """Exact draws of debiased squared norm, with action-independent random inputs."""
    cdf = np.cumsum(prior, axis=-1)
    indices = np.minimum(np.sum(uniforms[..., None] > cdf, axis=-1), len(state.grid) - 1)
    energies = state.grid[indices]
    return (np.sqrt(energies) + np.sqrt(nu2) * normal)**2 + nu2 * (chi_square - state.dimensions)
