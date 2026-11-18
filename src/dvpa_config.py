"""Public DVPA configuration/calibration. Preview never loads training data."""

from dataclasses import dataclass
from pathlib import Path
import copy
import math

import numpy as np
import yaml

from .brdp_group_accountant import sampled_gaussian_rdp_upper
from .brdp_source_protocol import preflight, validate_config
from .dvpa_filter import VERSION, sgm_cost

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/dvpa_hfull.yaml"
DIMENSIONS = dict(conv1=832, conv2=51264, fc1=524800, fc=5130)
DEFAULTS = dict(
    feedback=dict(
        feedback_mode="legacy_direction",
        energy_state="log_ar1_grid_v1",
        warmup_rounds=3,
        basis="fixed_after_warmup",
        state="two_phase_joint_gaussian",
        extra_covariance="dp_innovation",
        lambda0=1.0,
        lambda1=1.0,
        signal_scale=10.0,
        transition="public_mean_reversion",
        mean_reversion=0.95,
        prior_std=10.0,
        process_std=1.0,
        layer_correlation=0.1,
        phase_correlation=0.8,
        innovation_ema=0.1,
        covariance_shrinkage=0.1,
        covariance_floor=0.01,
        reference_clip=250.0,
        utility_proxy="dimension_signal",
        public_proxy_file=None,
        public_proxy_sha256=None,
        public_proxy_validation=None,
        public_proxy_probes=16,
        curvature_forecast="hold_current",
        curvature_mode="normalized",
        forecast_ema=0.25,
        forecast_damping=0.75,
        forecast_clip=0.1,
        forecast_horizon=3,
        forecast_validation=None,
    ),
    controller=dict(
        curvature_only=False,
        layer_policy=None,
        layer_mix=0.25,
        voi_explore_p_only=False,
        energy_gate_file=None,
        energy_gate_sha256=None,
        mode="decision_value",
        virtual_observations=64,
        quadrature_nodes=12,
        public_seed=20261028,
        time_multipliers=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
        p_min=0.02,
        simplex_units=6,
        tail="finite_full_remaining_rollout",
        tail_solver="templates",
        fixed_precision_schedule=None,
        layer_gamma=None,
        tail_profiles=[
            "uniform",
            "front_loaded",
            "back_loaded",
            "source_phase",
            "q_inverse",
        ],
        entropy_weight=1.0,
        pair_transfer=0.05,
        information_estimator="empirical_jensen_control_variate",
        information_value_weight=1.0,
        ranking_enabled=False,
        ranking_coarse_repeats=3,
        ranking_top_k=4,
        ranking_repeats=4,
        ranking_samples=256,
        ranking_max_samples=1024,
        ranking_quadrature_nodes=24,
        ranking_validation_nodes=48,
        ranking_gap_factor=3.0,
        ranking_tolerance=1e-9,
        local_search_enabled=False,
        local_search_steps=1,
        local_time_offsets=[0.01, 0.03, 0.05],
        local_layer_transfers=[0.001, 0.003, 0.01],
        local_min_decrease=1e-8,
    ),
    privacy=dict(
        order_selection="public_reference_search",
        integer_orders=[3, 64],
        clipping_source="gaussian_joint_public_preflight",
        floor_precision_ratio=0.2,
        guarantee="fixed_group_filter",
        numerical_reserve=1e-8,
    ),
    output=dict(save_released_updates=True, save_private_rng=False, resume=False),
)


def resolve(config):
    if not isinstance(config, dict) or set(config) - (
        {"schema_version", "base_config", "base", "method"} | set(DEFAULTS)
    ):
        raise ValueError("Unknown DVPA config fields")
    if config.get("schema_version") != 1 or config.get("method") != "DVPA":
        raise ValueError("DVPA schema_version=1 and method=DVPA required")
    cfg = dict(schema_version=1, method="DVPA")
    if "base" in config and "base_config" in config:
        raise ValueError("Supply exactly one base representation")
    if "base" in config:
        base = copy.deepcopy(config["base"])
    else:
        path = (ROOT / config.get("base_config", "")).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError("base_config must be a file inside BR_DP")
        base = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["base"] = validate_config(base)
    if (
        base["method"] != "JOINT"
        or base["mechanism"] != "gaussian"
        or "calibration_epsilons" in base
    ):
        raise ValueError("DVPA v1 requires unmodified Gaussian JOINT reference budgets")
    # Recycled noise is deliberately disabled at the release boundary too.
    cfg["base"]["recycle_prob"] = 0.0
    for section, defaults in DEFAULTS.items():
        supplied = config.get(section, {})
        if not isinstance(supplied, dict) or set(supplied) - set(defaults):
            raise ValueError(f"Unknown {section} fields")
        cfg[section] = copy.deepcopy(defaults) | copy.deepcopy(supplied)
    f, c, p, out = (cfg[k] for k in ("feedback", "controller", "privacy", "output"))
    fixed = [
        (f, "basis", "fixed_after_warmup"),
        (f, "state", "two_phase_joint_gaussian"),
        (f, "transition", "public_mean_reversion"),
        (c, "tail", "finite_full_remaining_rollout"),
        (c, "information_estimator", "empirical_jensen_control_variate"),
        (p, "order_selection", "public_reference_search"),
        (p, "clipping_source", "gaussian_joint_public_preflight"),
        (p, "guarantee", "fixed_group_filter"),
    ]
    if any(s[k] != v for s, k, v in fixed):
        raise ValueError("Unsupported DVPA model or privacy family")
    if out != dict(save_released_updates=True, save_private_rng=False, resume=False):
        raise ValueError(
            "Released history required; private RNG publication/resume unsupported"
        )
    if f["extra_covariance"] not in {"dp_innovation", "public_bound"}:
        raise ValueError("Unknown covariance model")
    from .dvpa_energy_feedback import ENERGY_STATE

    if f["feedback_mode"] not in {"legacy_direction", "energy"} or f["energy_state"] != ENERGY_STATE:
        raise ValueError("Unknown feedback or energy state")
    if type(c["curvature_only"]) is not bool or type(c["voi_explore_p_only"]) is not bool:
        raise ValueError("Invalid DVPA v2 switches")
    policy = c["layer_policy"]
    if policy not in {None, "p_dim", "p_curv", "p_legacy", "p_mix"}:
        raise ValueError("Unknown layer allocation policy")
    if type(c["layer_mix"]) not in (float, int) or not 0 <= c["layer_mix"] <= 1:
        raise ValueError("Invalid layer mix")
    if policy is not None and (c["tail_solver"] != "convex" or c["mode"] != "frozen_info"):
        raise ValueError("Explicit allocation policies require frozen-info convex planning")
    if (policy == "p_legacy" and c["curvature_only"]) or (policy in {"p_curv", "p_mix"}
            and f["utility_proxy"] != "public_curvature_signal"):
        raise ValueError("Allocation policy and curvature source disagree")
    if f["feedback_mode"] == "energy" and (not c["curvature_only"]
            or c["tail_solver"] != "convex" or c["mode"] not in {"frozen_info", "time_only"}):
        raise ValueError("Energy feedback uses a direction-free convex baseline")
    if c["voi_explore_p_only"]:
        path, sha = c["energy_gate_file"], c["energy_gate_sha256"]
        if (f["feedback_mode"] != "energy" or c["mode"] != "frozen_info"
                or f["curvature_mode"] != "absolute_scale" or c["information_value_weight"] != 1
                or policy not in (None, "p_curv") or c["fixed_precision_schedule"] is not None
                or not isinstance(path, str) or not path or not (ROOT / path).resolve().is_relative_to(ROOT)
                or not isinstance(sha, str) or len(sha) != 64 or any(x not in "0123456789abcdef" for x in sha)):
            raise ValueError("Energy VOI requires a pinned A/B/C gate and the calibrated fixed-z comparison")
    elif c["energy_gate_file"] is not None or c["energy_gate_sha256"] is not None:
        raise ValueError("Unused energy gate")
    if f["utility_proxy"] not in {"dimension_signal", "public_curvature_signal"}:
        raise ValueError("Unknown utility proxy")
    if f["curvature_forecast"] not in {"hold_current", "damped_log_trend"}:
        raise ValueError("Unknown curvature forecast")
    for key in ("forecast_ema", "forecast_damping"):
        v = f[key]
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not 0 <= v <= 1
        ):
            raise ValueError("Invalid forecast smoothing")
    if (
        isinstance(f["forecast_clip"], bool)
        or not isinstance(f["forecast_clip"], (int, float))
        or not math.isfinite(f["forecast_clip"])
        or f["forecast_clip"] <= 0
        or (
            f["forecast_horizon"] != "full"
            and (type(f["forecast_horizon"]) is not int or not 1 <= f["forecast_horizon"] <= 25)
        )
    ):
        raise ValueError("Invalid forecast bound/horizon")
    test64 = "public_test_count" in cfg["base"]
    if f["curvature_forecast"] != "hold_current":
        path = f["forecast_validation"]
        if (
            f["utility_proxy"] != "public_curvature_signal"
            or (not test64 and (not isinstance(path, str) or not path
                or not (ROOT / path).resolve().is_relative_to(ROOT)))
        ):
            raise ValueError(
                "History forecast requires public curvature and an inside-BR_DP validation path"
            )
    if (
        type(f["public_proxy_probes"]) is not int
        or not 2 <= f["public_proxy_probes"] <= 128
    ):
        raise ValueError("Public curvature needs 2..128 probes")
    if f["utility_proxy"] == "public_curvature_signal":
        keys = ("public_proxy_file",) if test64 else ("public_proxy_file", "public_proxy_validation")
        for key in keys:
            value = f[key]
            if (
                not isinstance(value, str)
                or not value
                or not (ROOT / value).resolve().is_relative_to(ROOT)
            ):
                raise ValueError("Public proxy paths must stay inside BR_DP")
        sha = f["public_proxy_sha256"]
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError("Pin the public proxy artifact SHA256")
        if test64 and (f["public_proxy_file"] != cfg["base"]["evaluation_public_file"]
                       or sha != cfg["base"]["evaluation_public_sha256"]):
            raise ValueError("The curvature proxy must be the reserved official test64 package")
        if f["extra_covariance"] != "dp_innovation":
            raise ValueError("Public proxy study preserves second-version covariance")
    if c["mode"] not in {"decision_value", "frozen_info", "static", "entropy", "time_only", "layer_only"}:
        raise ValueError("Unknown controller mode")
    if f["curvature_mode"] not in {"normalized", "absolute_scale"}:
        raise ValueError("Unknown curvature scale mode")
    if f["curvature_mode"] == "absolute_scale" and f["utility_proxy"] != "public_curvature_signal":
        raise ValueError("Absolute scale requires the public curvature proxy")
    if c["tail_solver"] not in {"templates", "convex"}:
        raise ValueError("Unknown time planner")
    if c["tail_solver"] == "convex" and (
        c["mode"] not in {"time_only", "frozen_info", "decision_value"}
        or c["ranking_enabled"] or c["local_search_enabled"]
    ):
        raise ValueError("Convex planner uses its own candidate and validation procedure")
    if (
        isinstance(c["information_value_weight"], bool)
        or not isinstance(c["information_value_weight"], (int, float))
        or not math.isfinite(c["information_value_weight"])
        or c["information_value_weight"] < 0
    ):
        raise ValueError("Nonnegative finite information value weight required")
    for section, keys in [
        (f, ["warmup_rounds"]),
        (
            c,
            [
                "virtual_observations",
                "quadrature_nodes",
                "public_seed",
                "simplex_units",
                "ranking_coarse_repeats",
                "ranking_top_k",
                "ranking_repeats",
                "ranking_samples",
                "ranking_max_samples",
                "ranking_quadrature_nodes",
                "ranking_validation_nodes",
                "local_search_steps",
            ],
        ),
    ]:
        for key in keys:
            if type(section[key]) is not int or section[key] < (
                0 if key == "public_seed" else 1
            ):
                raise ValueError(f"Invalid integer {key}")
    if f["warmup_rounds"] > base["rounds"] or c["virtual_observations"] % 2:
        raise ValueError("Invalid warmup or antithetic observation count")
    schedule, gamma = c["fixed_precision_schedule"], c["layer_gamma"]
    if schedule is not None or gamma is not None:
        if (
            not isinstance(schedule, list) or len(schedule) != base["rounds"]
            or any(type(z) not in (float, int) or not math.isfinite(z) or z <= 0 for z in schedule)
            or (policy is None and (type(gamma) not in (float, int) or not math.isfinite(gamma) or not 0 <= gamma <= 1))
            or (policy is not None and gamma is not None)
            or c["mode"] != "frozen_info" or c["tail_solver"] != "convex"
            or c["information_value_weight"] != 0
        ):
            raise ValueError("A fixed schedule requires positive precisions and one frozen-info layer allocation rule")
    if (
        type(c["ranking_enabled"]) is not bool
        or c["ranking_top_k"] < 2
        or c["ranking_repeats"] < 2
        or c["ranking_samples"] % 2
        or c["ranking_max_samples"] % 2
        or c["ranking_max_samples"] < c["ranking_samples"]
        or c["ranking_validation_nodes"] <= c["ranking_quadrature_nodes"]
    ):
        raise ValueError("Invalid numerical ranking configuration")
    for key in ["ranking_gap_factor", "ranking_tolerance"]:
        if (
            isinstance(c[key], bool)
            or not isinstance(c[key], (int, float))
            or not math.isfinite(c[key])
            or c[key] <= 0
        ):
            raise ValueError("Invalid numerical ranking tolerance")
    if type(c["local_search_enabled"]) is not bool or (
        c["local_search_enabled"] and not c["ranking_enabled"]
    ):
        raise ValueError("Fine search requires numerical ranking checks")
    if c["local_search_steps"] != 1:
        raise ValueError("This version uses one local refinement pass per round")
    for key in ["local_time_offsets", "local_layer_transfers"]:
        if (
            not isinstance(c[key], list)
            or not c[key]
            or len(set(c[key])) != len(c[key])
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or not 0 < v < 1
                for v in c[key]
            )
        ):
            raise ValueError("Invalid local precision steps")
    if (
        isinstance(c["local_min_decrease"], bool)
        or not isinstance(c["local_min_decrease"], (int, float))
        or not math.isfinite(c["local_min_decrease"])
        or c["local_min_decrease"] <= 0
    ):
        raise ValueError("Positive finite local improvement tolerance required")
    for section, keys in [
        (
            f,
            [
                "lambda0",
                "signal_scale",
                "prior_std",
                "reference_clip",
                "covariance_floor",
            ],
        ),
        (c, ["p_min", "pair_transfer"]),
        (p, ["floor_precision_ratio", "numerical_reserve"]),
    ]:
        for key in keys:
            v = section[key]
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v <= 0
            ):
                raise ValueError(f"Invalid positive {key}")
    for key in ["lambda1", "process_std"]:
        if (
            isinstance(f[key], bool)
            or not isinstance(f[key], (int, float))
            or not math.isfinite(f[key])
            or f[key] < 0
        ):
            raise ValueError(f"Invalid nonnegative {key}")
    for key in [
        "mean_reversion",
        "layer_correlation",
        "phase_correlation",
        "innovation_ema",
        "covariance_shrinkage",
    ]:
        if (
            isinstance(f[key], bool)
            or not isinstance(f[key], (int, float))
            or not math.isfinite(f[key])
            or not 0 <= f[key] < 1
        ):
            raise ValueError(f"Invalid covariance/transition parameter {key}")
    if (
        not 0 < c["p_min"] < 0.25
        or not 0 < p["floor_precision_ratio"] <= 1
        or not p["numerical_reserve"] < 0.01
    ):
        raise ValueError("Invalid precision floor or numerical reserve")
    if (
        not isinstance(c["entropy_weight"], (int, float))
        or not math.isfinite(c["entropy_weight"])
        or c["entropy_weight"] < 0
    ):
        raise ValueError("Invalid public entropy weight")
    if (
        not isinstance(c["time_multipliers"], list)
        or not c["time_multipliers"]
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v <= 0
            for v in c["time_multipliers"]
        )
    ):
        raise ValueError("Invalid precision multipliers")
    profiles = {"uniform", "front_loaded", "back_loaded", "source_phase", "q_inverse"}
    if (
        not c["tail_profiles"]
        or len(set(c["tail_profiles"])) != len(c["tail_profiles"])
        or set(c["tail_profiles"]) - profiles
    ):
        raise ValueError("Invalid tail profiles")
    orders = p["integer_orders"]
    if (
        not isinstance(orders, list)
        or len(orders) != 2
        or any(type(a) is not int for a in orders)
        or not 3 <= orders[0] <= orders[1] <= 128
    ):
        raise ValueError("Invalid public integer-order interval")
    return cfg


def load_config(path):
    return resolve(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def smoke_config(config):
    cfg = copy.deepcopy(config)
    cfg["base"].update(
        clients=6,
        rounds=4,
        local_epochs=1,
        batch_size=16,
        transition_round=3,
        warmup_rounds=1,
        dataset_mode="synthetic",
        device="cpu",
        synthetic_samples=300,
    )
    cfg["feedback"]["warmup_rounds"] = 1
    # Smaller finite search only for explicitly labelled engineering smoke.
    cfg["controller"].update(
        virtual_observations=8,
        quadrature_nodes=8,
        simplex_units=2,
        time_multipliers=[0.5, 1.0, 2.0],
    )
    return resolve(cfg)


@dataclass
class PublicProtocol:
    config: dict
    dimensions: dict
    counts: np.ndarray
    radii: np.ndarray
    rates: np.ndarray
    denominators: np.ndarray
    scales: np.ndarray
    orders: np.ndarray
    caps: np.ndarray
    reference_z: np.ndarray
    z0: float = 0.0
    z_floor: float = 0.0

    @property
    def base(self):
        return self.config["base"]

    @property
    def rounds(self):
        return self.base["rounds"]

    @property
    def p_min(self):
        return self.config["controller"]["p_min"]

    def charges(self, rounds, precision):
        """Broadcast round indices and precision; trailing axis is the 3 groups."""
        indices, z = np.broadcast_arrays(
            np.asarray(rounds, dtype=int) - 1, np.asarray(precision, dtype=float)
        )
        if (
            (indices < 0).any()
            or (indices >= self.rounds).any()
            or not np.isfinite(z).all()
            or (z < 0).any()
        ):
            raise ValueError("Invalid public cost query")
        result = np.zeros(z.shape + (3,))
        for g in range(3):
            rates = self.rates[indices, g]
            for q in np.unique(rates):
                mask = rates == q
                result[..., g][mask] = sgm_cost(
                    int(self.orders[g]), float(q), self.radii[g] ** 2 * z[mask]
                )
        return result

    def uniform_feasible(self, rounds, remaining):
        rounds = np.asarray(rounds, dtype=int)
        if not len(rounds):
            return 0.0
        remaining = np.asarray(remaining)
        low, high = 0.0, max(self.reference_z) * 2

        def feasible(z):
            return np.all(
                self.charges(rounds, z).sum(axis=0)
                <= remaining * (1 - self.config["privacy"]["numerical_reserve"])
            )

        while feasible(high):
            high *= 2
        for _ in range(55):
            mid = (low + high) / 2
            if feasible(mid):
                low = mid
            else:
                high = mid
        return low

    def as_dict(self):
        return dict(
            version=VERSION,
            dimensions=self.dimensions,
            counts=self.counts.tolist(),
            group_radii=self.radii.tolist(),
            rates=self.rates.tolist(),
            denominators=self.denominators.tolist(),
            final_scales=self.scales.tolist(),
            fixed_orders=self.orders.tolist(),
            rdp_caps=self.caps.tolist(),
            targets=self.base["epsilons"],
            delta=self.base["delta"],
            reference_precision=self.reference_z.tolist(),
            uniform_precision=self.z0,
            floor_precision=self.z_floor,
            guarantee="fixed caps valid for all histories; realized charges are diagnostics",
        )


def public_preflight(config, dimensions=None):
    cfg = resolve(config)
    dimensions = dict(DIMENSIONS if dimensions is None else dimensions)
    if dimensions != DIMENSIONS:
        raise ValueError("Author four-block model signature changed")
    ref = preflight(cfg["base"], dimensions)
    plans = ref["plans"]
    z = np.array(
        [
            sum(
                p.final_scale**2 * w / dict(p.layer_noise_std)[n] ** 2
                for n, w in p.layer_clip_weights
            )
            for p in plans
        ]
    )
    orders = []
    caps = []
    start, end = cfg["privacy"]["integer_orders"]
    candidates = np.arange(start, end + 1)
    rates = np.array([p.group_rates for p in plans])
    for g in range(3):
        B = np.asarray(cfg["base"]["epsilons"][g]) - math.log(
            1 / cfg["base"]["delta"]
        ) / (candidates - 1)
        low = np.zeros(len(candidates))
        high = np.full(len(candidates), 16.0)

        def costs(scale):
            total = np.zeros(len(candidates))
            for q in np.unique(rates[:, g]):
                rows = np.flatnonzero(rates[:, g] == q)
                # Public reference has two time weights; group equal values.
                for zz in np.unique(z[rows]):
                    num = np.count_nonzero(z[rows] == zz)
                    # Distinct candidate orders/scales: small integer grid.
                    total += num * np.array(
                        [
                            sgm_cost(
                                int(a), float(q), ref["group_radii"][g] ** 2 * zz * s
                            )
                            for a, s in zip(candidates, scale)
                        ]
                    )
            return total

        if np.any((costs(high) <= B) & (B > 0)):
            raise ValueError("Order search public bracket too small")
        for _ in range(46):
            mid = (low + high) / 2
            ok = (costs(mid) <= B) & (B > 0)
            low = np.where(ok, mid, low)
            high = np.where(ok, high, mid)
        best = int(np.argmax(low))
        if low[best] <= 0:
            raise ValueError("No feasible fixed-order anchor")
        orders.append(int(candidates[best]))
        caps.append(float(B[best]))
    protocol = PublicProtocol(
        cfg,
        dimensions,
        ref["counts"],
        ref["group_radii"],
        rates,
        np.array([p.denominator for p in plans]),
        np.array([p.final_scale for p in plans]),
        np.array(orders),
        np.array(caps),
        z,
    )
    protocol.z0 = protocol.uniform_feasible(
        np.arange(1, protocol.rounds + 1), protocol.caps
    )
    protocol.z_floor = cfg["privacy"]["floor_precision_ratio"] * protocol.z0
    fixed = cfg["controller"]["fixed_precision_schedule"]
    if fixed is not None:
        warmup = cfg["feedback"]["warmup_rounds"]
        charges = protocol.charges(np.arange(1, protocol.rounds + 1), fixed)
        if (
            any(z != protocol.z0 for z in fixed[:warmup])
            or np.any(np.asarray(fixed) < protocol.z_floor)
            or np.any(charges.sum(axis=0) > protocol.caps)
        ):
            raise ValueError("Fixed precision schedule must match warmup, respect the floor, and fit all three RDP caps")
    # Independent cross-check of optimized vectorized moments against the shared primitive.
    for g in range(3):
        actual = float(protocol.charges(1, protocol.z0)[g])
        expected = sampled_gaussian_rdp_upper(
            float(rates[0, g]), float(protocol.radii[g] ** 2 * protocol.z0), [orders[g]]
        )[0]
        if not np.isclose(actual, expected, rtol=1e-12, atol=1e-13):
            raise RuntimeError("SGM primitive mismatch")
    return protocol
