"""Author-matched model/training units and public grouped BR-DP preflight."""

from dataclasses import asdict, dataclass
import copy
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from .brdp_direction import direction_hash
from .brdp_group_accountant import (
    epsilon_from_rdp,
    joint_round_rdp,
    round_group_costs,
    source_orders,
)
from .brdp_noise import parameters, sample_scalar_noise
from .original_scheduler import groups
from .st_mechanism import clip_blocks
from .upstream_runner import source_model, source_partition


def public_training_data_config(cfg):
    """Share the pinned data schema without changing the native DP policy.

    The original data-only validator accepts a JOINT tag. Normalize that tag
    only on a copy; actual training/preflight/accounting retain cfg['method'].
    """
    if cfg['method'] not in {'JOINT','IDP_FEDAVG','SPEND_AS_YOU_GO'}:
        raise ValueError('Training-public data route supports DVPA/IDP/SAYG only')
    return dict(cfg,method='JOINT')


def validate_config(config):
    cfg = copy.deepcopy(config)
    required = {
        "seed",
        "dataset",
        "data_root",
        "clients",
        "rounds",
        "local_epochs",
        "batch_size",
        "lr",
        "rate",
        "clip",
        "delta",
        "epsilons",
        "group_proportions",
        "q_save",
        "q_spend",
        "transition_round",
        "device",
        "method",
        "mechanism",
        "warmup_rounds",
        "recycle_prob",
        "tau",
        "layer_decay",
        "adjacency",
        "partition",
        "normalization",
    }
    allowed = required | {
        "limit_train",
        "limit_test",
        "dataset_mode",
        "synthetic_samples",
        "calibration_epsilons",
        "evaluation_public_file",
        "evaluation_public_sha256",
        "public_train_file",
        "public_train_sha256",
        "public_train_count",
    }
    if not isinstance(cfg, dict) or not required <= set(cfg) or set(cfg) - allowed:
        raise ValueError("Unknown or missing source-protocol configuration fields")
    for key in [
        "seed",
        "clients",
        "rounds",
        "local_epochs",
        "batch_size",
        "transition_round",
        "warmup_rounds",
    ]:
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid integer {key}")
    if (
        cfg["clients"] < 3
        or cfg["transition_round"] > cfg["rounds"]
        or cfg["warmup_rounds"] > cfg["rounds"]
    ):
        raise ValueError("Invalid client count or phase horizon")
    for key in ["lr", "rate", "clip", "delta", "q_spend", "tau"]:
        if (
            isinstance(cfg[key], bool)
            or not isinstance(cfg[key], (float, int))
            or not math.isfinite(cfg[key])
            or cfg[key] <= 0
        ):
            raise ValueError(f"Invalid positive {key}")
    if not cfg["delta"] < 1 or not cfg["q_spend"] <= 1:
        raise ValueError("Invalid delta or q_spend")
    for key in ["epsilons", "group_proportions", "q_save"]:
        if not isinstance(cfg[key], list) or len(cfg[key]) != 3:
            raise ValueError("Exactly three privacy groups are required")
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v <= 0
            for v in cfg[key]
        ):
            raise ValueError(f"Invalid {key}")
    if any(v > 1 for v in cfg["q_save"]) or not math.isclose(
        sum(cfg["group_proportions"]), 1.0
    ):
        raise ValueError("Invalid group rates/proportions")
    if (
        len(set(cfg["epsilons"])) != 3
        or (
            np.bincount(groups(cfg["clients"], cfg["group_proportions"]), minlength=3)
            == 0
        ).any()
    ):
        raise ValueError("Require three distinct nonempty budget groups")
    if cfg["adjacency"] != "add_remove_client_contribution":
        raise ValueError(
            "Sampled reference bound currently supports fixed-slot add/remove contributions only"
        )
    if (
        cfg["dataset"] not in {"fmnist", "mnist"}
        or cfg["partition"] != "source_dirichlet_alpha_0.1"
        or cfg["normalization"] != "minus_one_to_one"
    ):
        raise ValueError("Source protocol requires FMNIST or MNIST/author Dirichlet/[-1,1]")
    if cfg["method"] not in {
        "UNIFORM",
        "TIME_ONLY",
        "LAYER_ONLY",
        "JOINT",
        "SPEND_AS_YOU_GO",
        "IDP_FEDAVG",
    }:
        raise ValueError("Unsupported time/layer allocation method")
    if cfg["mechanism"] not in {"gaussian", "brdp"}:
        raise ValueError("Unsupported noise mechanism")
    if set(cfg["layer_decay"]) != {"conv1", "conv2", "fc1", "fc"} or any(
        not math.isfinite(v) or v < 0 for v in cfg["layer_decay"].values()
    ):
        raise ValueError("Layer decay must name the author model blocks")
    parameters(1.0, cfg["tau"], cfg["recycle_prob"])
    if cfg["method"] in {"SPEND_AS_YOU_GO", "IDP_FEDAVG"} and (
        cfg["mechanism"] != "gaussian" or "calibration_epsilons" in cfg
    ):
        raise ValueError(
            "Original spend-as-you-go requires Gaussian noise and original budgets"
        )
    if "calibration_epsilons" in cfg:
        values = cfg["calibration_epsilons"]
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or not 0 < v <= cap
                for v, cap in zip(values, cfg["epsilons"], strict=True)
            )
        ):
            raise ValueError(
                "Calibration targets must be positive and within the three caps"
            )
    if cfg.get("dataset_mode", "official") not in {"official", "synthetic"}:
        raise ValueError("Unsupported dataset mode")
    if cfg.get("dataset_mode", "official") == "official" and (
        "limit_train" in cfg or "limit_test" in cfg
    ):
        raise ValueError("Formal protocol uses full clean official train/test splits")
    if ("evaluation_public_file" in cfg) != ("evaluation_public_sha256" in cfg):
        raise ValueError("Evaluation package and digest must be supplied together")
    if "public_train_count" in cfg and "public_train_file" not in cfg:
        raise ValueError("Public count requires a pinned training-public package")
    if "public_train_file" in cfg or "public_train_sha256" in cfg:
        from .dvpa_public_training import validate_public_train_config
        validate_public_train_config(public_training_data_config(cfg))
    return cfg


def load_datasets(cfg):
    from .data import load_data

    if cfg.get("dataset_mode") == "synthetic":
        rng = torch.Generator().manual_seed(cfg["seed"])
        count = max(40 * cfg["clients"], cfg.get("synthetic_samples", 600))
        train = TensorDataset(
            torch.rand(count, 1, 28, 28, generator=rng) * 2 - 1,
            torch.arange(count) % 10,
        )
        test = TensorDataset(
            torch.rand(40, 1, 28, 28, generator=rng) * 2 - 1, torch.arange(40) % 10
        )
    else:
        train, test = load_data(dict(dataset=cfg["dataset"], root=cfg["data_root"]))
        train = TensorDataset(train.tensors[0] * 2 - 1, train.tensors[1])
        test = TensorDataset(test.tensors[0] * 2 - 1, test.tensors[1])
        if len(train) != 60000 or len(test) != 10000:
            raise ValueError("Expected clean official 60000/10000 split")
        if "public_train_file" in cfg:
            from .dvpa_public_training import remove_public_training_examples
            train = remove_public_training_examples(public_training_data_config(cfg), train)
    return train, test


def partition_and_groups(cfg, labels):
    # Privacy starts AFTER the public benchmark partition is frozen. Never
    # recompute this label-based assignment under a client-neighbor comparison.
    np.random.seed(cfg["seed"])
    parts = source_partition(labels, cfg["clients"])
    ids = groups(cfg["clients"], cfg["group_proportions"])
    np.random.shuffle(ids)
    if sorted(i for part in parts for i in part) != list(range(len(labels))):
        raise ValueError("Invalid author partition coverage")
    return parts, ids


class PersistentSourceClient:
    """Author SGD/local-epoch semantics, without raw diagnostic publications.

    All clients train each round, even if not selected for aggregation. Their
    optimizer, momentum and CosineAnnealingLR survive global-model broadcasts.
    """

    def __init__(self, dataset, indices, cfg, device, slot=0):
        self.cfg, self.device = cfg, device
        self.model = source_model()(cfg["dataset"], cfg["batch_size"]).to(device)
        self.data = Subset(dataset, indices)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=5e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200
        )
        # Public but per-slot isolated training RNG: another client's data size
        # must not change this client's shuffle stream through shared RNG state.
        self.data_rng = torch.Generator().manual_seed(cfg["seed"] + 1000003 * slot)
        self.epochs_completed = 0

    def train(self, global_model, registry, return_update=True):
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()
        before = registry.pack(global_model) if return_update else None
        loader = DataLoader(
            self.data,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=0,
            generator=self.data_rng,
        )
        for _ in range(self.cfg["local_epochs"]):
            for x, y in loader:
                self.optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(
                    self.model(x.to(self.device)), y.to(self.device)
                )
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            self.epochs_completed += 1
        if not return_update:
            return None
        after = registry.pack(self.model)
        # Scheme2 units VERBATIM: u=(local-global)/initial_lr*rate.
        # After clipping/noise the server multiplies by initial_lr*rate.
        return {
            g: (after[g] - before[g]) / self.cfg["lr"] * self.cfg["rate"]
            for g in before
        }


@dataclass(frozen=True)
class SourceRoundPlan:
    round_index: int
    group_rates: tuple
    denominator: float
    group_radii: tuple
    final_scale: float
    time_fraction: float | None
    layer_clip_weights: tuple
    layer_budget_weights: tuple
    layer_noise_std: tuple
    planned_shapes: tuple

    def as_dict(self):
        return asdict(self)


def _time_layer_weights(cfg, names):
    temporal = cfg["method"] in {"TIME_ONLY", "JOINT"}
    spatial = cfg["method"] in {"LAYER_ONLY", "JOINT"}
    raw = np.asarray(
        [
            1.0 if not temporal or t < cfg["transition_round"] else 2.0
            for t in range(1, cfg["rounds"] + 1)
        ]
    )
    fractions = raw / raw.sum()
    spatial_rows = []
    for t in range(1, cfg["rounds"] + 1):
        logs = np.asarray(
            [-cfg["layer_decay"][g] * (t - 1) if spatial else 0.0 for g in names]
        )
        values = np.exp(logs - logs.max())
        spatial_rows.append(
            dict(zip(names, (values / values.sum()).tolist(), strict=True))
        )
    return fractions, spatial_rows


def preflight(cfg, dimensions):
    if cfg["method"] in {"SPEND_AS_YOU_GO", "IDP_FEDAVG"}:
        from .sayg_protocol import source_preflight

        return source_preflight(cfg, dimensions)
    names = list(dimensions)
    if set(names) != set(cfg["layer_decay"]):
        raise ValueError("Author model registry mismatch")
    orders = source_orders()
    counts = np.bincount(groups(cfg["clients"], cfg["group_proportions"]), minlength=3)
    fractions, spatial_rows = _time_layer_weights(cfg, names)
    clip_weights = dict.fromkeys(names, 1 / len(names))
    shapes = [
        (
            cfg["recycle_prob"]
            if cfg["mechanism"] == "brdp" and t > cfg["warmup_rounds"]
            else 0.0,
            cfg["tau"],
        )
        for t in range(1, cfg["rounds"] + 1)
    ]

    def costs_for_beta(beta, group):
        cache, rows = {}, []
        for t, f in enumerate(fractions, 1):
            q = cfg["q_save"][group] if t < cfg["transition_round"] else cfg["q_spend"]
            r, tau = shapes[t - 1]
            key = (q, float(f), r, tau)
            if key not in cache:
                cache[key] = joint_round_rdp(
                    q, beta * beta * f, orders, [(r, tau)] * len(names)
                )
            rows.append(cache[key])
        return np.asarray(rows)

    betas = []
    for g, target in enumerate(cfg.get("calibration_epsilons", cfg["epsilons"])):
        desired = target * (1 - 1e-6)
        low, high = 1e-12, 1.0
        if (
            epsilon_from_rdp(
                orders,
                costs_for_beta(low, g).sum(axis=0),
                cfg["delta"],
                warn_boundary=False,
            )[0]
            >= desired
        ):
            raise ValueError("Soft-recycling envelope alone exhausts this group budget")
        for _ in range(60):
            if (
                epsilon_from_rdp(
                    orders,
                    costs_for_beta(high, g).sum(axis=0),
                    cfg["delta"],
                    warn_boundary=False,
                )[0]
                > desired
            ):
                break
            high *= 2
        else:
            raise ValueError("Could not bracket public noise calibration")
        for _ in range(55):
            middle = (low + high) / 2
            if (
                epsilon_from_rdp(
                    orders,
                    costs_for_beta(middle, g).sum(axis=0),
                    cfg["delta"],
                    warn_boundary=False,
                )[0]
                <= desired
            ):
                low = middle
            else:
                high = middle
        betas.append(low)
    # Same harmonic-mean clipping structure as the author, with NEW calibrated
    # multipliers. Weighted average raw clip radius remains exactly c=250.
    kappa = 1 / float(np.dot(counts / cfg["clients"], betas))
    radii = cfg["clip"] * kappa * np.asarray(betas)
    plans = []
    for t, f in enumerate(fractions, 1):
        rates = cfg["q_save"] if t < cfg["transition_round"] else [cfg["q_spend"]] * 3
        denominator = float(np.dot(counts, rates))
        final_scale = cfg["lr"] * cfg["rate"] / denominator
        std = {
            g: cfg["clip"]
            * kappa
            * math.sqrt(clip_weights[g] / (f * spatial_rows[t - 1][g]))
            * final_scale
            for g in names
        }
        r, tau = shapes[t - 1]
        plans.append(
            SourceRoundPlan(
                t,
                tuple(rates),
                denominator,
                tuple(radii.tolist()),
                final_scale,
                float(f),
                tuple(clip_weights.items()),
                tuple(spatial_rows[t - 1].items()),
                tuple(std.items()),
                tuple((g, r, tau) for g in names),
            )
        )
    envelope = np.asarray(
        [
            round_group_costs(
                orders,
                p.group_radii,
                p.final_scale,
                dict(p.layer_clip_weights),
                dict(p.layer_noise_std),
                p.group_rates,
                {g: (r, tau) for g, r, tau in p.planned_shapes},
            )
            for p in plans
        ]
    )
    final_eps = [
        epsilon_from_rdp(orders, envelope[:, g].sum(axis=0), cfg["delta"])[0]
        for g in range(3)
    ]
    if any(
        eps > target for eps, target in zip(final_eps, cfg["epsilons"], strict=True)
    ):
        raise RuntimeError("Preflight group budget exceeded")
    return dict(
        orders=orders,
        counts=counts,
        betas=np.asarray(betas),
        kappa=kappa,
        group_radii=radii,
        plans=plans,
        envelope=envelope,
        final_epsilon=final_eps,
    )


def effective_shapes(plan, directions):
    return {
        g: (r if directions[g] is not None else 0.0, tau)
        for g, r, tau in plan.planned_shapes
    }


def actual_round_costs(
    cfg, orders, group_radii, final_scale, clip_weights, stds, rates, shapes
):
    if cfg["method"] in {"SPEND_AS_YOU_GO", "IDP_FEDAVG"}:
        from .sayg_protocol import gaussian_round_costs

        return gaussian_round_costs(
            orders, group_radii, final_scale, clip_weights, stds, rates, shapes
        )
    return round_group_costs(
        orders, group_radii, final_scale, clip_weights, stds, rates, shapes
    )


def release_source_sum(
    private_sum, plan, directions, actual_shapes, scalar_rng, noise_rng
):
    stds = dict(plan.layer_noise_std)
    if (
        set(private_sum) != set(stds)
        or set(directions) != set(stds)
        or set(actual_shapes) != set(stds)
    ):
        raise ValueError("Source release coverage mismatch")
    output = {}
    for g, raw in private_sum.items():
        sigma = stds[g]
        r, tau = actual_shapes[g]
        expected_r, expected_tau = effective_shapes(plan, directions)[g]
        if r != expected_r or tau != expected_tau:
            raise ValueError("Uncertified current noise shape")
        noise = (
            torch.randn(
                raw.shape, generator=noise_rng, device=raw.device, dtype=torch.float64
            )
            * sigma
        )
        if r > 0:
            u = directions[g].to(device=raw.device, dtype=torch.float64)
            if u.shape != raw.shape or abs(float(u.norm()) - 1) > 1e-10:
                raise ValueError("Invalid past-DP unit direction")
            z = sample_scalar_noise(sigma, tau, r, scalar_rng)
            noise = noise - torch.dot(noise, u) * u + z * u
        output[g] = (raw.double() * plan.final_scale + noise).to(raw.dtype)
        if not torch.isfinite(output[g]).all():
            raise RuntimeError("Nonfinite output: terminate without publishing")
    return output


def clip_source_update(update, radius, plan):
    radii = {g: radius * math.sqrt(w) for g, w in plan.layer_clip_weights}
    return clip_blocks(update, dict.fromkeys(update, True), radii)


def layer_log_rows(plan, dimensions, directions, actual_shapes):
    return [
        dict(
            round=plan.round_index,
            layer=g,
            dimension=dimensions[g],
            clip_weight=dict(plan.layer_clip_weights)[g],
            budget_weight=dict(plan.layer_budget_weights)[g],
            noise_std=sigma,
            recycle_prob=actual_shapes[g][0],
            tau=actual_shapes[g][1],
            planned_recycle_prob={k: r for k, r, _ in plan.planned_shapes}[g],
            denominator=plan.denominator,
            final_scale=plan.final_scale,
            direction_source_round=plan.round_index - 1,
            direction_hash=direction_hash(directions[g]),
        )
        for g, sigma in plan.layer_noise_std
    ]
