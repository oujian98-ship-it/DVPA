"""Stage D. Train on the official training set and score the test64 holdout."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import t as student_t
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset
import yaml

from scripts.diagnose_energy import OUT, TRAJECTORIES, read_csv, read_json, write_csv, write_json
import matplotlib.pyplot as plt
from scripts.run_dvpa_comparison import normalized_privacy, tensor_digest
from src.brdp_source_audit import audit as baseline_audit
from src.brdp_source_protocol import (PersistentSourceClient, load_datasets, partition_and_groups,
                                      preflight as baseline_preflight, validate_config)
from src.brdp_source_runner import run as run_baseline
from src.data import load_mnist_split, split_checksum
from src.dvpa_audit import audit
from src.dvpa_config import DEFAULT_CONFIG, DIMENSIONS, ROOT, load_config, public_preflight, resolve
from src.dvpa_energy_voi import load_energy_gate
from src.dvpa_public_training import PublicCurvatureProxy
from src.dvpa_test_public64 import (
    EVALUATION_SPLIT, PublicCurvatureProxy as Test64Proxy, configure, evaluation_data,
)
from src.dvpa_runner import inside_output, run
from src.layer_registry import LayerRegistry
from src.metrics import evaluate
from src.upstream_runner import source_model


POLICIES = {"P0": "p_dim", "Pcurv": "p_curv", "Plegacy": "p_legacy", "Pmix": "p_mix"}
NATIVE = {"SAYG": "SPEND_AS_YOU_GO", "IDP-FedAvg": "IDP_FEDAVG"}
CONTROLS = {"STATIC": "static", "Time-only": "time_only", "FROZEN_INFO": "frozen_info", "DVPA": "decision_value"}
SEEDS = (42, 43, 44, 45, 46)
EPOCHS = (1, 3, 5, 30)
LEVELS = (.8, 1., 1.2)
MAIN = (*CONTROLS, *NATIVE, "Curvature-only+Convex")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arms(stage, energy_gate=None):
    return {"calibration": ("No-DP", "STATIC"), "allocation-fixed": tuple(POLICIES),
            "allocation-closed": tuple(POLICIES),
            "main": MAIN + (("Energy-VOI+Convex",) if energy_gate else ())}[stage]


def fixed_schedule(path):
    if read_json(path / "audit.json")["status"] != "passed":
        raise ValueError("The shared fixed schedule needs an independently audited reference")
    return [json.loads(line)["certified"]["action"]["precision"] for line in
            (path / "actions.jsonl").read_text(encoding="utf-8").splitlines()]


def configuration(reference, stage, arm, seed, epochs, level, schedule=None, energy_gate=None, raw_clip=None):
    cfg = copy.deepcopy(reference)
    base = cfg["base"]
    base.update(seed=seed, local_epochs=epochs,
                epsilons=[round(e*level, 10) for e in reference["base"]["epsilons"]])
    if raw_clip is not None:
        base["clip"] = raw_clip
    configure(cfg)
    if arm in NATIVE:
        return validate_config(dict(base, method=NATIVE[arm]))
    if arm == "No-DP":
        return validate_config(base)
    c = cfg["controller"]
    if stage.startswith("allocation-") or arm in {"Curvature-only+Convex", "Energy-VOI+Convex"}:
        policy = POLICIES.get(arm, "p_curv")
        cfg["feedback"]["curvature_mode"] = "absolute_scale"
        if policy != "p_legacy":
            cfg["feedback"]["feedback_mode"] = "energy"
        c.update(mode="frozen_info", tail_solver="convex", curvature_only=policy != "p_legacy",
                 layer_policy=policy, layer_mix=.25, fixed_precision_schedule=schedule,
                 information_value_weight=0, ranking_enabled=False, local_search_enabled=False)
        if arm == "Energy-VOI+Convex":
            c.update(voi_explore_p_only=True, information_value_weight=1,
                     energy_gate_file=energy_gate.resolve().relative_to(ROOT).as_posix(),
                     energy_gate_sha256=sha(energy_gate))
    else:
        c["mode"] = CONTROLS[arm]
        if arm != "DVPA":
            c["information_value_weight"] = 0
    return resolve(cfg)


def no_dp_validation(cfg):
    """No-DP baseline with the common sampling, optimizer and evaluation split."""
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if cfg["device"] == "auto" else cfg["device"]
    train, test = load_datasets(cfg)
    test, evaluation = evaluation_data(cfg, test)
    parts, groups = partition_and_groups(cfg, train.tensors[1])
    clients = [PersistentSourceClient(train, indices, cfg, device, slot=i) for i, indices in enumerate(parts)]
    model = source_model()(cfg["dataset"], cfg["batch_size"]).to(device)
    registry = LayerRegistry(model)
    identity = dict(partition_sha256=split_checksum(parts), groups=groups.tolist(),
                    initial_model_sha256=tensor_digest(model.state_dict()))
    rng = np.random.default_rng()
    for round_index in range(1, cfg["rounds"]+1):
        rates = np.asarray(cfg["q_save"] if round_index < cfg["transition_round"] else [cfg["q_spend"]]*3)
        selected = rng.random(len(clients)) < rates[groups]
        delta = {n: torch.zeros(d, device=device) for n, d in registry.dimensions.items()}
        for i, client in enumerate(clients):
            update = client.train(model, registry, return_update=bool(selected[i]))
            if update is not None:
                for layer, value in update.items():
                    delta[layer].add_(value)
        scale = cfg["lr"]*cfg["rate"]/rates[groups].sum()
        registry.apply(model, {layer: value*scale for layer, value in delta.items()})
    loss, acc = evaluate(model, test, device)
    split = "test" if evaluation is not None else "validation"
    return {f"{split}_accuracy": acc, f"{split}_cross_entropy": loss, "privacy": "none",
            "evaluation_examples": len(test), **identity}


@torch.no_grad()
def public_kl(directory, cfg):
    """Public predictive KL for matched antithetic perturbations of each pre-round model."""
    base = cfg["base"]
    model = source_model()(base["dataset"], base["batch_size"]).eval()
    model.load_state_dict(torch.load(directory / "initial_model.pt", weights_only=True, map_location="cpu"))
    registry = LayerRegistry(model)
    images = (Test64Proxy if "public_test_count" in base else PublicCurvatureProxy)(cfg).images
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    images = images.to(device)
    names = {param: layer for layer, params in registry.parameters.items() for param in params}
    rng = torch.Generator().manual_seed(2026111709)
    probes = [{name: torch.randn(p.shape, generator=rng).to(device) for name, p in model.named_parameters()}
              for _ in range(4)]
    records = []
    for t, line in enumerate((directory / "actions.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        plan = json.loads(line)["certified"]["source_plan"]
        std = dict(plan["layer_noise_std"])
        target = model(images).softmax(-1)
        current = []
        for probe in probes:
            for sign in (-1, 1):
                perturbed = {name: p + sign*std[names[name]]*probe[name] for name, p in model.named_parameters()}
                changed = torch.func.functional_call(model, perturbed, (images,))
                current.append(float(F.kl_div(changed.log_softmax(-1), target, reduction="batchmean")))
        records.append(dict(round=t, public_kl=float(np.mean(current)), public_examples=len(images), virtual_draws=len(current)))
        registry.apply(model, torch.load(directory / "released" / f"round_{t:03d}.pt", weights_only=True, map_location=device))
    return records


def job_record(stage, arm, cfg, directory):
    path = directory / "checked_v2.json"
    if path.exists():
        cached = read_json(path)
        if cfg != cached["config"] or cached["stage"] != stage or cached["arm"] != arm:
            raise ValueError(f"Existing run has a different configuration: {directory}")
        if "model_sha256" in cached and sha(directory / "model.pt") != cached["model_sha256"]:
            raise ValueError(f"Completed model changed: {directory}")
        return cached
    started = perf_counter()
    if arm == "No-DP":
        directory.mkdir(parents=True, exist_ok=False)
        record = no_dp_validation(cfg)
    else:
        native = arm in NATIVE
        base = cfg if native else cfg["base"]
        use_test64 = "public_test_count" in base
        split = "test" if use_test64 else "validation"
        expected_split = EVALUATION_SPLIT if use_test64 else "independent_validation"
        if directory.exists():
            state = read_json(directory / "run_state.json")
            saved = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
            if state["status"] != "completed" or saved != cfg:
                raise ValueError(f"Existing run is incomplete or has a different configuration: {directory}")
            summary = read_json(directory / "summary.json")
        else:
            summary = (run_baseline if native else run)(cfg, directory)
        review = (baseline_audit if native else audit)(directory)
        if review["status"] != "passed" or summary["evaluation_split"] != expected_split:
            raise ValueError("Run failed independent replay or used the wrong evaluation data")
        write_json(directory / "audit.json", review)
        rounds = read_csv(directory / "rounds.csv")
        prefix = "test_"
        points = [dict(round=int(row["round"]), accuracy=float(row[prefix+"accuracy"]),
                       cross_entropy=float(row[prefix+"loss"])) for row in rounds]
        anchor_cfg = resolve(dict(schema_version=1, method="DVPA", base=dict(base, method="JOINT")))
        orders = public_preflight(anchor_cfg).orders
        privacy = normalized_privacy(directory, arm if native else "DVPA", orders)
        if any(row["epsilon_guarantee"] > cap + 1e-8 for row, cap in zip(privacy, base["epsilons"], strict=True)):
            raise ValueError("Method did not meet a formal group cap")
        if stage.startswith("allocation-"):
            for point, kl in zip(points, public_kl(directory, cfg), strict=True):
                point.update(public_kl=kl["public_kl"])
        partition = read_json(directory / "partition.json")
        record = dict({f"{split}_accuracy": points[-1]["accuracy"], f"{split}_cross_entropy": points[-1]["cross_entropy"],
                       f"{split}_curve": points}, evaluation_examples=summary["evaluation_examples"],
                      privacy=privacy, audit=review["status"], model_sha256=sha(directory / "model.pt"),
                      partition_sha256=partition["split_checksum"], groups=partition["group_ids"],
                      initial_model_sha256=tensor_digest(torch.load(directory / "initial_model.pt", map_location="cpu", weights_only=True)))
    record.update(stage=stage, arm=arm, config=cfg, elapsed_seconds=perf_counter()-started)
    write_json(path, record)
    return record


def summarize(out, records, split="validation"):
    accuracy = f"{split}_accuracy"
    cross_entropy = f"{split}_cross_entropy"
    metrics = [dict(name=r["name"], arm=r["arm"], seed=r["seed"], level=r["level"],
                    local_epochs=r["epochs"], raw_clip=r["raw_clip"],
                    accuracy=r[accuracy], cross_entropy=r[cross_entropy],
                    evaluation_examples=r.get("evaluation_examples")) for r in records]
    write_csv(out / f"{split}_metrics.csv", metrics)
    if any("privacy" in r for r in records):
        curves, privacy = [], []
        for row, result in zip(metrics, records, strict=True):
            tags = {k: row[k] for k in ("name", "arm", "seed", "level", "local_epochs", "raw_clip")}
            curves.extend(dict(tags, **point) for point in result.get(f"{split}_curve", []))
            if result.get("privacy", "none") != "none":
                privacy.extend(dict(tags, **group) for group in result["privacy"])
        if curves:
            write_csv(out / f"{split}_curves.csv", curves)
            draw_curves(out, curves, split)
        if privacy:
            write_csv(out / "group_privacy.csv", privacy)
    if len({r["level"] for r in metrics}) > 1:
        draw_privacy_utility(out, metrics, split)
    reference = "P0" if any(r["arm"] == "P0" for r in records) else "STATIC"
    targets = {(r["seed"], r["level"], r["epochs"], r["raw_clip"]): r[accuracy]
               for r in records if r["arm"] == reference}
    pairs = []
    groups = {(r["arm"], r["level"], r["epochs"], r["raw_clip"]) for r in records if r["arm"] != reference}
    for arm, level, epochs, clip in sorted(groups):
        block = [r for r in records if (r["arm"], r["level"], r["epochs"], r["raw_clip"]) == (arm, level, epochs, clip)
                 and (r["seed"], level, epochs, clip) in targets]
        if not block:
            continue
        delta = np.array([r[accuracy]-targets[r["seed"], level, epochs, clip] for r in block])
        width = float(student_t.ppf(.975, len(delta)-1) * delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta) > 1 else None
        pairs.append(dict(arm=arm, reference=reference, level=level, local_epochs=epochs, raw_clip=clip,
                          seeds=[r["seed"] for r in block],
                          differences=delta.tolist(), mean_delta=float(delta.mean()),
                          ci_low=float(delta.mean()-width) if width is not None else None,
                          ci_high=float(delta.mean()+width) if width is not None else None))
    if pairs:
        write_csv(out / f"{split}_paired_ci.csv", pairs)


def draw_curves(out, curves, split="validation"):
    protocols = sorted({(r["level"], r["local_epochs"], r["raw_clip"]) for r in curves})
    fig, axes = plt.subplots(len(protocols), 1, figsize=(8, 3.6*len(protocols)), squeeze=False, layout="constrained")
    for protocol, ax in zip(protocols, axes.flat, strict=True):
        block = [r for r in curves if (r["level"], r["local_epochs"], r["raw_clip"]) == protocol]
        for arm in sorted({r["arm"] for r in block}):
            sub = [r for r in block if r["arm"] == arm]
            rounds = sorted({r["round"] for r in sub})
            means = [np.mean([r["accuracy"] for r in sub if r["round"] == t]) for t in rounds]
            ax.plot(rounds, means, label=arm)
        level, epochs, clip = protocol
        ax.set(xlabel="Round", ylabel=f"Mean {split} accuracy (%)",
               title=f"Privacy cap ×{level:g}, local epochs {epochs}, raw clip {clip:g}")
        ax.legend(ncols=2, fontsize=8)
    fig.savefig(out / f"{split}_curves.png", dpi=150)
    plt.close(fig)


def draw_privacy_utility(out, metrics, split):
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for arm in sorted({r["arm"] for r in metrics}):
        sub = [r for r in metrics if r["arm"] == arm]
        for epochs, raw_clip in sorted({(r["local_epochs"], r["raw_clip"]) for r in sub}):
            block = [r for r in sub if (r["local_epochs"], r["raw_clip"]) == (epochs, raw_clip)]
            levels = sorted({r["level"] for r in block})
            means = [np.mean([r["accuracy"] for r in block if r["level"] == level]) for level in levels]
            ax.plot(levels, means, marker="o", label=f"{arm} ({epochs} ep, clip {raw_clip:g})")
    ax.set(xlabel="Multiplier of the three formal privacy caps", ylabel=f"Mean {split} accuracy (%)")
    ax.legend(fontsize=7)
    fig.savefig(out / f"{split}_privacy_utility.png", dpi=150)
    plt.close(fig)


def specifications(args, reference):
    selected = tuple(dict.fromkeys(args.arm or arms(args.stage, args.energy_gate)))
    if not set(selected) <= set(arms(args.stage, args.energy_gate)):
        raise ValueError("Arm is incompatible with the specified stage")
    levels = args.level or (LEVELS if args.stage == "main" else (1.,))
    epochs = args.epochs
    clips = args.raw_clip
    if epochs is None and args.epochs_from:
        selection = read_json(args.epochs_from)
        if selection.get("tuning_split") != "independent_validation" or selection.get("selected_local_epochs") not in EPOCHS:
            raise ValueError("Expected a completed independent-validation epoch selection")
        epochs = [selection["selected_local_epochs"]]
        if clips is None:
            clips = [selection["selected_raw_clip"]]
    epochs = epochs or (EPOCHS if args.stage == "calibration" else (30,))
    default_clip = reference["base"]["clip"]
    clips = clips or [default_clip]
    if not np.isfinite(clips).all() or min(clips) <= 0:
        raise ValueError("Raw clip values must be positive and finite")
    schedule, anchor_clip = None, default_clip
    if args.stage == "allocation-fixed":
        schedule = fixed_schedule(args.fixed_reference)
        anchor_clip = load_config(args.fixed_reference / "config.yaml")["base"]["clip"]
    scaled_schedules = {}
    for level in dict.fromkeys(levels):
        for epoch in dict.fromkeys(epochs):
            for seed in dict.fromkeys(args.seed):
                for arm in selected:
                    for clip in dict.fromkeys([default_clip] if arm == "No-DP" else clips):
                        scaled = schedule
                        if schedule is not None and clip != anchor_clip:
                            key = (clip, level)
                            if key not in scaled_schedules:
                                scaled = [value*(anchor_clip/clip)**2 for value in schedule]
                                prototype = configuration(reference, args.stage, arm, seed, epoch, level,
                                                          energy_gate=args.energy_gate, raw_clip=clip)
                                preflight = public_preflight(prototype)
                                warmup = prototype["feedback"]["warmup_rounds"]
                                scaled[:warmup] = [float(preflight.z0)]*warmup
                                scaled_schedules[key] = scaled
                            scaled = scaled_schedules[key]
                        cfg = configuration(reference, args.stage, arm, seed, epoch, level, scaled, args.energy_gate, clip)
                        base = cfg.get("base", cfg)
                        if arm == "Energy-VOI+Convex":
                            load_energy_gate(cfg)
                        suffix = f"_clip{clip:g}" if clip != default_clip else ""
                        yield dict(name=f"{arm}_seed{seed}_epoch{epoch}_cap{level:g}{suffix}", arm=arm, seed=seed,
                                   epochs=epoch, level=level, raw_clip=clip, privacy_caps=base["epsilons"], config=cfg)


def check_preflight(planned):
    checked = set()
    for job in planned:
        key = (job["arm"], job["epochs"], job["level"], job["raw_clip"])
        if key in checked or job["arm"] == "No-DP":
            continue
        if job["arm"] in NATIVE:
            baseline_preflight(job["config"], DIMENSIONS)
        else:
            public_preflight(job["config"])
        checked.add(key)
    return len(checked)


def verify_pairing(rows):
    slots = {}
    for r in rows:
        identity = tuple(r[k] for k in ("partition_sha256", "groups", "initial_model_sha256"))
        other = slots.setdefault(r["seed"], identity)
        if other != identity:
            raise ValueError("Client partition, grouping or initialization differs within seed")


def lock(args, planned, records):
    if args.stage == "calibration" or len(set(args.seed)) < 5 or args.arm or len({(r["epochs"], r["raw_clip"]) for r in planned}) != 1:
        raise ValueError("Final testing requires at least five seeds, all arms and one locked epoch/clip setting")
    if args.stage == "main" and len(set(args.level or LEVELS)) < 2:
        raise ValueError("The main design requires multiple privacy levels")
    if len(records) != len(planned) or any(r.get("audit") != "passed" for r in records):
        raise ValueError("Full evaluation and audit must complete before locking")
    lock_path = args.out / "final_design.lock.json"
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump(dict(stage=args.stage, jobs=[dict(r, model_sha256=c["model_sha256"],
                  config_sha256=sha(args.out / "runs" / r["name"] / "config.yaml"))
                  for r, c in zip(planned, records, strict=True)]), handle, indent=2, allow_nan=False)


def final_test(out):
    design = read_json(out / "final_design.lock.json")
    first = design["jobs"][0]["config"]
    base = first.get("base", first)
    if (out / "test_metrics.csv").exists():
        raise ValueError("This design already has final evaluation results")
    records = []
    for job in design["jobs"]:
        job_base = job["config"].get("base", job["config"])
        keys = ("dataset", "data_root", "public_test_count", "evaluation_public_sha256")
        if any(job_base.get(key) != base.get(key) for key in keys):
            raise ValueError("Locked jobs must have the same dataset and official test split")
        path = out / "runs" / job["name"]
        if sha(path / "model.pt") != job["model_sha256"] or sha(path / "config.yaml") != job["config_sha256"]:
            raise ValueError("A locked model/config was changed")
    official = load_mnist_split(base["data_root"], base["dataset"], "test")
    if len(official) != 10000:
        raise ValueError("Unexpected official test size")
    test = TensorDataset(official.tensors[0]*2-1, official.tensors[1])
    test, _ = evaluation_data(base, test)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for job in design["jobs"]:
        path = out / "runs" / job["name"]
        model = source_model()(base["dataset"], base["batch_size"]).to(device)
        model.load_state_dict(torch.load(path / "model.pt", map_location=device, weights_only=True))
        loss, acc = evaluate(model, test, device)
        records.append(dict(job, test_accuracy=acc, test_cross_entropy=loss, evaluation_examples=len(test)))
    summarize(out, records, "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "lock", "test"))
    parser.add_argument("--stage", choices=("calibration", "allocation-fixed", "allocation-closed", "main"), default="main")
    parser.add_argument("--reference", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", choices=("fmnist", "mnist"), help="Override the reference dataset")
    parser.add_argument("--fixed-reference", type=Path, default=TRAJECTORIES[-1].parent / "convex_time_only_seed42")
    parser.add_argument("--energy-gate", type=Path)
    parser.add_argument("--seed", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--level", nargs="+", type=float, choices=LEVELS)
    epoch = parser.add_mutually_exclusive_group()
    epoch.add_argument("--epochs", nargs="+", type=int, choices=EPOCHS)
    epoch.add_argument("--epochs-from", type=Path, help="Read the validation-only epoch selection")
    parser.add_argument("--raw-clip", nargs="+", type=float, help="Raw clipping calibration candidates")
    parser.add_argument("--arm", nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    reference = None if args.command == "test" else load_config(args.reference)
    if reference is not None and args.dataset:
        reference["base"]["dataset"] = args.dataset
    dataset = args.dataset or (reference["base"]["dataset"] if reference else "fmnist")
    args.out = inside_output(args.out or (OUT / "official-test64" / dataset / args.stage))
    if args.command == "test":
        final_test(args.out)
        return
    planned = list(specifications(args, reference))
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "plan":
        count = check_preflight(planned)
        write_json(args.out / "test_plan.json", planned)
        print(f"Planned {len(planned)} {args.stage} runs on the official test64 protocol; checked {count} distinct privacy preflights")
        return
    records = []
    for spec in planned:
        directory = args.out / "runs" / spec["name"]
        if args.command == "lock" and not (directory / "checked_v2.json").is_file():
            raise ValueError(f"Missing completed run: {spec['name']}")
        result = job_record(args.stage, spec["arm"], spec["config"], directory)
        records.append(dict(result, **{k: v for k, v in spec.items() if k != "config"}))
    verify_pairing(records)
    if args.command == "lock":
        lock(args, planned, records)
    else:
        summarize(args.out, records, "test")


if __name__ == "__main__":
    main()
