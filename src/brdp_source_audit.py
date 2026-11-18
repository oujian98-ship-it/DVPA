"""Independent three-group replay of source-matched BR-DP artifacts."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from .brdp_direction import DirectionState
from .brdp_group_accountant import (
    ACCOUNTANT_VERSION,
    GroupPrivacyLedger,
)
from .brdp_source_protocol import (
    actual_round_costs,
    effective_shapes,
    layer_log_rows,
    preflight,
    public_training_data_config,
    validate_config,
)
from .data import split_checksum
from .dvpa_public_training import client_training_count
from .layer_registry import LayerRegistry
from .upstream_runner import source_model


def _rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _close(actual, expected, label):
    if isinstance(expected, str) or isinstance(expected, bool):
        if actual != str(expected):
            raise ValueError(f"Audit mismatch: {label}")
    elif not math.isfinite(float(actual)) or not math.isclose(
        float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-11
    ):
        raise ValueError(f"Audit mismatch: {label}")


def _compare_row(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("Audit row fields mismatch")
    for k, v in expected.items():
        _close(actual[k], v, k)


def audit(directory):
    root = Path(directory)
    cfg = validate_config(
        yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    )
    implementation = json.loads(
        (root / "implementation.json").read_text(encoding="utf-8")
    )
    if implementation["accountant"] != ACCOUNTANT_VERSION:
        raise ValueError("Wrong accountant family")
    if (
        hashlib.sha256((root / "config.yaml").read_bytes()).hexdigest()
        != implementation["config_sha256"]
    ):
        raise ValueError("Config hash mismatch")
    for field, folder in [
        ("source_sha256", "source_snapshot"),
        ("author_source_sha256", "upstream_source"),
    ]:
        for relative, digest in implementation[field].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe snapshot path")
            if (
                hashlib.sha256((root / folder / path).read_bytes()).hexdigest()
                != digest
            ):
                raise ValueError("Source snapshot hash mismatch")
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    # Match initialization order WITHOUT reading any raw client training state.
    for _ in range(cfg["clients"]):
        source_model()(cfg["dataset"], cfg["batch_size"])
    model = source_model()(cfg["dataset"], cfg["batch_size"])
    initial = torch.load(
        root / "initial_model.pt", map_location="cpu", weights_only=True
    )
    if set(initial) != set(model.state_dict()) or any(
        not torch.equal(initial[k], v) for k, v in model.state_dict().items()
    ):
        raise ValueError("Author initialization does not match seed/client order")
    registry = LayerRegistry(model)
    if sum(registry.dimensions.values()) != 582026:
        raise ValueError("Author model signature mismatch")
    public = preflight(cfg, registry.dimensions)
    saved_preflight = json.loads(
        (root / "privacy_preflight.json").read_text(encoding="utf-8")
    )
    if saved_preflight.get("schedule_metadata") != public.get("schedule_metadata"):
        raise ValueError("Public schedule provenance/recurrence mismatch")
    for plan in public["plans"]:
        if not math.isclose(
            float(np.dot(public["counts"] / cfg["clients"], plan.group_radii)),
            cfg["clip"],
            rel_tol=1e-12,
        ):
            raise ValueError("Per-round average raw clip mismatch")
    if saved_preflight["plans"] != [
        json.loads(json.dumps(p.as_dict())) for p in public["plans"]
    ]:
        raise ValueError("Public noise/clip schedule mismatch")
    for name, expected in [
        ("beta", public["betas"]),
        ("group_clip_raw", public["group_radii"]),
        ("orders", public["orders"]),
    ]:
        if not np.allclose(saved_preflight[name], expected, rtol=1e-12, atol=1e-12):
            raise ValueError("Public preflight parameter mismatch")
    if not math.isclose(
        float(np.dot(public["counts"] / cfg["clients"], public["group_radii"])),
        cfg["clip"],
        rel_tol=1e-12,
    ):
        raise ValueError("Average raw clip is not c=250")
    partition = json.loads((root / "partition.json").read_text(encoding="utf-8"))
    checksum = split_checksum(partition["parts"])
    if (
        checksum != partition["split_checksum"]
        or len(partition["parts"]) != cfg["clients"]
    ):
        raise ValueError("Partition checksum/count mismatch")
    if len(partition["group_ids"]) != cfg["clients"] or not np.array_equal(
        np.bincount(partition["group_ids"], minlength=3), public["counts"]
    ):
        raise ValueError("Privacy group assignment/count mismatch")
    all_ids = [i for p in partition["parts"] for i in p]
    expected_count = (
        max(40 * cfg["clients"], cfg.get("synthetic_samples", 600))
        if cfg.get("dataset_mode") == "synthetic"
        else client_training_count(cfg)
    )
    if sorted(all_ids) != list(range(expected_count)):
        raise ValueError("Partition coverage mismatch")
    if "public_train_file" in cfg:
        from .dvpa_public_training import partition_record, evaluation_record
        data_cfg = public_training_data_config(cfg)
        if json.loads((root / "data_partition_protocol.json").read_text(encoding="utf-8")) != partition_record(data_cfg):
            raise ValueError("Training-public partition provenance mismatch")
        if json.loads((root / "evaluation_protocol.json").read_text(encoding="utf-8")) != evaluation_record(data_cfg, cfg["public_train_sha256"]):
            raise ValueError("Full official test evaluation protocol mismatch")
    with np.load(root / "privacy_envelope.npz") as archive:
        if not np.array_equal(archive["orders"], public["orders"]) or not np.allclose(
            archive["envelope"], public["envelope"], rtol=1e-12, atol=1e-12
        ):
            raise ValueError("Uniform envelope mismatch")
    ledger = GroupPrivacyLedger(
        public["orders"], cfg["epsilons"], cfg["delta"], public["envelope"]
    )
    direction = DirectionState(registry.dimensions)
    layers, groups, rounds = (
        _rows(root / f) for f in ["layers.csv", "group_privacy.csv", "rounds.csv"]
    )
    layer_index = {(int(r["round"]), r["layer"]): r for r in layers}
    group_index = {(int(r["round"]), int(r["group"])): r for r in groups}
    if len(layers) != len(layer_index) or len(layer_index) != cfg["rounds"] * len(
        registry.names
    ):
        raise ValueError("Incomplete/duplicate layer records")
    if (
        len(groups) != len(group_index)
        or len(group_index) != cfg["rounds"] * 3
        or len(rounds) != cfg["rounds"]
    ):
        raise ValueError("Incomplete/duplicate round/group records")
    increments, last_group_rows = [], []
    non_gaussian = 0
    for t, plan in enumerate(public["plans"], 1):
        directions = direction.for_round(t)
        shapes = effective_shapes(plan, directions)
        for row in layer_log_rows(plan, registry.dimensions, directions, shapes):
            _compare_row(layer_index[t, row["layer"]], row)
            non_gaussian += row["recycle_prob"] > 0
        # Actual logged stds, radii, q and BR-DP shape are rechecked above, then
        # charged once as a joint round with each group's sampling probability.
        costs = actual_round_costs(
            cfg,
            public["orders"],
            plan.group_radii,
            plan.final_scale,
            dict(plan.layer_clip_weights),
            {g: float(layer_index[t, g]["noise_std"]) for g in registry.names},
            plan.group_rates,
            {
                g: (
                    float(layer_index[t, g]["recycle_prob"]),
                    float(layer_index[t, g]["tau"]),
                )
                for g in registry.names
            },
        )
        increments.append(costs)
        charged = ledger.accept(t, costs)
        for g, row in enumerate(charged):
            full = {
                **row,
                "clients": int(public["counts"][g]),
                "sample_rate": plan.group_rates[g],
                "clip_raw": plan.group_radii[g],
                "denominator": plan.denominator,
            }
            _compare_row(group_index[t, g], full)
        last_group_rows = charged
        released = torch.load(
            root / "released" / f"round_{t:03d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if set(released) != set(registry.names) or any(
            x.ndim != 1
            or x.numel() != registry.dimensions[g]
            or not torch.isfinite(x).all()
            for g, x in released.items()
        ):
            raise ValueError("Invalid published tensors")
        registry.apply(model, released)
        direction.observe(t, released)
        _close(rounds[t - 1]["round"], t, "round order")
        _close(rounds[t - 1]["denominator"], plan.denominator, "denominator")
        _close(
            rounds[t - 1]["local_epochs_completed"],
            t * cfg["local_epochs"],
            "local epoch clock",
        )
        _close(
            rounds[t - 1]["scheduler_last_epoch"],
            t * cfg["local_epochs"],
            "cosine clock",
        )
        expected_lr = (
            cfg["lr"] * (1 + math.cos(math.pi * t * cfg["local_epochs"] / 200)) / 2
        )
        _close(rounds[t - 1]["next_local_lr"], expected_lr, "cosine learning rate")
        _close(
            rounds[t - 1]["non_gaussian_layers"],
            sum(r > 0 for r, _ in shapes.values()),
            "shape count",
        )
        for g, name in enumerate(
            ["epsilon_group_10", "epsilon_group_20", "epsilon_group_30"]
        ):
            _close(rounds[t - 1][name], charged[g]["epsilon_guarantee"], name)
    with np.load(root / "accountant.npz") as archive:
        for key, expected in [
            ("orders", public["orders"]),
            ("actual_increments", np.asarray(increments)),
            ("actual_rdp", ledger.actual),
            ("uniform_rdp", ledger.uniform),
            ("targets", np.asarray(cfg["epsilons"])),
        ]:
            if archive[key].shape != expected.shape or not np.allclose(
                archive[key], expected, rtol=1e-10, atol=1e-12
            ):
                raise ValueError("RDP archive mismatch")
    final = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    if set(final) != set(model.state_dict()) or any(
        not torch.allclose(v, final[k], rtol=1e-5, atol=1e-6)
        for k, v in model.state_dict().items()
    ):
        raise ValueError("Final model does not match released history")
    state = torch.load(
        root / "directions_state.pt", map_location="cpu", weights_only=True
    )
    if state["last_observed"] != cfg["rounds"] or any(
        not torch.equal(state["ema"][g], direction.ema[g]) for g in registry.names
    ):
        raise ValueError("Direction final replay mismatch")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (
        summary["status"] != "completed"
        or summary["accountant"] != ACCOUNTANT_VERSION
        or summary["split_checksum"] != checksum
    ):
        raise ValueError("Summary scope/status mismatch")
    if "public_train_file" in cfg and (
        summary.get("evaluation_examples") != cfg.get("validation_count", 10000)
        or summary.get("client_training_examples") != client_training_count(cfg)
        or summary.get("public_source") != "official_training_split"
        or ("validation_count" in cfg and summary.get("evaluation_split") != "independent_validation")
    ):
        raise ValueError("Training-public summary data counts mismatch")
    from .dvpa_test_public64 import verify_evaluation

    verify_evaluation(cfg, root, summary)
    _close(summary["non_gaussian_layer_releases"], non_gaussian, "summary shape count")
    _close(
        summary["final_test_accuracy"],
        float(rounds[-1]["test_accuracy"]),
        "final metric",
    )
    _close(
        summary["best_test_accuracy"],
        max(float(r["test_accuracy"]) for r in rounds),
        "best metric",
    )
    for g, row in enumerate(last_group_rows):
        for key, value in row.items():
            actual = summary["group_privacy"][g][key]
            if isinstance(value, bool):
                if actual is not value:
                    raise ValueError("Actual-path diagnostic label mismatch")
            elif isinstance(value, (int, float)):
                _close(actual, value, "summary group privacy")
            elif actual != value:
                raise ValueError("Summary group field mismatch")
    client_rows = _rows(root / "client_privacy.csv")
    if len(client_rows) != cfg["clients"] or {
        int(r["client_slot"]) for r in client_rows
    } != set(range(cfg["clients"])):
        raise ValueError("Client privacy coverage mismatch")
    for row in client_rows:
        i = int(row["client_slot"])
        g = int(partition["group_ids"][i])
        _compare_row(
            row,
            dict(
                client_slot=i,
                group=g,
                epsilon_target=cfg["epsilons"][g],
                delta=cfg["delta"],
                epsilon_guarantee=last_group_rows[g]["epsilon_guarantee"],
                epsilon_actual_path_accounting=last_group_rows[g][
                    "epsilon_actual_path_accounting"
                ],
            ),
        )
    return dict(
        status="passed",
        rounds=cfg["rounds"],
        groups=3,
        orders=len(public["orders"]),
        group_epsilon_guarantees=[r["epsilon_guarantee"] for r in last_group_rows],
        group_actual_path_accounting=[
            r["epsilon_actual_path_accounting"] for r in last_group_rows
        ],
        non_gaussian_layer_releases=non_gaussian,
        scope="artifact/source/accounting/history audit; not runtime proof or raw optimizer replay",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    result = audit(args.results_dir)
    (Path(args.results_dir) / "audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
