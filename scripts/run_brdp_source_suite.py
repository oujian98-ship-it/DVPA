"""Serial author-matched formal suite, requiring an unchanged tested code gate."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from src.brdp_group_accountant import epsilon_from_rdp
from src.brdp_source_audit import audit
from src.brdp_source_protocol import (
    load_datasets,
    partition_and_groups,
    preflight,
    validate_config,
)
from src.brdp_source_runner import _json, run
from src.data import split_checksum
from src.layer_registry import LayerRegistry
from src.upstream_runner import source_model

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_GATE_FILES = {
    "src/brdp_group_accountant.py",
    "src/brdp_source_protocol.py",
    "src/brdp_source_runner.py",
    "src/brdp_source_audit.py",
    "scripts/run_brdp_source_suite.py",
    "configs/brdp_source_brdp_joint_seed42.yaml",
    "configs/brdp_source_gaussian_joint_seed42.yaml",
}


def workspace_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Suite paths must stay inside BR_DP")
    return path


def verify_gate(relative="reports/brdp_source_protocol/test_gate.json"):
    gate = json.loads(workspace_path(relative).read_text(encoding="utf-8"))
    if gate.get("status") != "passed" or gate.get("tests_passed", 0) < 74:
        raise RuntimeError("Formal experiments require a passed test gate")
    if not REQUIRED_GATE_FILES <= set(gate["source_sha256"]):
        raise RuntimeError("Incomplete tested source manifest")
    for relative, digest in gate["source_sha256"].items():
        if hashlib.sha256(workspace_path(relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(
                "Tested source/config changed: rerun tests before formal training"
            )
    return gate


def jobs():
    records = []
    for mechanism in ["brdp", "gaussian"]:
        path = workspace_path(f"configs/brdp_source_{mechanism}_joint_seed42.yaml")
        cfg = validate_config(yaml.safe_load(path.read_text(encoding="utf-8")))
        required = dict(
            seed=42,
            clients=100,
            rounds=25,
            local_epochs=30,
            batch_size=125,
            lr=0.001,
            rate=0.9,
            clip=250.0,
            delta=1e-5,
            transition_round=13,
            epsilons=[10.0, 20.0, 30.0],
            group_proportions=[0.34, 0.43, 0.23],
            q_save=[0.5, 0.6, 0.7],
            q_spend=0.9,
            method="JOINT",
            mechanism=mechanism,
        )
        if (
            any(cfg[k] != v for k, v in required.items())
            or cfg.get("dataset_mode", "official") != "official"
        ):
            raise ValueError("Formal suite must match the locked screenshot protocol")
        records.append(dict(name=f"{mechanism}_joint_seed42", config=cfg))
    return records


def inspect_formal_inputs():
    records = jobs()
    train, test = load_datasets(records[0]["config"])
    parts, ids = partition_and_groups(records[0]["config"], train.tensors[1])
    dimensions = LayerRegistry(source_model()("fmnist", 125)).dimensions
    details = []
    for job in records:
        public = preflight(job["config"], dimensions)
        details.append(
            dict(
                name=job["name"],
                group_counts=public["counts"].tolist(),
                final_epsilon_uniform_envelope=public["final_epsilon"],
                group_clip_raw=public["group_radii"].tolist(),
                orders=len(public["orders"]),
            )
        )
    return dict(
        status="passed",
        parameters=sum(dimensions.values()),
        train_samples=len(train),
        test_samples=len(test),
        normalization="[-1,1]",
        pixel_min=float(train.tensors[0].min()),
        pixel_max=float(train.tensors[0].max()),
        partition_checksum=split_checksum(parts),
        group_counts=np.bincount(ids).tolist(),
        local_training="all clients; persistent SGD/momentum and per-local-epoch cosine",
        jobs=details,
    )


def run_suite(output):
    gate = verify_gate()
    checked = inspect_formal_inputs()
    out = workspace_path(output)
    out.mkdir(parents=True, exist_ok=False)
    _json(out / "formal_inputs.json", checked)
    _json(out / "test_gate.json", gate)
    state = dict(
        status="running",
        started_unix=time.time(),
        current_job=None,
        completed=[],
        per_run_privacy_targets=[10.0, 20.0, 30.0],
        privacy_scope="each run separately; two-run composition reported after completion",
    )
    _json(out / "suite_state.json", state)
    completed = []
    try:
        for job in jobs():
            state["current_job"] = job["name"]
            _json(out / "suite_state.json", state)
            progress = workspace_path(
                ".private_diagnostics/brdp_source_suite_progress.json"
            )
            summary = run(job["config"], out / job["name"], progress)
            check = audit(out / job["name"])
            _json(out / job["name"] / "audit.json", check)
            completed.append((job["name"], summary))
            state["completed"].append(
                dict(name=job["name"], summary=summary, audit=check)
            )
            _json(out / "suite_state.json", state)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        first = json.loads(
            (out / completed[0][0] / "partition.json").read_text(encoding="utf-8")
        )
        total = None
        for name, _ in completed:
            partition = json.loads(
                (out / name / "partition.json").read_text(encoding="utf-8")
            )
            if partition != first:
                raise ValueError(
                    "Cannot compose groups with different frozen client assignments"
                )
            with np.load(out / name / "accountant.npz") as archive:
                total = (
                    archive["uniform_rdp"].copy()
                    if total is None
                    else total + archive["uniform_rdp"]
                )
                orders = archive["orders"].copy()
        combined = [epsilon_from_rdp(orders, total[g], 1e-5)[0] for g in range(3)]
        _json(
            out / "combined_release_privacy.json",
            dict(
                scope="both released runs on the same fixed client slots",
                delta=1e-5,
                group_epsilon_guarantees=combined,
                warning="The single-run 10/20/30 targets do not cover both runs jointly",
            ),
        )
        state["status"] = "completed"
        state["current_job"] = None
        state["combined_release_privacy"] = combined
        state["single_seed_accuracy_difference_brdp_minus_gaussian"] = (
            completed[0][1]["final_test_accuracy"]
            - completed[1][1]["final_test_accuracy"]
        )
        _json(out / "suite_state.json", state)
        return state
    except Exception as exc:
        state.update(
            status="failed", failure_type=type(exc).__name__, failure_message=str(exc)
        )
        _json(out / "suite_state.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="results/brdp_source_matched_seed42_20261027"
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        result = inspect_formal_inputs()
        _json(
            workspace_path("reports/brdp_source_protocol/formal_input_preflight.json"),
            result,
        )
    else:
        result = run_suite(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
