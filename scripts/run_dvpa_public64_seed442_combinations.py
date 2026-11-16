"""Run twenty joint-parameter DVPA trials on the public64 FMNIST protocol."""

import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.run_dvpa_comparison import normalized_privacy, tensor_digest, verify_identity
from scripts.run_dvpa_suite import verify_formal, verify_gate
from scripts.run_train_public_baselines import metric_record
from src.brdp_source_protocol import load_datasets
from src.brdp_source_runner import _json
from src.dvpa_audit import audit
from src.dvpa_config import ROOT, load_config, public_preflight, resolve
from src.dvpa_public64 import verify_validation
from src.dvpa_runner import run
from src.metrics import evaluate
from src.st_runner import write_csv
from src.upstream_runner import source_model

SEED = 442
OUTPUT = ROOT / "results/dvpa_public64_fmnist_seed442_combinations20_20261110"
BASE_CONFIG = ROOT / "configs/dvpa_fmnist_trainpublic64_weight10_seed42.yaml"
REFERENCE_SEED42 = ROOT / "results/trainpublic64_fmnist_seed42_dvpa_idp_sayg_20261106/DVPA_seed42"


def planned_jobs(reference):
    jobs = [
        ("01_seed442_baseline", {"controller": {"information_value_weight": 10, "p_min": 0.02}}, "Same-seed reference"),
        ("02_repeat_group07", {"controller": {"information_value_weight": 10, "p_min": 0.04}}, "Repeat group07 settings"),
        ("03_repeat_group05", {"controller": {"information_value_weight": 15, "p_min": 0.02}}, "Repeat group05 settings"),
        ("15_weight15_layer_floor_06", {"controller": {"information_value_weight": 15, "p_min": 0.06}},
         "Extend the layer-floor and weight interaction above 0.05"),
    ]
    index = 4
    for weight in (5, 10, 15, 20):
        for floor in (0.03, 0.04, 0.05):
            if (weight, floor) == (10, 0.04):
                continue
            jobs.append((
                f"{index:02d}_weight{weight}_layer_floor_{int(floor * 100):02d}",
                {"controller": {"information_value_weight": weight, "p_min": floor}},
                "Joint information-value weight and layer floor",
            ))
            index += 1
    for suffix, ratio in (("012", 0.12), ("028", 0.28)):
        jobs.append((
            f"{index:02d}_weight15_layer04_round_floor{suffix}",
            {"controller": {"information_value_weight": 15, "p_min": 0.04},
             "privacy": {"floor_precision_ratio": ratio}},
            "Joint weight, layer floor, and round precision floor",
        ))
        index += 1
    dense = sorted(reference["controller"]["time_multipliers"] + [0.625, 0.875, 1.125, 1.375])
    jobs.extend([
        (
            "18_weight15_layer04_dense_time",
            {"controller": {"information_value_weight": 15, "p_min": 0.04, "time_multipliers": dense}},
            "Joint weight, layer floor, and denser time search",
        ),
        (
            "19_weight10_layer05_narrow_time",
            {"controller": {"information_value_weight": 10, "p_min": 0.05,
                            "time_multipliers": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]}},
            "Joint weight, layer floor, and narrower time search",
        ),
        (
            "20_weight15_layer04_finer_local",
            {"controller": {"information_value_weight": 15, "p_min": 0.04,
                            "local_time_offsets": [0.005, 0.01, 0.02, 0.03, 0.04],
                            "local_layer_transfers": [0.0005, 0.001, 0.002, 0.003, 0.005],
                            "pair_transfer": 0.025}},
            "Joint weight, layer floor, and finer local search",
        ),
    ])
    if len(jobs) != 20:
        raise ValueError(f"Expected 20 combinations, found {len(jobs)}")
    return jobs


def build_jobs(reference):
    jobs = []
    for name, changes, rationale in planned_jobs(reference):
        config = copy.deepcopy(reference)
        config["base"]["seed"] = SEED
        for section, values in changes.items():
            config[section].update(values)
        config = resolve(config)
        verify_formal(config)
        orders = public_preflight(config).orders.tolist()
        jobs.append((name, config, changes, rationale, orders))
    if len({json.dumps(config, sort_keys=True) for _, config, _, _, _ in jobs}) != 20:
        raise ValueError("Configurations must be distinct")
    return jobs


def write_report(state):
    _json(OUTPUT / "suite_state.json", state)
    ordered = sorted(state["completed"], key=lambda row: (-row["peak_accuracy"], row["name"]))
    if ordered:
        write_csv(OUTPUT / "results_by_peak.csv", ordered)
        write_csv(OUTPUT / "group_privacy.csv", state["group_privacy"])
    baseline = next((row for row in state["completed"] if row["name"] == "01_seed442_baseline"), None)
    completed = {row["name"]: row for row in ordered}
    parameters = {row["name"]: row["parameters"] for row in state["plan"]}
    lines = [
        "# DVPA FMNIST seed442 多参数组合扫描",
        "",
        f"状态 `{state['status']}`，完成并审计 {len(ordered)}/20 组，失败 {len(state['failed'])} 组。当前任务 `{state['current_job']}`。",
        "每组均为25轮训练，64张公开辅助、59,936张客户端训练、10,000张官方测试评估。各组使用相同seed442初始化和客户端分区。",
        "不同运行各自生成客户端采样和DP噪声。组间效果以第01组seed442基准计算。seed42的70.32%仅作背景参考。",
        "",
        "| 排名 | 组 | 联动参数 | 最高准确率 | 峰值轮次 | 末轮准确率 | 对seed442基准的变化（百分点） |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for rank, name in enumerate(parameters, 1):
        description = "，".join(f"`{key}`={value}" for section in parameters[name].values() for key, value in section.items())
        row = completed.get(name)
        if row is None:
            lines.append(f"| — | {name} | {description} | — | — | — | — |")
            continue
        delta = "基准" if name == "01_seed442_baseline" else (
            "待基准完成" if baseline is None else f"{row['peak_accuracy'] - baseline['peak_accuracy']:+.2f}"
        )
        lines.append(f"| {rank} | {name} | {description} | {row['peak_accuracy']:.2f}% | {row['peak_round']} | {row['final_accuracy']:.2f}% | {delta} |")
    lines += ["", "各组隐私上限为ε=10／20／30、δ=10⁻⁵。完整25轮实际RDP及认证费用见group_privacy.csv。"]
    if state["failed"]:
        lines += ["", "失败记录", *(f"- {row['name']}：`{row['error']}`" for row in state["failed"])]
    (OUTPUT / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def identity_from_run(folder):
    partition = json.loads((folder / "partition.json").read_text(encoding="utf-8"))
    initial = torch.load(folder / "initial_model.pt", map_location="cpu", weights_only=True)
    return {
        "parts": partition["parts"],
        "group_ids": partition["group_ids"],
        "split_checksum": partition["split_checksum"],
        "initial_tensor_sha256": tensor_digest(initial),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full training")
    gate = verify_gate()
    reference = load_config(BASE_CONFIG)
    verify_formal(reference)
    verify_validation(reference)
    seed42 = json.loads((REFERENCE_SEED42 / "summary.json").read_text(encoding="utf-8"))
    if json.loads((REFERENCE_SEED42 / "audit.json").read_text(encoding="utf-8"))["status"] != "passed":
        raise ValueError("Existing public64 reference did not pass audit")
    jobs = build_jobs(reference)
    if args.check:
        print(f"Preflight passed for {len(jobs)} distinct seed{SEED} configs, regression gate {gate['tests_passed']}.")
        return

    OUTPUT.mkdir(exist_ok=False)
    _json(OUTPUT / "test_gate.json", gate)
    (OUTPUT / "driver_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (OUTPUT / "configs").mkdir()
    for name, config, _, _, _ in jobs:
        (OUTPUT / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    state = dict(
        status="running", process_id=os.getpid(), seed=SEED, current_job=None,
        started_unix=time.time(), reference_seed42_peak=seed42["best_test_accuracy"],
        plan=[dict(name=name, parameters=changes, rationale=why) for name, _, changes, why, _ in jobs],
        completed=[], failed=[], group_privacy=[],
    )
    write_report(state)
    try:
        _, test_data = load_datasets(reference["base"])
        if len(test_data) != 10000:
            raise ValueError("Expected the full official test set")
        device = torch.device("cuda:0")
        expected_identity = None
        for name, config, _, _, orders in jobs:
            folder = OUTPUT / name
            state.update(current_job=name, current_job_started_unix=time.time())
            write_report(state)
            print(f"START {name} seed={SEED}", flush=True)
            try:
                summary = run(config, folder)
                checked = audit(folder)
                _json(folder / "audit.json", checked)
                metric = metric_record(folder, name, summary, checked, False)
                if expected_identity is None:
                    expected_identity = identity_from_run(folder)
                identity = verify_identity(folder, expected_identity)
                model = source_model()("fmnist", 125).to(device)
                model.load_state_dict(torch.load(folder / "model.pt", map_location=device, weights_only=True))
                loss, accuracy = evaluate(model, test_data, device)
                del model
                if not np.isclose(accuracy, summary["final_test_accuracy"], rtol=0, atol=1e-9):
                    raise ValueError("Full-test evaluation recheck failed")
                _json(folder / "evaluation_recheck.json", dict(examples=10000, accuracy=accuracy, cross_entropy=loss))
                privacy = normalized_privacy(folder, "DVPA", orders)
                peak = summary["best_test_accuracy"]
                state["completed"].append(dict(
                    name=name, peak_accuracy=peak, peak_round=metric["best_round"],
                    final_accuracy=accuracy, seconds=round(time.time()-state["current_job_started_unix"], 1),
                    audit=checked["status"],
                    same_initialization_and_partition=identity["initial_models_identical"] and identity["partitions_and_groups_identical"],
                ))
                state["group_privacy"].extend(dict(name=name, **row) for row in privacy)
                print(f"DONE {name} peak={peak:.2f}% final={accuracy:.2f}%", flush=True)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                state["failed"].append(dict(name=name, error=f"{type(exc).__name__}: {exc}"))
            state["current_job"] = None
            write_report(state)
            torch.cuda.empty_cache()
        state["status"] = "completed_with_failures" if state["failed"] else "completed"
    except BaseException:
        state["status"] = "interrupted"
        raise
    finally:
        state.update(current_job=None, ended_unix=time.time())
        write_report(state)
    if state["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
