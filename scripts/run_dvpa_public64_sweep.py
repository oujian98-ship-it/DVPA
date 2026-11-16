"""Run ten fresh FMNIST seed42 DVPA configurations on the public64 protocol."""

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

from scripts.run_dvpa_comparison import normalized_privacy, verify_identity
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


REFERENCE = ROOT / "results/trainpublic64_fmnist_seed42_dvpa_idp_sayg_20261106"
OUTPUT = ROOT / "results/dvpa_public64_fmnist_seed42_sweep10_20261109"
CONFIG = ROOT / "configs/dvpa_fmnist_trainpublic64_weight10_seed42.yaml"


def planned_jobs(reference):
    time_grid = sorted(reference["controller"]["time_multipliers"] + [0.625, 0.875, 1.125, 1.375])
    return [
        ("01_weight5", {"controller": {"information_value_weight": 5}}),
        ("02_weight20", {"controller": {"information_value_weight": 20}}),
        ("03_weight1", {"controller": {"information_value_weight": 1}}),
        ("04_weight3", {"controller": {"information_value_weight": 3}}),
        ("05_weight15", {"controller": {"information_value_weight": 15}}),
        ("06_layer_floor_001", {"controller": {"p_min": 0.01}}),
        ("07_layer_floor_004", {"controller": {"p_min": 0.04}}),
        ("08_dense_time_grid", {"controller": {"time_multipliers": time_grid}}),
        ("09_round_floor_010", {"privacy": {"floor_precision_ratio": 0.1}}),
        ("10_weight5_layer_floor_001", {"controller": {"information_value_weight": 5, "p_min": 0.01}}),
    ]


def report(out, state):
    _json(out / "suite_state.json", state)
    rows = sorted(state["completed"], key=lambda row: (-row["peak_accuracy"], row["name"]))
    if rows:
        write_csv(out / "results_by_peak.csv", rows)
        write_csv(out / "group_privacy.csv", state["group_privacy"])
    lines = [
        "# DVPA FMNIST seed42 参数扫描",
        "",
        f"状态 `{state['status']}`，已完成并审计 {len(rows)}/10，失败 {len(state['failed'])}。当前任务 `{state['current_job']}`。",
        "每组重新训练25轮，使用相同初始化、客户端分区和隐私组。64张公开辅助，59,936张客户端训练，10,000张官方测试用于评估。",
        f"历史同协议10倍权重参考峰值为 {state['reference_peak']:.2f}%。各次运行独立生成采样和DP噪声。以下峰值用于单seed探索性调参。",
        "",
        "| 组 | 相对历史配置的改动 | 最高准确率 | 峰值轮次 | 末轮准确率 | 峰值差（百分点） |",
        "|---|---|---:|---:|---:|---:|",
    ]
    completed = {row["name"]: row for row in rows}
    planned = {row["name"]: row["parameters"] for row in state["plan"]}
    for name in [*completed, *(name for name in planned if name not in completed)]:
        description = "，".join(
            f"`{key}`={value}" for changes in planned[name].values() for key, value in changes.items()
        )
        row = completed.get(name)
        if row is None:
            lines.append(f"| {name} | {description} | — | — | — | — |")
        else:
            lines.append(
                f"| {name} | {description} | {row['peak_accuracy']:.2f}% | {row['peak_round']} | "
                f"{row['final_accuracy']:.2f}% | {row['peak_delta_pp']:+.2f} |"
            )
    lines.extend(["", "隐私过滤器逐组核验 ε 上限10／20／30，δ=10⁻⁵。各组完整25轮的实际RDP与认证费用记录在 group_privacy.csv。"])
    if state["failed"]:
        lines.extend(["", "失败记录"])
        lines.extend(f"- {row['name']} `{row['error']}`" for row in state["failed"])
    (out / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify all ten formal configurations without training")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("The formal queue requires the CUDA training device")
    gate = verify_gate()
    reference = load_config(CONFIG)
    previous = REFERENCE / "DVPA_seed42"
    if load_config(previous / "config.yaml") != reference:
        raise ValueError("The sweep base must match the historical public64 reference run")
    if json.loads((previous / "audit.json").read_text(encoding="utf-8"))["status"] != "passed":
        raise ValueError("Historical reference did not pass the privacy audit")
    reference_peak = json.loads((previous / "summary.json").read_text(encoding="utf-8"))["best_test_accuracy"]
    identity = json.loads((REFERENCE / "expected_identity.json").read_text(encoding="utf-8"))
    verify_identity(previous, identity)
    verify_validation(reference)
    jobs = []
    for name, changes in planned_jobs(reference):
        cfg = copy.deepcopy(reference)
        for section, updates in changes.items():
            cfg[section].update(updates)
        cfg = resolve(cfg)
        verify_formal(cfg)
        protocol = public_preflight(cfg)
        jobs.append((name, cfg, changes, protocol.orders.tolist()))
    if len(jobs) != 10 or len({json.dumps(job[1], sort_keys=True) for job in jobs}) != 10:
        raise ValueError("Expected ten distinct configurations")
    if args.check:
        print(f"Preflight passed for {len(jobs)} distinct DVPA runs. Existing regression gate: {gate['tests_passed']} tests.")
        return

    OUTPUT.mkdir(exist_ok=False)
    _json(OUTPUT / "test_gate.json", gate)
    (OUTPUT / "driver_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (OUTPUT / "configs").mkdir()
    for name, cfg, _, _ in jobs:
        (OUTPUT / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    state = dict(
        status="running", process_id=os.getpid(), seed=42, current_job=None,
        started_unix=time.time(), reference_peak=reference_peak,
        plan=[dict(name=name, parameters=changes) for name, _, changes, _ in jobs],
        completed=[], failed=[], group_privacy=[],
    )
    report(OUTPUT, state)
    try:
        train, test = load_datasets(reference["base"])
        if len(train) != 59936 or len(test) != 10000:
            raise ValueError("Incorrect public64 data protocol")
        del train
        device = torch.device("cuda:0")
        for name, cfg, _, orders in jobs:
            folder = OUTPUT / name
            state.update(current_job=name, current_job_started_unix=time.time())
            report(OUTPUT, state)
            print(f"START {name}", flush=True)
            try:
                summary = run(cfg, folder)
                checked = audit(folder)
                _json(folder / "audit.json", checked)
                verify_identity(folder, identity)
                record = metric_record(folder, name, summary, checked, False)
                model = source_model()("fmnist", 125).to(device)
                model.load_state_dict(torch.load(folder / "model.pt", map_location=device, weights_only=True))
                loss, accuracy = evaluate(model, test, device)
                del model
                if not np.isclose(accuracy, summary["final_test_accuracy"], rtol=0, atol=1e-9):
                    raise ValueError("Full-test accuracy recheck mismatch")
                _json(folder / "evaluation_recheck.json", dict(examples=len(test), accuracy=accuracy, cross_entropy=loss))
                privacy = normalized_privacy(folder, "DVPA", orders)
                peak = summary["best_test_accuracy"]
                state["completed"].append(dict(
                    name=name, peak_accuracy=peak, peak_round=record["best_round"],
                    final_accuracy=accuracy, peak_delta_pp=round(peak - reference_peak, 4),
                    seconds=round(time.time() - state["current_job_started_unix"], 1), audit=checked["status"],
                ))
                state["group_privacy"].extend(dict(name=name, **row) for row in privacy)
                print(f"DONE {name} peak={peak:.2f}% final={accuracy:.2f}%", flush=True)
            except Exception as exc:
                import traceback

                traceback.print_exc()
                state["failed"].append(dict(name=name, error=f"{type(exc).__name__}: {exc}"))
            state["current_job"] = None
            report(OUTPUT, state)
            torch.cuda.empty_cache()
        state["status"] = "completed_with_failures" if state["failed"] else "completed"
    except BaseException:
        state["status"] = "interrupted"
        raise
    finally:
        state.update(current_job=None, ended_unix=time.time())
        report(OUTPUT, state)
    if state["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
