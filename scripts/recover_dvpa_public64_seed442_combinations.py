"""Finish the interrupted seed442 DVPA combination sweep without overwriting runs."""

import os
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts import run_dvpa_public64_seed442_combinations as sweep
from scripts.run_dvpa_comparison import normalized_privacy, verify_identity
from scripts.run_train_public_baselines import metric_record
from src.brdp_source_protocol import load_datasets
from src.brdp_source_runner import _json
from src.dvpa_audit import audit
from src.dvpa_config import load_config
from src.dvpa_public64 import verify_validation
from src.dvpa_runner import run
from src.metrics import evaluate
from src.st_runner import write_csv
from src.upstream_runner import source_model


def write_report(state):
    _json(sweep.OUTPUT / "suite_state.json", state)
    complete = sorted(state["completed"], key=lambda row: (-row["peak_accuracy"], row["name"]))
    completed = {row["name"]: row for row in complete}
    params = {row["name"]: row["parameters"] for row in state["plan"]}
    baseline = completed.get("01_seed442_baseline")
    lines = [
        "# DVPA FMNIST seed442 多参数组合扫描", "",
        f"状态 `{state['status']}`；完整且通过审计的参数组 {len(complete)}/20；失败 {len(state['failed'])} 组。",
        "统一协议：FMNIST，官方训练集固定留出64张公开辅助数据、其余59,936张客户端训练，官方10,000张测试集仅评估；100客户端、25轮、seed442。",
        "每组有独立的客户端采样与DP噪声。隐私会计按每次完整训练运行分别核验ε=10/20/30、δ=10⁻⁵；不同调参运行若作为同一数据主体的联合发布，需另行组合会计。",
        "峰值准确率以每轮官方测试集准确率的最高值为准；差值相对同seed第01组。seed42的70.32%仅作背景参照。", "",
        "|排名|参数组|联动设置|最高准确率|峰值轮|末轮准确率|相对seed442基准变化（百分点）|",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    rank = 0
    for row in complete:
        name = row["name"]
        cfg = params[name]
        change = "，".join(f"`{key}`={value}" for section in cfg.values() for key, value in section.items())
        rank += 1
        delta = "基准" if name == "01_seed442_baseline" else f"{row['peak_accuracy'] - baseline['peak_accuracy']:+.2f}" if baseline else "—"
        lines.append(f"|{rank}|{name}|{change}|{row['peak_accuracy']:.2f}%|{row['peak_round']}|{row['final_accuracy']:.2f}%|{delta}|")
    for name, cfg in params.items():
        if name not in completed:
            change = "，".join(f"`{key}`={value}" for section in cfg.values() for key, value in section.items())
            lines.append(f"|—|{name}|{change}|未完成|—|—|—|")
    lines += [
        "", "完整每组隐私会计记录见 `group_privacy.csv`；可复算排序见 `results_by_peak.csv`。",
        "", "恢复记录：原第16组运行在已保存17/25轮后中断，检查点标记 `resume_supported=false`。原始部分结果保留在 `16_weight15_layer04_round_floor028`；重新完整运行写入独立目录 `16_weight15_layer04_round_floor028_retry01`，不覆盖原记录。表中只采用完整且通过审计的运行。原中断运行已经发布的17轮仍是隐私消耗，`interrupted_attempts.json` 记录该次尝试。",
    ]
    if state["failed"]:
        lines += ["", "失败记录", *(f"- {row['name']}：`{row['error']}`" for row in state["failed"])]
    (sweep.OUTPUT / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(sweep.OUTPUT / "results_by_peak.csv", complete)
    write_csv(sweep.OUTPUT / "group_privacy.csv", state["group_privacy"])


def main():
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full training")

    output = sweep.OUTPUT
    state_path = output / "suite_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    prior_status = state.get("status")
    prior_current = state.get("current_job")
    snapshot = output / "suite_state_before_recovery.json"
    if not snapshot.exists():
        snapshot.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    interrupted = output / "16_weight15_layer04_round_floor028"
    interrupted_state_path = interrupted / "run_state.json"
    interrupted_state = json.loads(interrupted_state_path.read_text(encoding="utf-8"))
    interrupted_history = {
        "name": "16_weight15_layer04_round_floor028",
        "status_before_recovery": prior_status,
        "queue_current_job": prior_current,
        "completed_rounds": interrupted_state.get("completed_rounds"),
        "reserved_round": interrupted_state.get("reserved_round"),
        "resume_supported": interrupted_state.get("resume_supported"),
        "spent_at_interruption": interrupted_state.get("spent"),
        "original_folder": str(interrupted),
        "recorded_unix": time.time(),
    }
    _json(output / "interrupted_attempts.json", [interrupted_history])

    reference = load_config(sweep.BASE_CONFIG)
    verify_validation(reference)
    jobs = sweep.build_jobs(reference)
    plan = {row["name"]: row for row in state["plan"]}
    complete_names = {row["name"] for row in state["completed"]}
    _, test_data = load_datasets(reference["base"])
    if len(test_data) != 10000:
        raise ValueError("Expected the full official test set")
    device = torch.device("cuda:0")
    baseline_folder = output / "01_seed442_baseline"
    expected_identity = sweep.identity_from_run(baseline_folder)

    state.update(status="recovering", process_id=os.getpid(), current_job=None,
                 recovery_started_unix=time.time())
    write_report(state)
    try:
        for name, config, _, _, orders in jobs:
            if name in complete_names:
                continue
            if name not in plan:
                raise ValueError(f"Missing original plan record: {name}")
            folder = output / (name + "_retry01") if name == "16_weight15_layer04_round_floor028" else output / name
            state.update(current_job=name, current_job_started_unix=time.time())
            write_report(state)
            print(f"RECOVERY START {name} seed=442 output={folder.name}", flush=True)
            try:
                summary = run(config, folder)
                checked = audit(folder)
                _json(folder / "audit.json", checked)
                metric = metric_record(folder, name, summary, checked, False)
                identity = verify_identity(folder, expected_identity)
                model = source_model()("fmnist", 125).to(device)
                model.load_state_dict(torch.load(folder / "model.pt", map_location=device, weights_only=True))
                loss, accuracy = evaluate(model, test_data, device)
                del model
                if not np.isclose(accuracy, summary["final_test_accuracy"], rtol=0, atol=1e-9):
                    raise ValueError("Full-test evaluation recheck failed")
                _json(folder / "evaluation_recheck.json", dict(examples=10000, accuracy=accuracy, cross_entropy=loss))
                privacy = normalized_privacy(folder, "DVPA", orders)
                state["completed"].append(dict(
                    name=name, peak_accuracy=summary["best_test_accuracy"], peak_round=metric["best_round"],
                    final_accuracy=accuracy,
                    seconds=round(time.time() - state["current_job_started_unix"], 1),
                    audit=checked["status"],
                    same_initialization_and_partition=identity["initial_models_identical"] and identity["partitions_and_groups_identical"],
                    result_folder=folder.name,
                ))
                state["group_privacy"].extend(dict(name=name, **row) for row in privacy)
                print(f"RECOVERY DONE {name} peak={summary['best_test_accuracy']:.2f}% final={accuracy:.2f}%", flush=True)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                state["failed"].append(dict(name=name, error=f"{type(exc).__name__}: {exc}"))
                print(f"RECOVERY FAILED {name}: {type(exc).__name__}: {exc}", flush=True)
            state["current_job"] = None
            write_report(state)
            torch.cuda.empty_cache()
        state["status"] = "completed_with_failures" if state["failed"] else "completed_with_interrupted_attempt_retried"
    except BaseException:
        state["status"] = "interrupted_during_recovery"
        raise
    finally:
        state.update(current_job=None, recovery_ended_unix=time.time())
        write_report(state)


if __name__ == "__main__":
    main()
