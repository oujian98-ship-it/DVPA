"""Fresh frozen-current-DVPA / IDP / SAYG runs on FMNIST and MNIST, seeds42-44."""

import argparse
import contextlib
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# PyTorch and the conda MKL wheel otherwise load two OpenMP runtimes on this
# host. Sequential MKL avoids that conflict; do not suppress duplicate checks.
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import yaml

from scripts.run_dvpa_comparison import (
    dataset_digest, normalized_privacy, tensor_digest, verify_identity,
)
from scripts.run_dvpa_suite import verify_formal, verify_gate
from src.brdp_source_audit import audit as audit_baseline
from src.brdp_source_protocol import load_datasets, partition_and_groups, preflight
from src.brdp_source_runner import _json, run as run_baseline
from src.data import split_checksum
from src.dvpa_audit import audit as audit_dvpa
from src.dvpa_config import ROOT, load_config, public_preflight, resolve, smoke_config
from src.dvpa_dataset_support import (
    PublicCurvatureProxy, baseline_evaluation, create_transfer_validation,
    make_public_mnist, verify_forecast_validation, verify_public_validation,
)
from src.dvpa_runner import inside_output, run as run_dvpa
from src.layer_registry import LayerRegistry
from src.metrics import evaluate
from src.st_runner import write_csv
from src.upstream_runner import source_model

DATASETS = ("fmnist", "mnist")
SEEDS = (42, 43, 44)
LABELS = ("DVPA", "IDP_FedAvg", "Time_Adaptive")
TITLES = {"DVPA": "当前 DVPA", "IDP_FedAvg": "IDP-FedAvg 适配版",
          "Time_Adaptive": "spend-as-you-go 适配版"}
CONFIG_DIR = ROOT / "configs/dvpa_datasets_20261105"


def configuration(dataset, seed, smoke=False):
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError("Only requested two datasets and seeds42/43/44 are allowed")
    cfg = load_config(ROOT / "configs/dvpa_fmnist_forecast_seed42.yaml")
    cfg["base"].update(dataset=dataset, seed=seed)
    if dataset == "mnist":
        path = ROOT / "data/public_mnist_proxy.pt"
        cfg["feedback"].update(
            public_proxy_file="data/public_mnist_proxy.pt",
            public_proxy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            public_proxy_validation="reports/dvpa_datasets/mnist_transfer_validation.json",
            forecast_validation="reports/dvpa_datasets/mnist_transfer_validation.json",
        )
    cfg["base"].update(
        evaluation_public_file=cfg["feedback"]["public_proxy_file"],
        evaluation_public_sha256=cfg["feedback"]["public_proxy_sha256"],
    )
    cfg = resolve(cfg)
    if smoke:
        cfg = smoke_config(cfg)
        cfg["feedback"]["public_proxy_probes"] = 2
    else:
        verify_formal(cfg)
    return cfg


def jobs(datasets=DATASETS, seeds=SEEDS, smoke=False):
    if (not datasets or len(set(datasets)) != len(datasets)
        or not set(datasets) <= set(DATASETS) or not seeds
        or len(set(seeds)) != len(seeds) or not set(seeds) <= set(SEEDS)):
        raise ValueError("Invalid or duplicate dataset/seed request")
    records = []
    for dataset in datasets:
        for seed in seeds:
            dvpa = configuration(dataset, seed, smoke)
            for label, method in [("DVPA", None), ("IDP_FedAvg", "IDP_FEDAVG"),
                                  ("Time_Adaptive", "SPEND_AS_YOU_GO")]:
                cfg = dvpa if method is None else copy.deepcopy(dvpa["base"])
                if method:
                    cfg.update(method=method, mechanism="gaussian", recycle_prob=0.0)
                records.append(dict(dataset=dataset, seed=seed, label=label,
                                    name=f"{dataset}_{label}_seed{seed}", config=cfg))
    return records


def prepare():
    """Create all input artifacts BEFORE the exact-source regression gate."""
    torch.set_num_threads(1)
    original = load_config(ROOT / "configs/dvpa_fmnist_forecast_seed42.yaml")
    verify_public_validation(original)
    verify_forecast_validation(original)
    path = ROOT / "data/public_mnist_proxy.pt"
    if not path.exists():
        make_public_mnist(path, original["base"]["data_root"])
    config = configuration("mnist", 42)
    directory = ROOT / "reports/dvpa_datasets"
    directory.mkdir(parents=True, exist_ok=True)
    proof = create_transfer_validation(config)
    _json(directory / "mnist_transfer_validation.json", proof)
    verify_public_validation(config)
    verify_forecast_validation(config)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    records = jobs()
    for job in records:
        (CONFIG_DIR / (job["name"] + ".yaml")).write_text(
            yaml.safe_dump(job["config"], sort_keys=False), encoding="utf-8")
    inputs = {}
    for dataset in DATASETS:
        cfg = configuration(dataset, 42)
        train, test = load_datasets(cfg["base"])
        held, evaluation = baseline_evaluation(cfg["base"], test)
        if len(train) != 60000 or len(held) != 9000:
            raise ValueError("Formal dataset size mismatch")
        inputs[dataset] = dict(train=dataset_digest(train), full_test=dataset_digest(test),
                               evaluation=evaluation, training_examples=len(train))
        del train, test, held
    # Accountant preflight uses public parameters only, once per distinct method.
    dimensions = LayerRegistry(source_model()("fmnist", 125)).dimensions
    fees = {}
    for job in records[:3]:
        public = public_preflight(job["config"]) if job["label"] == "DVPA" else preflight(job["config"], dimensions)
        fees[job["label"]] = public.as_dict() if job["label"] == "DVPA" else dict(
            final_epsilon=public["final_epsilon"], counts=public["counts"].tolist())
    result = dict(status="passed", jobs=18, datasets=inputs, privacy_preflights=fees,
                  frozen_source_config="configs/dvpa_fmnist_forecast_seed42.yaml",
                  mnist_validation_kind=proof["validation_kind"],
                  mnist_empirical_benefit_validated=False,
                  configurations=[j["name"] + ".yaml" for j in records])
    _json(CONFIG_DIR / "manifest.json", result)
    print(json.dumps(dict(status="prepared", jobs=18, datasets=list(inputs)), ensure_ascii=False), flush=True)
    return result


def statistics(completed):
    keys = [(r["dataset"], r["seed"], r["label"]) for r in completed]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate run in result aggregation")
    output = []
    for dataset in DATASETS:
        for label in LABELS:
            rows = [r for r in completed if r["dataset"] == dataset and r["label"] == label]
            if not rows:
                continue
            values = np.array([r["summary"]["final_test_accuracy"] for r in rows])
            best = np.array([r["summary"]["best_test_accuracy"] for r in rows])
            output.append(dict(dataset=dataset, label=label, n=len(rows),
                complete_seed_set={r["seed"] for r in rows} == set(SEEDS),
                final_accuracy_mean=float(values.mean()),
                final_accuracy_sample_std=float(values.std(ddof=1)) if len(rows) > 1 else None,
                best_accuracy_mean=float(best.mean()),
                best_accuracy_sample_std=float(best.std(ddof=1)) if len(rows) > 1 else None))
    return output


def report(out, state):
    rows, privacy, paired = [], [], []
    for done in state["completed"]:
        summary = done["summary"]
        rows.append(dict(dataset=done["dataset"], seed=done["seed"], label=done["label"],
            final_test_accuracy=summary["final_test_accuracy"], best_test_accuracy=summary["best_test_accuracy"],
            final_cross_entropy=done["evaluation_recheck"]["cross_entropy"],
            evaluation_examples=summary["evaluation_examples"], audit=done["audit"]["status"]))
        for row in done["group_privacy"]:
            privacy.append(dict(dataset=done["dataset"], seed=done["seed"], label=done["label"], **row))
    index = {(r["dataset"], r["seed"], r["label"]): r for r in rows}
    sayg_label = "SAYG" if any(r["label"] == "SAYG" for r in rows) else "Time_Adaptive"
    for dataset in DATASETS:
        for seed in SEEDS:
            if all((dataset, seed, m) in index for m in LABELS):
                value = index[dataset, seed, "DVPA"]["final_test_accuracy"]
                paired.append(dict(dataset=dataset, seed=seed,
                    dvpa_minus_idp_pp=value - index[dataset, seed, "IDP_FedAvg"]["final_test_accuracy"],
                    dvpa_minus_sayg_pp=value - index[dataset, seed, sayg_label]["final_test_accuracy"]))
    stats = statistics(state["completed"])
    for name, values in [("results_by_seed.csv", rows), ("mean_std.csv", stats),
                         ("group_privacy.csv", privacy), ("paired_improvements.csv", paired)]:
        if values:
            write_csv(out / name, values)
    _json(out / "comparison.json", dict(status=state["status"], expected_runs=state["expected_runs"],
        completed_runs=len(rows), results=rows, statistics=stats, group_privacy=privacy,
        paired=paired, privacy_policy="common per-run caps10/20/30; actual costs reported; not equal consumption"))
    lines = ["# 当前 DVPA 与两条 baseline：FMNIST / MNIST × seeds42、43、44", "",
        f"状态：{state['status']}；完成 {len(rows)}/{state['expected_runs']} 次。当前任务：{state.get('current_job')}。", "",
        "所有方法每次全新训练；每数据集60,000张训练、固定1,000张公开机制数据、其余同一9,000张评估。",
        "同数据集同seed共用分区、隐私组和CNN初始化。100客户端，25轮，每轮所有客户端30本地epochs，batch125。",
        "主指标固定为第25轮准确率；中途最高值另列，不按测试最高值选最终模型。样本标准差采用ddof=1。",
        "当前DVPA完整保留排名检查、细搜索、公开曲率和短期预测；第三项协方差校准停用。",
        "IDP保留恒定q=.9；SAYG保留原递推与分阶段q。各自裁剪/噪声/有效会计保留，是同上限完整方法比较。",
        "MNIST固定迁移FMNIST的控制参数，只有公开输入与数值检查，不声称已有MNIST代理收益验证。", ""]
    for dataset in DATASETS:
        lines += [f"## {dataset.upper()}", "", "| 方法 | n | seed42 | seed43 | seed44 | 最终均值±样本标准差 |", "|---|---:|---:|---:|---:|---:|"]
        for label in LABELS:
            matches = [v for v in stats if v["dataset"] == dataset and v["label"] == label]
            stat = matches[0] if matches else None
            cells = [f"{index[dataset, seed, label]['final_test_accuracy']:.4f}%" if (dataset, seed, label) in index else "—" for seed in SEEDS]
            cell = "—" if stat is None else (f"{stat['final_accuracy_mean']:.4f}% ± {stat['final_accuracy_sample_std']:.4f}个百分点" if stat['n'] > 1 else f"{stat['final_accuracy_mean']:.4f}%（仅1次）")
            lines.append(f"| {TITLES[label]} | {stat['n'] if stat else 0} | {' | '.join(cells)} | {cell} |")
        lines += ["", "| Seed | DVPA−IDP，百分点 | DVPA−SAYG，百分点 |", "|---|---:|---:|"]
        for p in paired:
            if p["dataset"] == dataset:
                lines.append(f"| {p['seed']} | {p['dvpa_minus_idp_pp']:+.4f} | {p['dvpa_minus_sayg_pp']:+.4f} |")
        lines.append("")
    lines += ["## 每次运行的三组隐私费用", "", "| 数据集 | Seed | 方法 | 组 | 认证ε上界 | 阶数 | 实际参数累计RDP上界 |", "|---|---:|---|---:|---:|---:|---:|"]
    for p in privacy:
        lines.append(f"| {p['dataset']} | {p['seed']} | {p['label']} | {p['group']+1} | {p['epsilon_guarantee']:.6f} | {p['diagnostic_alpha']} | {p['rdp_path_at_diagnostic_alpha']:.8f} |")
    lines += ["", "共同上限10/20/30、δ=10⁻⁵，费用表示机制参数对应的上界。DVPA路径ε诊断不是更小的全策略保证。",
              "不同阶数RDP不可直接互比；不同方法费用不强行相等。每次运行分别认证，不代表18次联合发布仍满足单次上限。",
              "公开seed控制划分/初始化；真实DP采样与噪声使用独立秘密随机数。部分完成时不作三种子稳定性结论。"]
    if state["smoke"]:
        lines += ["", "本目录为合成工程检查，不能作为FMNIST或MNIST正式效用结果。"]
    (out / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(output, datasets=DATASETS, seeds=SEEDS, smoke=False, enforce_gate=True):
    gate = verify_gate() if enforce_gate else None
    records = jobs(datasets, seeds, smoke)
    out = inside_output(output)
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status="preparing", expected_runs=len(records), current_job=None,
        completed=[], identity_checks=[], seeds=list(seeds), datasets=list(datasets),
        started_unix=time.time(), process_id=os.getpid(), smoke=smoke)
    _json(out / "suite_state.json", state)
    if gate:
        _json(out / "test_gate.json", gate)
    (out / "driver_snapshot.py").write_bytes(Path(__file__).read_bytes())
    report(out, state)
    expected, anchors = {}, {}
    try:
        torch.set_num_threads(1)
        input_records = {}
        for dataset in datasets:
            cfg = next(j["config"] for j in records if j["dataset"] == dataset and j["label"] == "DVPA")
            train, test = load_datasets(cfg["base"])
            _, evaluation = baseline_evaluation(cfg["base"], test)
            input_records[dataset] = dict(train=dataset_digest(train), test=dataset_digest(test), evaluation=evaluation)
            for seed in seeds:
                dc = next(j["config"] for j in records if j["dataset"] == dataset and j["seed"] == seed and j["label"] == "DVPA")
                if not smoke:
                    verify_public_validation(dc)
                    verify_forecast_validation(dc)
                parts, ids = partition_and_groups(dc["base"], train.tensors[1])
                torch.manual_seed(seed)
                for _ in range(dc["base"]["clients"]):
                    source_model()(dataset, dc["base"]["batch_size"])
                initial = source_model()(dataset, dc["base"]["batch_size"]).state_dict()
                expected[dataset, seed] = dict(parts=parts, group_ids=ids.tolist(),
                    split_checksum=split_checksum(parts), initial_tensor_sha256=tensor_digest(initial))
                anchors[dataset, seed] = public_preflight(dc).orders
            del train, test, initial
        _json(out / "formal_inputs.json", dict(datasets=input_records, jobs=records,
            expected_identity={f"{d}_{s}": v for (d, s), v in expected.items()},
            device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
        for job in records:
            state.update(status="running", current_job=job["name"], current_job_started_unix=time.time())
            _json(out / "suite_state.json", state)
            report(out, state)
            print("START " + job["name"], flush=True)
            start = time.time()
            with (out / (job["name"] + ".stdout.log")).open("w", encoding="utf-8", buffering=1) as stdout, (out / (job["name"] + ".stderr.log")).open("w", encoding="utf-8", buffering=1) as stderr:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    folder = out / job["name"]
                    # Controller ablations use the full DVPA runner and its
                    # adaptive privacy auditor; they differ only in the
                    # controller mode locked in their DVPA config.
                    is_dvpa = job["label"] in {"DVPA", "STATIC", "FROZEN_INFO"}
                    if is_dvpa:
                        summary = run_dvpa(job["config"], folder)
                        checked = audit_dvpa(folder)
                    else:
                        summary = run_baseline(job["config"], folder,
                            ROOT / ".private_diagnostics/dvpa_datasets_progress.json")
                        checked = audit_baseline(folder)
                    identity = verify_identity(folder, expected[job["dataset"], job["seed"]])
                    base = job["config"]["base"] if is_dvpa else job["config"]
                    _, test = load_datasets(base)
                    test, _ = baseline_evaluation(base, test)
                    model = source_model()(job["dataset"], base["batch_size"]).to("cuda:0" if torch.cuda.is_available() else "cpu")
                    model.load_state_dict(torch.load(folder / "model.pt", weights_only=True, map_location=next(model.parameters()).device))
                    loss, accuracy = evaluate(model, test, next(model.parameters()).device)
                    if not np.isclose(accuracy, summary["final_test_accuracy"], rtol=0, atol=1e-9) or summary["evaluation_examples"] != len(test):
                        raise ValueError("Independent final evaluation differs from training output")
                    evaluation = dict(examples=len(test), accuracy=accuracy, cross_entropy=loss)
                    _json(folder / "evaluation_recheck.json", evaluation)
                    _json(folder / "audit.json", checked)
                    privacy = normalized_privacy(
                        folder, "DVPA" if is_dvpa else job["label"],
                        anchors[job["dataset"], job["seed"]]
                    )
                    for g, p in enumerate(privacy):
                        if p["epsilon_guarantee"] > base["epsilons"][g] + 1e-8:
                            raise ValueError("Privacy cap exceeded")
                    del test, model
            state["identity_checks"].append(dict(name=job["name"], **identity))
            state["completed"].append(dict(**{k: job[k] for k in ("name", "dataset", "seed", "label")},
                summary=summary, audit=checked, group_privacy=privacy,
                evaluation_recheck=evaluation, elapsed_seconds=time.time() - start))
            _json(out / "suite_state.json", state)
            report(out, state)
            print(f"DONE {job['name']} accuracy={accuracy:.4f}% audit={checked['status']}", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        state.update(status="completed", current_job=None, completed_unix=time.time())
        _json(out / "suite_state.json", state)
        report(out, state)
        return dict(status="completed", completed_runs=len(records), report=str(out / "report_zh.md"))
    except BaseException as error:
        state.update(status="failed", failure_type=type(error).__name__, failure_message=str(error), failed_unix=time.time())
        _json(out / "suite_state.json", state)
        report(out, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.test:
        from scripts.run_dvpa_suite import test_gate
        test_gate()
    if args.output:
        result = run_suite(args.output, seeds=(42,) if args.smoke else SEEDS, smoke=args.smoke)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if not (args.prepare or args.test or args.output):
        parser.error("Use --prepare, --test, or --output")


if __name__ == "__main__":
    main()
