"""Nine fresh runs: paper IDP, paper spend-as-you-go, full BR-DP; seeds42-44."""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import shutil
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
SEEDS = (42, 43, 44)
METHODS = ("IDP_FedAvg", "Time_Adaptive", "BRDP_JOINT")
METHOD_TITLES = {
    "IDP_FedAvg": "论文IDP-FedAvg适配版",
    "Time_Adaptive": "论文Time-Adaptive（spend-as-you-go）适配版",
    "BRDP_JOINT": "BR-DP时空联合",
}
CONFIG_METHODS = {
    "IDP_FedAvg": ("IDP_FEDAVG", "gaussian"),
    "Time_Adaptive": ("SPEND_AS_YOU_GO", "gaussian"),
    "BRDP_JOINT": ("JOINT", "brdp"),
}
REQUIRED = {
    "src/sayg_protocol.py",
    "src/brdp_source_protocol.py",
    "src/brdp_source_runner.py",
    "src/brdp_source_audit.py",
    "src/brdp_group_accountant.py",
    "scripts/run_three_methods_seeds.py",
    "tests/test_three_methods_seeds.py",
    "configs/three_method_seeds/manifest.json",
}


def inside(relative):
    p = (ROOT / relative).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("All experiment paths must stay inside BR_DP")
    return p


def verify_gate(relative="reports/three_method_seeds/test_gate.json"):
    gate = json.loads(inside(relative).read_text(encoding="utf-8"))
    if gate.get("status") != "passed" or gate.get("tests_passed", 0) < 96:
        raise RuntimeError("Passed regression/IDP/three-seed tests required")
    expected = REQUIRED | {
        f"configs/three_method_seeds/{m}_seed{s}.yaml" for s in SEEDS for m in METHODS
    }
    if not expected <= set(gate["source_sha256"]):
        raise RuntimeError("Incomplete tested manifest")
    for p, d in gate["source_sha256"].items():
        if hashlib.sha256(inside(p).read_bytes()).hexdigest() != d:
            raise RuntimeError("Tested code/config changed")
    return gate


def jobs():
    manifest = json.loads(
        inside("configs/three_method_seeds/manifest.json").read_text(encoding="utf-8")
    )
    if manifest["seeds"] != list(SEEDS) or len(manifest["jobs"]) != 9:
        raise ValueError("Exactly seeds42/43/44 and nine jobs required")
    records = []
    expected = dict(
        dataset="fmnist",
        clients=100,
        rounds=25,
        local_epochs=30,
        batch_size=125,
        lr=0.001,
        rate=0.9,
        clip=250.0,
        delta=1e-5,
        epsilons=[10.0, 20.0, 30.0],
        group_proportions=[0.34, 0.43, 0.23],
        q_save=[0.5, 0.6, 0.7],
        q_spend=0.9,
        transition_round=13,
        warmup_rounds=3,
        tau=1.0,
        layer_decay={"conv1": 0.15, "conv2": 0.1, "fc1": 0.05, "fc": 0.01},
    )
    for record, (seed, label) in zip(
        manifest["jobs"], [(s, m) for s in SEEDS for m in METHODS], strict=True
    ):
        if (
            record["seed"] != seed
            or record["label"] != label
            or record["name"] != f"{label}_seed{seed}"
        ):
            raise ValueError("Invalid/duplicate job ordering")
        config_path = f"configs/three_method_seeds/{label}_seed{seed}.yaml"
        if record["config_path"] != config_path:
            raise ValueError("Invalid config path")
        cfg = validate_config(
            yaml.safe_load(inside(config_path).read_text(encoding="utf-8"))
        )
        if (
            cfg["seed"] != seed
            or any(cfg[k] != v for k, v in expected.items())
            or cfg.get("dataset_mode", "official") != "official"
            or "calibration_epsilons" in cfg
        ):
            raise ValueError("Common-cap full-method protocol changed")
        if (cfg["method"], cfg["mechanism"]) != CONFIG_METHODS[label] or cfg[
            "recycle_prob"
        ] != (0.005 if label == "BRDP_JOINT" else 0.0):
            raise ValueError("Wrong method/noise")
        records.append(dict(**record, config=cfg))
    return records


def stats(rows):
    """Sample SD (ddof=1), plus paired per-seed accuracy differences."""
    keys = [(r["seed"], r["label"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate method/seed result")
    means = []
    for label in METHODS:
        selected = [r for r in rows if r["label"] == label]
        if not selected:
            continue
        item = dict(label=label, n=len(selected))
        for metric in ["final_test_accuracy", "best_test_accuracy"]:
            values = np.array([r[metric] for r in selected], dtype=float)
            item[metric + "_mean"] = float(values.mean())
            item[metric + "_sample_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else None
            )
        means.append(item)
    paired = []
    index = {(r["seed"], r["label"]): r for r in rows}
    for seed in SEEDS:
        if not all((seed, label) in index for label in METHODS):
            continue
        ours = index[seed, "BRDP_JOINT"]
        paired.append(
            dict(
                seed=seed,
                brdp_minus_idp_final_pp=ours["final_test_accuracy"]
                - index[seed, "IDP_FedAvg"]["final_test_accuracy"],
                brdp_minus_sayg_final_pp=ours["final_test_accuracy"]
                - index[seed, "Time_Adaptive"]["final_test_accuracy"],
            )
        )
    summary = {}
    for field in ["brdp_minus_idp_final_pp", "brdp_minus_sayg_final_pp"]:
        values = [p[field] for p in paired]
        summary[field] = dict(
            n=len(values),
            mean=float(np.mean(values)) if values else None,
            sample_std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
            positive_seeds=sum(v > 0 for v in values),
        )
    return dict(means=means, paired=paired, paired_summary=summary)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def verify_seed(out, completed, seed):
    records = [c for c in completed if c["seed"] == seed]
    if len(records) != 3:
        return None
    folders = [out / c["name"] for c in records]
    partitions = [
        json.loads((p / "partition.json").read_text(encoding="utf-8")) for p in folders
    ]
    if any(p != partitions[0] for p in partitions[1:]):
        raise ValueError("Within-seed partition/privacy groups differ")
    initial = [
        torch.load(p / "initial_model.pt", weights_only=True, map_location="cpu")
        for p in folders
    ]
    if any(
        v.keys() != initial[0].keys()
        or any(not torch.equal(v[k], initial[0][k]) for k in v)
        for v in initial[1:]
    ):
        raise ValueError("Within-seed initial models differ")
    total = None
    for p in folders:
        with np.load(p / "accountant.npz") as a:
            total = (
                a["uniform_rdp"].copy() if total is None else total + a["uniform_rdp"]
            )
            orders = a["orders"].copy()
    return dict(
        seed=seed,
        initial_models_identical=True,
        partitions_and_groups_identical=True,
        three_run_composed_epsilon=[
            epsilon_from_rdp(orders, total[g], 1e-5)[0] for g in range(3)
        ],
        delta=1e-5,
        scope="this seed's three fresh runs only; excludes previous experiments and other seeds",
    )


def report(out, state):
    rows = []
    privacy = []
    for c in state["completed"]:
        s = c["summary"]
        r = dict(
            seed=c["seed"],
            label=c["label"],
            final_test_accuracy=s["final_test_accuracy"],
            best_test_accuracy=s["best_test_accuracy"],
            non_gaussian_layer_releases=s["non_gaussian_layer_releases"],
            audit=c["audit"]["status"],
        )
        for g in s["group_privacy"]:
            r[f"epsilon_{g['epsilon_target']:g}_actual"] = g[
                "epsilon_actual_path_accounting"
            ]
            privacy.append(dict(seed=c["seed"], label=c["label"], **g))
        rows.append(r)
    statistical = stats(rows)
    write_csv(out / "results_by_seed.csv", rows)
    write_csv(out / "group_privacy.csv", privacy)
    write_csv(out / "mean_std.csv", statistical["means"])
    write_csv(out / "paired_improvements.csv", statistical["paired"])
    result = dict(
        status=state["status"],
        completed_runs=len(rows),
        expected_runs=9,
        results=rows,
        statistics=statistical,
        privacy_policy="common caps10/20/30; actual bounds reported, not equal consumption",
        within_seed_verification=state.get("within_seed_verification", []),
    )
    _json(out / "comparison.json", result)
    lines = [
        "# 三种完整方法 × seeds42/43/44",
        "",
        f"队列状态：{state['status']}；完成{len(rows)}/9次。",
        "",
        "共同协议：FMNIST官方60k/10k、作者Dirichlet alpha0.1、100客户端、25轮、每轮30epochs、batch125、",
        "lr0.001、持久SGD/momentum0.9/weight_decay5e-4、每本地epoch余弦调度；共同Scheme2更新单位。",
        "同一种子内样本分区、隐私组与初始模型完全一致；不同种子独立划分和初始化。所有方法每轮训练全部客户端。",
        "",
        "**本批比较共同epsilon上限10/20/30、delta1e-5下的完整方法，不是等实际消耗实验。**",
        "IDP采用恒定q0.9和一次校准的局部sigma；Time-Adaptive明确指原spend-as-you-go递推；",
        "BR-DP为当前新时间/层级日程和方向软筛选，完整使用其预定预算校准。三组实际费用逐项报告。",
        "",
        "| Seed | 方法 | 最终准确率 | 最高准确率 | 三组实际epsilon会计上界 | 审计 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        eps = ", ".join(f"{r[f'epsilon_{g}_actual']:.6f}" for g in [10, 20, 30])
        lines.append(
            f"| {r['seed']} | {METHOD_TITLES[r['label']]} | {r['final_test_accuracy']:.2f}% | {r['best_test_accuracy']:.2f}% | {eps} | {r['audit']} |"
        )
    lines += [
        "",
        "## 均值与样本标准差",
        "",
        "样本标准差采用ddof=1；完整结果每个方法n=3。未完成时统计标为部分结果。",
        "",
        "| 方法 | n | 最终准确率均值±标准差 | 最高准确率均值±标准差 |",
        "| --- | --- | --- | --- |",
    ]
    for r in statistical["means"]:

        def cell(metric):
            sd = r[metric + "_sample_std"]
            return (
                f"{r[metric + '_mean']:.2f}% ± {sd:.2f}"
                if sd is not None
                else f"{r[metric + '_mean']:.2f}%（标准差待更多种子）"
            )

        lines.append(
            f"| {METHOD_TITLES[r['label']]} | {r['n']} | {cell('final_test_accuracy')} | {cell('best_test_accuracy')} |"
        )
    lines += [
        "",
        "## 每个种子的完整方法差值",
        "",
        "| Seed | BR-DP−IDP最终准确率 | BR-DP−spend-as-you-go最终准确率 |",
        "| --- | --- | --- |",
    ]
    for r in statistical["paired"]:
        lines.append(
            f"| {r['seed']} | {r['brdp_minus_idp_final_pp']:+.2f}个百分点 | {r['brdp_minus_sayg_final_pp']:+.2f}个百分点 |"
        )
    if len(rows) == 9:
        lines += [""]
        for key, title in [
            ("brdp_minus_idp_final_pp", "IDP"),
            ("brdp_minus_sayg_final_pp", "spend-as-you-go"),
        ]:
            v = statistical["paired_summary"][key]
            lines.append(
                f"相对{title}：最终差值均值{v['mean']:+.2f}个百分点，样本标准差{v['sample_std']:.2f}；{v['positive_seeds']}/3个种子为正。"
            )
    lines += [
        "",
        "## 解释边界",
        "",
        "完整方法的准确率差异可能包含预算利用率、调度、裁剪与噪声分布的共同影响，不能单独归因于BR-DP噪声。",
        "三种方法使用相同有效隐私上限，但各自会计的松紧程度不同；报告值为保证上界，不是真实泄漏的精确测量。",
        "作者IDP/SAYG校准均保留其epochs接口语义，并按同601阶orders保守检查；不冒充原论文表格逐位复现。",
        "共同训练适配统一为现有Scheme2更新单位；原Scheme1源码的delta/(lr*rate)差异未保留，避免训练底座差异混入预算方法比较。",
        "两基线采用正确的高斯联合抽样会计，BR-DP采用非高斯会计。weight/bias联合裁剪、独立shuffle流、干净测试划分三组一致。",
        "可信服务器仅发布DP聚合、固定槽位客户端贡献加入/移除；不包含原客户端分担噪声的额外服务器侧保护。",
        "每个种子的三次发布另算组合保证。不同种子客户端分区/组归属改变，不直接跨种子相加组RDP来声称九次联合保证。",
        "全部数据是公开基准；本批及历史多批结果的整体联合隐私保证未作认证。三种子结果描述均值和变动，不自动等同统计显著性或普遍稳定提升。",
        "",
    ]
    (out / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def run_suite(relative):
    gate = verify_gate()
    records = jobs()
    out = inside(relative)
    out.mkdir(parents=True, exist_ok=False)
    state = dict(
        status="preparing",
        started_unix=time.time(),
        current_job=None,
        completed=[],
        within_seed_verification=[],
    )
    _json(out / "suite_state.json", state)
    _json(out / "test_gate.json", gate)
    shutil.copyfile(Path(__file__), out / "driver_snapshot.py")
    try:
        train, test = load_datasets(records[0]["config"])
        dims = LayerRegistry(source_model()("fmnist", 125)).dimensions
        plans = []
        splits = {}
        for record in records:
            cfg = record["config"]
            if cfg["seed"] not in splits:
                parts, ids = partition_and_groups(cfg, train.tensors[1])
                splits[cfg["seed"]] = dict(
                    split_checksum=split_checksum(parts),
                    group_counts=np.bincount(ids).tolist(),
                )
            p = preflight(cfg, dims)
            plans.append(dict(name=record["name"], final_guarantee=p["final_epsilon"]))
        _json(
            out / "formal_inputs.json",
            dict(
                train_samples=len(train),
                test_samples=len(test),
                parameters=sum(dims.values()),
                splits=splits,
                public_privacy_plans=plans,
                jobs=records,
            ),
        )
        state["status"] = "running"
        _json(out / "suite_state.json", state)
        for job in records:
            state["current_job"] = job["name"]
            _json(out / "suite_state.json", state)
            summary = run(
                job["config"],
                out / job["name"],
                inside(".private_diagnostics/three_method_seeds_progress.json"),
            )
            checked = audit(out / job["name"])
            _json(out / job["name"] / "audit.json", checked)
            state["completed"].append(
                dict(
                    name=job["name"],
                    label=job["label"],
                    seed=job["seed"],
                    summary=summary,
                    audit=checked,
                )
            )
            if job["label"] == "BRDP_JOINT":
                state["within_seed_verification"].append(
                    verify_seed(out, state["completed"], job["seed"])
                )
            _json(out / "suite_state.json", state)
            report(out, state)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        state.update(status="completed", current_job=None, completed_unix=time.time())
        report(out, state)
        _json(out / "suite_state.json", state)
        return dict(
            status=state["status"],
            completed=len(state["completed"]),
            report=str(out / "report_zh.md"),
        )
    except Exception as exc:
        state.update(
            status="failed", failure_type=type(exc).__name__, failure_message=str(exc)
        )
        _json(out / "suite_state.json", state)
        report(out, state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="results/three_methods_seeds42_43_44_20261028"
    )
    args = parser.parse_args()
    print(json.dumps(run_suite(args.output), indent=2))
