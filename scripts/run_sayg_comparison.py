"""Seed42 comparison of actual author spend-as-you-go and matched BR-DP.

All paths/writes stay inside BR_DP. Test gate and exact configs are checked
before training. Public planning is independent of accuracy and private noise.
"""

import argparse
import csv
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
from src.layer_registry import LayerRegistry
from src.upstream_runner import source_model

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "src/sayg_protocol.py",
    "src/brdp_source_protocol.py",
    "src/brdp_group_accountant.py",
    "src/brdp_source_runner.py",
    "src/brdp_source_audit.py",
    "tests/test_sayg_comparison.py",
    "scripts/run_sayg_comparison.py",
    "configs/sayg_source_seed42.yaml",
    "configs/sayg_matched_brdp_seed42.yaml",
}


def inside(relative):
    p = (ROOT / relative).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("All comparison files must stay inside BR_DP")
    return p


def verify_gate(relative="reports/sayg_comparison/test_gate.json"):
    gate = json.loads(inside(relative).read_text(encoding="utf-8"))
    if gate.get("status") != "passed" or gate.get("tests_passed", 0) < 85:
        raise RuntimeError("Passed regression and spend-as-you-go tests required")
    if not REQUIRED <= set(gate["source_sha256"]):
        raise RuntimeError("Incomplete spend-as-you-go test gate")
    for name, digest in gate["source_sha256"].items():
        if hashlib.sha256(inside(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Tested source/config changed; rerun tests")
    return gate


def prepared_jobs():
    configs = [
        validate_config(
            yaml.safe_load(inside(f"configs/{n}").read_text(encoding="utf-8"))
        )
        for n in ("sayg_source_seed42.yaml", "sayg_matched_brdp_seed42.yaml")
    ]
    expected = dict(
        seed=42,
        dataset="fmnist",
        clients=100,
        rounds=25,
        local_epochs=30,
        batch_size=125,
        lr=0.001,
        rate=0.9,
        clip=250.0,
        epsilons=[10.0, 20.0, 30.0],
        delta=1e-5,
        group_proportions=[0.34, 0.43, 0.23],
        transition_round=13,
        q_save=[0.5, 0.6, 0.7],
        q_spend=0.9,
    )
    for cfg in configs:
        if (
            any(cfg[k] != v for k, v in expected.items())
            or cfg.get("dataset_mode", "official") != "official"
        ):
            raise ValueError(
                "Formal comparison must match screenshot settings, seed42, full data"
            )
    if (
        configs[0]["method"] != "SPEND_AS_YOU_GO"
        or configs[1]["method"] != "JOINT"
        or configs[1]["mechanism"] != "brdp"
    ):
        raise ValueError("Wrong baseline or BR-DP method")
    different = {
        k
        for k in set(configs[0]) | set(configs[1])
        if configs[0].get(k) != configs[1].get(k)
    }
    if not different <= {"method", "mechanism", "recycle_prob", "calibration_epsilons"}:
        raise ValueError("Training/common privacy configurations differ")
    dims = LayerRegistry(source_model()("fmnist", 125)).dimensions
    public = [preflight(cfg, dims) for cfg in configs]
    np.testing.assert_allclose(
        configs[1]["calibration_epsilons"],
        public[0]["final_epsilon"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        public[1]["final_epsilon"], public[0]["final_epsilon"], rtol=1.1e-6, atol=1e-8
    )
    return [
        dict(name=name, config=cfg, actual_target=p["final_epsilon"])
        for name, cfg, p in zip(
            ("sayg_source_seed42", "brdp_joint_matched_seed42"),
            configs,
            public,
            strict=True,
        )
    ]


def summarize(out, state):
    names = [c["name"] for c in state["completed"]]
    partitions = [
        json.loads((out / n / "partition.json").read_text(encoding="utf-8"))
        for n in names
    ]
    if partitions[0] != partitions[1]:
        raise ValueError("Frozen partitions/groups differ")
    models = [
        torch.load(out / n / "initial_model.pt", map_location="cpu", weights_only=True)
        for n in names
    ]
    if models[0].keys() != models[1].keys() or any(
        not torch.equal(models[0][k], models[1][k]) for k in models[0]
    ):
        raise ValueError("Initial model tensors differ")
    rows, privacy, rdp = [], [], None
    for c in state["completed"]:
        s = c["summary"]
        row = dict(
            method=c["name"],
            seed=42,
            final_test_accuracy=s["final_test_accuracy"],
            best_test_accuracy=s["best_test_accuracy"],
            non_gaussian_layer_releases=s["non_gaussian_layer_releases"],
            audit=c["audit"]["status"],
        )
        for g in s["group_privacy"]:
            row[f"epsilon_group_{g['epsilon_target']:g}_actual"] = g[
                "epsilon_actual_path_accounting"
            ]
            privacy.append(dict(method=c["name"], **g))
        rows.append(row)
        with np.load(out / c["name"] / "accountant.npz") as a:
            rdp = a["uniform_rdp"].copy() if rdp is None else rdp + a["uniform_rdp"]
            orders = a["orders"].copy()
    np.testing.assert_allclose(
        [
            r["epsilon_guarantee"]
            for r in state["completed"][0]["summary"]["group_privacy"]
        ],
        [
            r["epsilon_guarantee"]
            for r in state["completed"][1]["summary"]["group_privacy"]
        ],
        rtol=1.1e-6,
        atol=1e-8,
    )
    for filename, data in [
        ("comparison.csv", rows),
        ("comparison_group_privacy.csv", privacy),
    ]:
        with (out / filename).open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    combined = [epsilon_from_rdp(orders, rdp[g], 1e-5)[0] for g in range(3)]
    difference = rows[1]["final_test_accuracy"] - rows[0]["final_test_accuracy"]
    result = dict(
        status="completed",
        rows=rows,
        accuracy_difference_percentage_points=difference,
        same_partition_and_groups=True,
        initial_model_tensors_identical=True,
        privacy_caps=[10, 20, 30],
        actual_privacy_matched=True,
        single_seed_only=True,
        combined_release_epsilon=combined,
        delta=1e-5,
    )
    _json(out / "comparison.json", result)
    lines = [
        "# 原源码spend-as-you-go与BR-DP联合调度：seed42",
        "",
        "两组都完成25轮并通过独立账本/发布历史审计。原项目和trynew未修改。",
        "",
        "共同底座：官方FMNIST60k/10k、作者Dirichlet alpha0.1、相同分区与初始模型、",
        "100客户端、30本地epochs、batch125、lr0.001、持久SGD/momentum和每epoch余弦时钟。",
        "共同隐私约束：固定客户端槽位贡献加入/移除，可信服务器仅发布DP聚合；三组10/20/30，delta1e-5。",
        "",
        "| 方法 | 最终准确率 | 最高准确率 | 三组实际会计上界 | 审计 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        eps = [row[f"epsilon_group_{g}_actual"] for g in [10, 20, 30]]
        lines.append(
            f"| {row['method']} | {row['final_test_accuracy']:.2f}% | {row['best_test_accuracy']:.2f}% | "
            + ", ".join(f"{v:.8f}" for v in eps)
            + f" | {row['audit']} |"
        )
    lines += [
        "",
        f"BR-DP最终准确率差值：{difference:+.2f}个百分点。只有seed42，不声称稳定提升或显著性。",
        "",
        "原方法执行作者Client的epsilon余量递推和epochs=剩余轮数的Opacus校准语义；",
        "每轮重新生成组半径与全局sigma，保留第13轮采样切换。",
        "修正联合裁剪，并使用独立的实际发布账本核验，属于原源码方法在共同协议下的适配，不冒充论文表格逐位复现。",
        "",
        "原方法未完全耗尽名义预算。主对比中的BR-DP按原方法实际会计上界标定，未增加其预算；",
        "原先满上限BR-DP(72.19%)为另一预算设置，只作补充参考，不计作本表提升。",
        "",
        "轨迹诊断与全过程包络保证分别记录；本批无回退时二者一致。",
        "同时发布本批两次训练的组合epsilon："
        + ", ".join(f"{v:.8f}" for v in combined)
        + "，delta1e-5。",
        "此组合不包括此前已发布实验；多批联合发布需要进一步累加RDP。",
        "",
        "各组逐轮记录group_privacy.csv；100客户端记录client_privacy.csv；601阶向量accountant.npz；",
        "实际噪声尺度layers.csv；原递推参数轨迹privacy_preflight.json中的schedule_metadata。",
        "",
    ]
    (out / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def run_suite(relative):
    gate = verify_gate()
    jobs = prepared_jobs()
    train, test = load_datasets(jobs[0]["config"])
    parts, ids = partition_and_groups(jobs[0]["config"], train.tensors[1])
    out = inside(relative)
    out.mkdir(parents=True, exist_ok=False)
    _json(out / "test_gate.json", gate)
    _json(
        out / "formal_inputs.json",
        dict(
            train_samples=len(train),
            test_samples=len(test),
            group_counts=np.bincount(ids).tolist(),
            client_counts=len(parts),
            jobs=jobs,
            privacy_matching="BR-DP calibrated to original method actual bounds",
        ),
    )
    state = dict(
        status="running", started_unix=time.time(), current_job=None, completed=[]
    )
    _json(out / "suite_state.json", state)
    try:
        for job in jobs:
            state["current_job"] = job["name"]
            _json(out / "suite_state.json", state)
            summary = run(
                job["config"],
                out / job["name"],
                inside(".private_diagnostics/sayg_comparison_progress.json"),
            )
            checked = audit(out / job["name"])
            _json(out / job["name"] / "audit.json", checked)
            state["completed"].append(
                dict(name=job["name"], summary=summary, audit=checked)
            )
            _json(out / "suite_state.json", state)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        comparison = summarize(out, state)
        state.update(status="completed", current_job=None, comparison=comparison)
        _json(out / "suite_state.json", state)
        return state
    except Exception as exc:
        state.update(
            status="failed", failure_type=type(exc).__name__, failure_message=str(exc)
        )
        _json(out / "suite_state.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/sayg_vs_brdp_seed42_20261027")
    args = parser.parse_args()
    print(json.dumps(run_suite(args.output), indent=2))
