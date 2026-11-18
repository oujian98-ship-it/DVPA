"""Fresh DVPA/IDP-FedAvg/spend-as-you-go comparison under common training/caps.

Preserves each full method's sampling/clip/noise policy and valid accountant.
This is not an equal-realized-spend or isolated-controller experiment.
"""

import argparse
import contextlib
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch

from scripts.run_dvpa_suite import verify_formal, verify_gate as verify_dvpa_gate
from src.brdp_source_audit import audit as audit_baseline
from src.brdp_source_protocol import load_datasets, partition_and_groups, preflight
from src.brdp_source_runner import _json, run as run_baseline
from src.data import split_checksum
from src.dvpa_audit import audit as audit_dvpa
from src.dvpa_config import DEFAULT_CONFIG, ROOT, load_config, public_preflight, smoke_config
from src.dvpa_runner import inside_output, run as run_dvpa
from src.dvpa_test_public64 import evaluation_data
from src.layer_registry import LayerRegistry
from src.st_runner import write_csv
from src.upstream_runner import source_model

LABELS = ("DVPA", "IDP_FedAvg", "Time_Adaptive")
TITLES = {
    "DVPA": "DVPA（Gaussian）",
    "IDP_FedAvg": "IDP-FedAvg适配版",
    "Time_Adaptive": "spend-as-you-go适配版",
}
TEST_MODULE = "tests.test_dvpa_comparison"


def jobs(seeds, smoke=False, device=None):
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("Unique nonnegative integer seeds required")
    result = []
    for seed in seeds:
        dvpa = load_config(DEFAULT_CONFIG)
        if smoke:
            dvpa = smoke_config(dvpa)
        dvpa["base"]["seed"] = seed
        if device is not None:
            dvpa["base"]["device"] = device
        if not smoke:
            verify_formal(dvpa)
        result.append(
            dict(seed=seed, label="DVPA", name=f"DVPA_seed{seed}", config=dvpa)
        )
        for label, method in [
            ("IDP_FedAvg", "IDP_FEDAVG"),
            ("Time_Adaptive", "SPEND_AS_YOU_GO"),
        ]:
            # Exactly the same data/model/training fields, native budget policy.
            base = copy.deepcopy(dvpa["base"])
            base.update(method=method, mechanism="gaussian", recycle_prob=0.0)
            result.append(
                dict(seed=seed, label=label, name=f"{label}_seed{seed}", config=base)
            )
    return result


def comparison_manifest():
    gate = verify_dvpa_gate()
    manifest = dict(gate["source_sha256"])
    for name in ("scripts/run_dvpa_comparison.py", "tests/test_dvpa_comparison.py"):
        manifest[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return manifest


def test_gate():
    directory = ROOT / "reports/dvpa_comparison"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = comparison_manifest()
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_dvpa_comparison.py",
            "--junitxml",
            str(directory / "pytest.xml"),
            "--basetemp",
            str(directory / "pytest_workspace"),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (directory / "pytest.log").write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout, flush=True)
    if proc.returncode:
        raise RuntimeError("DVPA comparison integration tests failed")
    cases = list(ET.parse(directory / "pytest.xml").getroot().iter("testcase"))
    if (
        len(cases) < 5
        or any(
            x.find("failure") is not None
            or x.find("error") is not None
            or x.find("skipped") is not None
            for x in cases
        )
        or {x.attrib["classname"] for x in cases} != {TEST_MODULE}
        or comparison_manifest() != manifest
    ):
        raise RuntimeError("Incomplete/changed comparison gate")
    gate = dict(
        status="passed",
        base_tests_passed=verify_dvpa_gate()["tests_passed"],
        integration_tests_passed=len(cases),
        source_sha256=manifest,
    )
    _json(directory / "test_gate.json", gate)
    return gate


def verify_gate():
    gate = json.loads(
        (ROOT / "reports/dvpa_comparison/test_gate.json").read_text(encoding="utf-8")
    )
    if (
        gate.get("status") != "passed"
        or gate.get("integration_tests_passed", 0) < 5
        or gate.get("source_sha256") != comparison_manifest()
    ):
        raise RuntimeError("Tested comparison sources changed; rerun --test")
    return gate


def tensor_digest(state):
    h = hashlib.sha256()
    for name, value in state.items():
        x = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((str(x.dtype), tuple(x.shape))).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def dataset_digest(dataset):
    return tensor_digest({str(i): x for i, x in enumerate(dataset.tensors)})


def verify_identity(directory, expected):
    part = json.loads((directory / "partition.json").read_text(encoding="utf-8"))
    if any(part[k] != expected[k] for k in ("parts", "group_ids", "split_checksum")):
        raise ValueError("Within-seed partition/group mismatch")
    initial = torch.load(
        directory / "initial_model.pt", map_location="cpu", weights_only=True
    )
    if tensor_digest(initial) != expected["initial_tensor_sha256"]:
        raise ValueError("Within-seed initial model mismatch")
    return dict(initial_models_identical=True, partitions_and_groups_identical=True)


def normalized_privacy(directory, label, anchors):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    result = []
    if label == "DVPA":
        for row in summary["group_privacy"]:
            result.append(
                dict(
                    group=row["group"],
                    epsilon_guarantee=row["epsilon_guarantee"],
                    epsilon_path_diagnostic=row["epsilon_path_diagnostic"],
                    path_is_global_guarantee=False,
                    diagnostic_alpha=row["alpha"],
                    rdp_path_at_diagnostic_alpha=row["rdp_spent"],
                    accountant="fixed_order_adaptive_filter",
                )
            )
    else:
        with np.load(directory / "accountant.npz") as data:
            orders = data["orders"].copy()
            actual = data["actual_rdp"].copy()
        for g, row in enumerate(summary["group_privacy"]):
            index = np.flatnonzero(orders == anchors[g])
            if len(index) != 1:
                raise ValueError("Missing common diagnostic order")
            result.append(
                dict(
                    group=g,
                    epsilon_guarantee=row["epsilon_guarantee"],
                    epsilon_path_diagnostic=row["epsilon_actual_path_accounting"],
                    path_is_global_guarantee=False,
                    diagnostic_alpha=int(anchors[g]),
                    rdp_path_at_diagnostic_alpha=float(actual[g, index[0]]),
                    accountant="precommitted_multiorder_gaussian_envelope",
                )
            )
    return result


def report(out, state):
    rows = []
    privacy = []
    for done in state["completed"]:
        s = done["summary"]
        row = dict(
            seed=done["seed"],
            label=done["label"],
            final_test_accuracy=s["final_test_accuracy"],
            best_test_accuracy=s["best_test_accuracy"],
            audit=done["audit"]["status"],
        )
        for p in done["group_privacy"]:
            g = p["group"]
            row[f"group_{g}_epsilon_guarantee"] = p["epsilon_guarantee"]
            row[f"group_{g}_epsilon_path_diagnostic"] = p["epsilon_path_diagnostic"]
            privacy.append(dict(seed=done["seed"], label=done["label"], **p))
        rows.append(row)
    if rows:
        write_csv(out / "results_by_seed.csv", rows)
        write_csv(out / "group_privacy.csv", privacy)
    index = {(r["seed"], r["label"]): r for r in rows}
    paired = []
    for seed in state["seeds"]:
        if all((seed, label) in index for label in LABELS):
            paired.append(
                dict(
                    seed=seed,
                    dvpa_minus_idp_final_pp=index[seed, "DVPA"]["final_test_accuracy"]
                    - index[seed, "IDP_FedAvg"]["final_test_accuracy"],
                    dvpa_minus_sayg_final_pp=index[seed, "DVPA"]["final_test_accuracy"]
                    - index[seed, "Time_Adaptive"]["final_test_accuracy"],
                )
            )
    if paired:
        write_csv(out / "paired_improvements.csv", paired)
    means = []
    for label in LABELS:
        selected = [r for r in rows if r["label"] == label]
        if selected:
            values = np.array([r["final_test_accuracy"] for r in selected])
            means.append(
                dict(
                    label=label,
                    n=len(values),
                    final_accuracy_mean=float(values.mean()),
                    final_accuracy_sample_std=float(values.std(ddof=1))
                    if len(values) > 1
                    else None,
                )
            )
    if means:
        write_csv(out / "mean_std.csv", means)
    _json(
        out / "comparison.json",
        dict(
            status=state["status"],
            seeds=state["seeds"],
            results=rows,
            statistics=means,
            paired=paired,
            expected_runs=3 * len(state["seeds"]),
            identity_checks=state["identity_checks"],
            privacy_policy="common per-run caps; native method policies; not equal consumption",
        ),
    )
    lines = [
        "# DVPA、IDP-FedAvg与spend-as-you-go统一协议比较",
        "",
        f"状态：{state['status']}；完成{len(rows)}/{3 * len(state['seeds'])}。",
        "",
        "同一种子共用数据/分区/组归属、作者CNN初始化、持久训练设置和组上限10/20/30、delta1e-5。",
        "IDP保留恒定q=.9；SAYG保留作者预算递推和分阶段采样；DVPA采用历史驱动精度与前置过滤。",
        "各方法的裁剪半径与噪声计划保留自身定义；这是完整方法比较，不是控制器净收益实验。",
        "",
        "| Seed | 方法 | 最终准确率 | 最高准确率 | 三组认证epsilon上界 | 审计 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        eps = " / ".join(f"{r[f'group_{g}_epsilon_guarantee']:.6f}" for g in range(3))
        lines.append(
            f"| {r['seed']} | {TITLES[r['label']]} | {r['final_test_accuracy']:.2f}% | {r['best_test_accuracy']:.2f}% | {eps} | {r['audit']} |"
        )
    lines += [
        "",
        "实际参数对应的累计RDP费用及路径诊断另列group_privacy.csv。DVPA的路径转换不能当作更小的全局保证。",
        "每种方法使用适用的有效会计；会计紧度、预算使用、采样和裁剪差异均可能影响效果。",
        "seed控制公开划分/初始化；真实DP采样与噪声由独立OS秘密种子产生。",
        "",
    ]
    for r in paired:
        lines.append(
            f"seed{r['seed']}：DVPA−IDP {r['dvpa_minus_idp_final_pp']:+.2f}个百分点；DVPA−SAYG {r['dvpa_minus_sayg_final_pp']:+.2f}个百分点。"
        )
    if state["smoke"]:
        lines += ["", "本目录为合成工程检查，不是正式FMNIST效用证据。"]
    (out / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(output, seeds=(42,), smoke=False, device=None, enforce_gate=True):
    gate = verify_gate() if enforce_gate else None
    records = jobs(list(seeds), smoke, device)
    out = inside_output(output)
    out.mkdir(parents=True, exist_ok=False)
    state = dict(
        status="preparing",
        seeds=list(seeds),
        smoke=smoke,
        current_job=None,
        completed=[],
        identity_checks=[],
        started_unix=time.time(),
    )
    _json(out / "suite_state.json", state)
    if gate:
        _json(out / "test_gate.json", gate)
    # Prevent accidental reuse of historical scalar results: all jobs are fresh.
    (out / "driver_snapshot.py").write_bytes(Path(__file__).read_bytes())
    expected = {}
    anchors = {}
    try:
        torch.set_num_threads(1)
        base = records[0]["config"]["base"]
        train, test = load_datasets(base)
        input_hashes = dict(train=dataset_digest(train), test=dataset_digest(test))
        _, evaluation = evaluation_data(base, test)
        dimensions = LayerRegistry(
            source_model()("fmnist", records[0]["config"]["base"]["batch_size"])
        ).dimensions
        public_plans = []
        for seed in seeds:
            cfg = next(
                j["config"]
                for j in records
                if j["seed"] == seed and j["label"] == "DVPA"
            )
            base = cfg["base"]
            parts, ids = partition_and_groups(base, train.tensors[1])
            torch.manual_seed(seed)
            for _ in range(base["clients"]):
                source_model()("fmnist", base["batch_size"])
            initial = source_model()("fmnist", base["batch_size"]).state_dict()
            expected[seed] = dict(
                parts=parts,
                group_ids=ids.tolist(),
                split_checksum=split_checksum(parts),
                initial_tensor_sha256=tensor_digest(initial),
            )
            p = public_preflight(cfg)
            anchors[seed] = p.orders
        for job in records:
            p = (
                public_preflight(job["config"])
                if job["label"] == "DVPA"
                else preflight(job["config"], dimensions)
            )
            public_plans.append(
                dict(
                    name=job["name"],
                    public_protocol=p.as_dict()
                    if job["label"] == "DVPA"
                    else dict(
                        final_epsilon=p["final_epsilon"],
                        plans=[v.as_dict() for v in p["plans"]],
                    ),
                )
            )
        _json(
            out / "formal_inputs.json",
            dict(
                dataset_scope="synthetic engineering"
                if smoke
                else "official FMNIST with 60,000 training and 9,936 evaluation images",
                dataset_sha256=input_hashes,
                evaluation=evaluation,
                parameters=sum(dimensions.values()),
                expected_identity=expected,
                public_plans=public_plans,
                jobs=records,
                device=torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "cpu",
            ),
        )
        del train, test, initial
        for job in records:
            state.update(status="running", current_job=job["name"])
            _json(out / "suite_state.json", state)
            report(out, state)
            print(f"START {job['name']}", flush=True)
            with (
                (out / f"{job['name']}.stdout.log").open(
                    "w", encoding="utf-8"
                ) as stdout,
                (out / f"{job['name']}.stderr.log").open(
                    "w", encoding="utf-8"
                ) as stderr,
            ):
                with (
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    if job["label"] == "DVPA":
                        summary = run_dvpa(job["config"], out / job["name"])
                        checked = audit_dvpa(out / job["name"])
                    else:
                        summary = run_baseline(
                            job["config"],
                            out / job["name"],
                            ROOT / ".private_diagnostics/dvpa_comparison_progress.json",
                        )
                        checked = audit_baseline(out / job["name"])
            identity = verify_identity(out / job["name"], expected[job["seed"]])
            state["identity_checks"].append(dict(name=job["name"], **identity))
            _json(out / job["name"] / "audit.json", checked)
            privacy = normalized_privacy(
                out / job["name"], job["label"], anchors[job["seed"]]
            )
            state["completed"].append(
                dict(
                    **{k: job[k] for k in ("name", "seed", "label")},
                    summary=summary,
                    audit=checked,
                    group_privacy=privacy,
                )
            )
            _json(out / "suite_state.json", state)
            report(out, state)
            print(
                f"DONE {job['name']} accuracy={summary['final_test_accuracy']:.2f}% audit={checked['status']}",
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        state.update(status="completed", current_job=None, completed_unix=time.time())
        _json(out / "suite_state.json", state)
        report(out, state)
        return dict(
            status="completed",
            completed=len(state["completed"]),
            report=str(out / "report_zh.md"),
        )
    except Exception as error:
        state.update(
            status="failed",
            failure_type=type(error).__name__,
            failure_message=str(error),
        )
        _json(out / "suite_state.json", state)
        report(out, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda:0"])
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        test_gate()
    if args.output:
        print(
            json.dumps(
                run_suite(args.output, args.seeds, args.smoke, args.device), indent=2
            )
        )
    elif not args.test:
        parser.error("--output or --test required")


if __name__ == "__main__":
    main()
