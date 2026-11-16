"""Run public fixed-sensitivity, matched-guarantee mechanism checks, not FL."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml

from src.brdp_net_gain import (
    FixedGeometryAccountant,
    profile_metrics,
    public_mc_case,
    saved_geometry,
    static_geometry,
)
from src.brdp_source_runner import _json
from src.layer_registry import LayerRegistry
from src.upstream_runner import source_model

ROOT = Path(__file__).resolve().parents[1]


def inside(path):
    p = (ROOT / path).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("All outputs must stay inside BR_DP")
    return p


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def geometry_dict(p):
    return dict(
        name=p.name,
        q=p.q.tolist(),
        radii=p.radii.tolist(),
        scales=p.scales.tolist(),
        clip_weights=p.clip_weights.tolist(),
        base_stds=p.base_stds.tolist(),
        dimensions=p.dimensions,
        targets=p.targets.tolist(),
        warmup=p.warmup,
    )


def validate_config(c):
    required = {
        "public_seed",
        "delta",
        "recycle_probs",
        "taus",
        "sampling_rates",
        "round_counts",
        "epsilon_values",
        "block_counts",
        "synthetic_block_dimension",
        "monte_carlo_samples",
        "saved_profile",
    }
    if set(c) != required:
        raise ValueError("Unknown/missing benchmark configuration")
    if (
        not 0 < c["delta"] < 1
        or c["public_seed"] != 42
        or c["monte_carlo_samples"] < 10000
    ):
        raise ValueError("Invalid delta/seed/sample count")
    if any(not 0 < r < 1 for r in c["recycle_probs"]) or any(
        not 0.05 <= t <= 8 for t in c["taus"]
    ):
        raise ValueError("Invalid soft shape")
    if (
        any(not 0 < q <= 1 for q in c["sampling_rates"])
        or any(t < 1 for t in c["round_counts"])
        or any(e <= 0 or e > 100 for e in c["epsilon_values"])
    ):
        raise ValueError("Invalid scenario grid")
    return c


def tested_gate():
    p = inside("reports/brdp_net_gain/test_gate.json")
    g = json.loads(p.read_text(encoding="utf-8"))
    if g.get("status") != "passed" or g.get("tests_passed", 0) < 20:
        raise RuntimeError("Passed mechanism/regression tests required")
    for name, digest in g["source_sha256"].items():
        if hashlib.sha256(inside(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Tested benchmark code/config changed")
    return g


def assess(p, r, tau, gaussian, reference, account, delta):
    base = dict(
        profile=p.name,
        rounds=len(p.q),
        blocks=len(p.dimensions),
        groups=len(p.targets),
        target_caps=p.targets.tolist(),
        recycle_prob=r,
        tau=tau,
        gaussian_multiplier=gaussian["multiplier"],
        gaussian_epsilon=gaussian["epsilon"],
        gaussian_best_alpha=gaussian["best_alpha"],
        gaussian_fractional_fallback_orders=gaussian.get(
            "gaussian_fractional_fallback_orders", []
        ),
    )
    if r == 0:
        fitted = gaussian
    else:
        fitted = account.calibrate(gaussian["epsilon"], r, tau, delta)
    base.update(status=fitted["status"])
    if fitted["status"] != "FEASIBLE":
        base.update(
            bound_floor_epsilon=fitted["bound_floor_epsilon"],
            direction_gain=False,
            tail2_gain=False,
            energy_gain=False,
        )
        return base
    ratio = fitted["multiplier"] / gaussian["multiplier"]
    metrics = profile_metrics(p, ratio, r, tau)
    base.update(
        brdp_multiplier=fitted["multiplier"],
        brdp_epsilon=fitted["epsilon"],
        brdp_best_alpha=fitted["best_alpha"],
        reference_only_gaussian_multiplier=reference["multiplier"],
        gaussian_reference_slack_variance_factor=(
            reference["multiplier"] / gaussian["multiplier"]
        )
        ** 2,
        recycling_over_reference_scale_factor=fitted["multiplier"]
        / reference["multiplier"],
        **metrics,
    )
    base.update(
        direction_gain=bool(r > 0 and metrics["ideal_parallel_mse_ratio"] < 1 - 1e-6),
        tail2_gain=bool(r > 0 and metrics["tail_ratio_at_2_gaussian_std"] < 1 - 1e-6),
        energy_gain=bool(r > 0 and metrics["trajectory_total_energy_ratio"] < 1 - 1e-6),
    )
    if np.any(
        np.array(base["brdp_epsilon"]) > np.array(base["gaussian_epsilon"]) * (1 + 1e-9)
    ):
        raise RuntimeError("BR-DP guarantee weaker than Gaussian reference")
    return base


def run(config, output):
    cfg = validate_config(config)
    gate = tested_gate()
    out = inside(output)
    out.mkdir(parents=True, exist_ok=False)
    _json(out / "test_gate.json", gate)
    (out / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    started = time.time()
    _json(
        out / "state.json",
        dict(
            status="running", phase="fixed_geometry_calibration", started_unix=started
        ),
    )
    dimensions = LayerRegistry(source_model()("fmnist", 125)).dimensions
    file = inside(cfg["saved_profile"])
    saved = json.loads(file.read_text(encoding="utf-8"))
    current = saved_geometry("current_FL_fixed_geometry", saved, dimensions)
    profiles = [current]
    for q in cfg["sampling_rates"]:
        for t in cfg["round_counts"]:
            for eps in cfg["epsilon_values"]:
                for k in cfg["block_counts"]:
                    p = static_geometry(
                        f"public_q{q:g}_T{t}_eps{eps:g}_K{k}",
                        q,
                        t,
                        eps,
                        k,
                        cfg["synthetic_block_dimension"],
                    )
                    profiles.append(p)
    _json(
        out / "geometry_lock.json",
        dict(
            source_preflight_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            current_geometry=geometry_dict(current),
            rule="freeze all geometry; calibrate common noise scale only; each BR-DP group no weaker than Gaussian",
        ),
    )
    rows = []
    geometry_by_name = {p.name: p for p in profiles}
    try:
        for index, p in enumerate(profiles):
            fingerprint = json.dumps(geometry_dict(p), sort_keys=True)
            account = FixedGeometryAccountant(p)
            gaussian = account.calibrate(
                p.targets, 0.0, 1.0, cfg["delta"], mode="gaussian"
            )
            if gaussian["status"] != "FEASIBLE":
                raise RuntimeError("Registered Gaussian baseline infeasible")
            reference = account.calibrate(
                gaussian["epsilon"], 0.0, 1.0, cfg["delta"], mode="reference_bound"
            )
            rows.append(assess(p, 0.0, 1.0, gaussian, reference, account, cfg["delta"]))
            for r in cfg["recycle_probs"]:
                for tau in cfg["taus"]:
                    rows.append(
                        assess(
                            p,
                            float(r),
                            float(tau),
                            gaussian,
                            reference,
                            account,
                            cfg["delta"],
                        )
                    )
            if json.dumps(geometry_dict(p), sort_keys=True) != fingerprint:
                raise RuntimeError("Frozen clipping/sampling geometry changed")
            _json(
                out / "state.json",
                dict(
                    status="running",
                    phase="parameter_grid",
                    completed_profiles=index + 1,
                    total_profiles=len(profiles),
                    last_profile=p.name,
                    candidate_rows=len(rows),
                    elapsed_seconds=time.time() - started,
                ),
            )
            if index == 0:
                write_csv(out / "current_FL.csv", rows)
            print(
                f"NET-GAIN profile={index + 1}/{len(profiles)} rows={len(rows)} {p.name}",
                flush=True,
            )
        write_csv(out / "mechanisms.csv", rows)
        current_rows = [r for r in rows if r["profile"] == current.name]
        write_csv(out / "current_FL.csv", current_rows)
        # Deterministic selection from analytic predictions, never observed MC
        # outcomes or private model accuracy. No subsequent training is tuned.
        selected = []
        selected.append(
            next(
                r
                for r in current_rows
                if r["recycle_prob"] == 0.005 and r["tau"] == 1.0
            )
        )
        feasible = [
            r
            for r in current_rows
            if r["recycle_prob"] > 0 and r["status"] == "FEASIBLE"
        ]
        if feasible:
            selected.append(min(feasible, key=lambda r: r["ideal_parallel_mse_ratio"]))
        successes = [r for r in rows if r.get("direction_gain")]
        if successes:
            selected.extend(
                [
                    min(successes, key=lambda r: r["ideal_parallel_mse_ratio"]),
                    min(
                        successes,
                        key=lambda r: (max(r["target_caps"]), r["rounds"], r["blocks"]),
                    ),
                ]
            )
        tail_only = [
            r for r in rows if r.get("tail2_gain") and not r.get("direction_gain")
        ]
        if tail_only:
            selected.append(
                min(tail_only, key=lambda r: r["tail_ratio_at_2_gaussian_std"])
            )
        mc = []
        seen = set()
        for row in selected:
            key = (row["profile"], row["recycle_prob"], row["tau"])
            if key in seen or row["status"] != "FEASIBLE":
                continue
            seen.add(key)
            p = geometry_by_name[row["profile"]]
            mc.append(
                dict(
                    profile=p.name,
                    recycle_prob=row["recycle_prob"],
                    tau=row["tau"],
                    **public_mc_case(
                        row["gaussian_multiplier"],
                        row["brdp_multiplier"],
                        row["recycle_prob"],
                        row["tau"],
                        p.dimensions[0],
                        cfg["monte_carlo_samples"],
                        cfg["public_seed"] + len(mc),
                    ),
                )
            )
        if any(
            not r["moment_check_passed"]
            or not r["energy_check_passed"]
            or any(not r[f"tail_{k}_check_passed"] for k in [1, 2, 3])
            for r in mc
        ):
            raise RuntimeError(
                "Public Monte Carlo check outside preregistered six-SE tolerance"
            )
        _json(out / "monte_carlo.json", mc)
        summary = dict(
            status="completed",
            scope="public mechanism analysis; no new FL training or private-data queries",
            profiles=len(profiles),
            rows=len(rows),
            non_gaussian_candidates=sum(r["recycle_prob"] > 0 for r in rows),
            current_non_gaussian_direction_gains=sum(
                r.get("direction_gain", False) for r in current_rows
            ),
            current_non_gaussian_tail2_gains=sum(
                r.get("tail2_gain", False) for r in current_rows
            ),
            current_infeasible=sum(r["status"] != "FEASIBLE" for r in current_rows),
            all_direction_gains=len(successes),
            tail2_only_candidates=len(tail_only),
            current_parameter_result=selected[0],
            best_current_feasible=min(
                feasible, key=lambda r: r["ideal_parallel_mse_ratio"]
            )
            if feasible
            else None,
            best_public_direction_gain=min(
                successes, key=lambda r: r["ideal_parallel_mse_ratio"]
            )
            if successes
            else None,
            monte_carlo_cases=len(mc),
            monte_carlo_checks="passed",
            elapsed_seconds=time.time() - started,
            order_count=len(account.orders),
            privacy_bound="existing conservative analytic sampled BR-DP bound; negative results are not impossibility results",
        )
        _json(out / "summary.json", summary)
        make_report(out, summary, rows, mc)
        _json(
            out / "state.json",
            dict(status="completed", elapsed_seconds=time.time() - started),
        )
        return summary
    except Exception as exc:
        _json(
            out / "state.json",
            dict(
                status="failed",
                type=type(exc).__name__,
                message=str(exc),
                elapsed_seconds=time.time() - started,
            ),
        )
        raise


def make_report(out, s, rows, mc):
    cur = s["current_parameter_result"]
    best = s["best_current_feasible"]
    lines = [
        "# BR-DP机制净收益验证",
        "",
        "本报告是公开噪声机制验证，不是新的联邦准确率实验。原项目与trynew只读。",
        "",
        "## 严格固定的条件",
        "",
        "当前25轮配置的客户端采样率、每组逐轮裁剪半径、weight/bias联合裁剪、各层预算权重、聚合缩放全部冻结。",
        "只改变公共噪声尺度和软筛选参数。先按正确Gaussian SGM校准，再要求BR-DP每组保证不弱于该高斯保证。",
        "同一噪声尺度乘数必须满足三组约束，允许BR-DP某些组更强，绝不允许更弱。几何与来源哈希见geometry_lock.json。",
        "",
        "## 当前参数r=0.005、tau=1",
        "",
    ]
    if cur["status"] == "FEASIBLE":
        lines += [
            "| 指标 | BR-DP / Gaussian |",
            "| --- | ---: |",
            f"| 基础噪声标准差 | {cur['noise_std_ratio']:.8f} |",
            f"| 完全对齐方向的噪声MSE（预热后） | {cur['ideal_parallel_mse_ratio']:.8f} |",
            f"| 全25轮受保护轴累计MSE | {cur['trajectory_protected_axes_mse_ratio']:.8f} |",
            f"| 全25轮全模型累计噪声能量 | {cur['trajectory_total_energy_ratio']:.8f} |",
            f"| 超过同一2σ_G绝对阈值的概率（预热后） | {cur['tail_ratio_at_2_gaussian_std']:.8f} |",
            "",
            "Gaussian三组epsilon上界："
            + str(cur["gaussian_epsilon"])
            + "。BR-DP三组上界："
            + str(cur["brdp_epsilon"])
            + "。",
            "",
            f"固定基础sigma时方向方差因子为{cur['fixed_sigma_variance_factor']:.10f}；若要方向MSE净降低，标准差膨胀倍率必须低于{cur['max_std_ratio_for_perfect_direction_gain']:.8f}。",
            f"实际倍率为{cur['noise_std_ratio']:.8f}，因此在当前参数/会计下："
            + (
                "理想对齐方向也无MSE净收益。"
                if cur["even_perfect_alignment_fails"]
                else "理想对齐方向存在MSE净收益，仍需满足对齐条件。"
            ),
            f"基础高斯参考界松弛对应的方差因子：{cur['gaussian_reference_slack_variance_factor']:.8f}；额外软筛选相对该参考的sigma倍率：{cur['recycling_over_reference_scale_factor']:.8f}。",
            "",
        ]
    lines += [
        "## 参数扫描与成立/失败边界",
        "",
        f"共{s['profiles']}个公开场景、{s['non_gaussian_candidates']}个非高斯候选。当前固定联邦几何的54个候选中，方向MSE净收益{s['current_non_gaussian_direction_gains']}个，2σ_G尾部概率净收益{s['current_non_gaussian_tail2_gains']}个，会计上界不可行{s['current_infeasible']}个。",
        f"所有公开场景方向MSE净收益候选{s['all_direction_gains']}个；只改善2σ_G尾部而未改善方向MSE的候选{s['tail2_only_candidates']}个。",
        "",
    ]
    examples = []
    if best:
        examples.append(("当前联邦几何中方向MSE最好的非高斯候选", best))
    if s["best_public_direction_gain"]:
        examples.append(
            ("公开边界场景中的方向MSE收益示例", s["best_public_direction_gain"])
        )
    for title, r in examples:
        lines += [
            f"### {title}",
            "",
            f"场景：{r['profile']}；r={r['recycle_prob']}，tau={r['tau']}。",
            f"sigma倍率{r['noise_std_ratio']:.6f}，理想方向MSE比{r['ideal_parallel_mse_ratio']:.6f}，全轨迹总能量比{r['trajectory_total_energy_ratio']:.6f}。",
            "有用方向与历史轴的cos²必须严格高于："
            + str(r["minimum_cosine_squared_for_direction_gain"])
            + "；None表示即使完全对齐也不满足净收益条件。",
            "",
        ]
    lines += [
        "条件解释：先看加入隐私成本后的sigma膨胀，再看软筛选降低方差的程度。两者相乘仍小于1才有理想方向MSE净收益。",
        "若理想方向条件都不成立，任何方向偏差都无法使该方向MSE受益；若成立，仍需历史方向与真实有用方向足够对齐。",
        "即使方向MSE改善，全模型总噪声能量也可能增加，尤其每层只保护一个方向而层维度很大时。",
        "尾部概率必须在同一绝对阈值下比较；尾部改善与MSE改善是不同结论。",
        "",
        "## 蒙特卡洛核验与限制",
        "",
        f"选择{s['monte_carlo_cases']}个由解析结果预先确定的场景，各500,000次公开噪声采样；矩、尾部与能量检查全部通过六标准误容差。",
        "有用方向是公开给定的合成方向，检查cos²=0、0.5、1；没有把带噪历史方向当成私有真梯度，也没有读取或发布正式训练噪声。",
        "蒙特卡洛只验证分布/误差统计，不能替代DP证明。全部收费来自解析有效上界，不使用数值偏移网格充当隐私证书。",
        "负结果仅覆盖当前噪声家族、固定几何、登记网格及保守会计，不证明BR-DP或更紧会计下不可能改善。",
        "公开高预算/少轮数的成功案例不自动迁移到25轮联邦配置；本报告不声称提升了FMNIST准确率。",
        "",
        "文件：mechanisms.csv全扫描、current_FL.csv当前几何、summary.json、monte_carlo.json、geometry_lock.json、config.yaml和test_gate.json。",
        "",
    ]
    (out / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/brdp_net_gain.yaml")
    parser.add_argument("--output", default="results/brdp_net_gain_verified_20261028")
    args = parser.parse_args()
    s = run(
        yaml.safe_load(inside(args.config).read_text(encoding="utf-8")), args.output
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in s.items()
                if k
                not in {
                    "current_parameter_result",
                    "best_current_feasible",
                    "best_public_direction_gain",
                }
            },
            indent=2,
        )
    )
