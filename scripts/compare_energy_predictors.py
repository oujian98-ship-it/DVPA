"""Stage B. Compare public, energy and legacy predictors on held-out seeds."""

import argparse
import copy
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.diagnose_energy import Archive, OUT, TRAJECTORIES, read_json, write_csv, write_json
from src.dvpa_config import public_preflight
from src.dvpa_controller import spatial_optimum
from src.dvpa_energy_feedback import LayerEnergyState, energy_observations
from src.dvpa_energy_forecast import CurvaturePredictor, HORIZONS, VARIANTS, public_features
from src.dvpa_feedback import DPHistoryFeedback
from src.dvpa_filter import GroupRDPFilter


def analysis_protocol(archive):
    cfg = copy.deepcopy(archive.config)
    cfg["feedback"]["curvature_mode"] = "absolute_scale"
    cfg["controller"].update(tail_solver="convex", mode="frozen_info", ranking_enabled=False,
                             local_search_enabled=False, information_value_weight=0)
    return public_preflight(cfg)


def replay(archive, *, after_release=False):
    """Replay at the chosen side of the boundary of the current release."""
    protocol = analysis_protocol(archive)
    legacy = DPHistoryFeedback(protocol)
    energy = LayerEnergyState(protocol)
    ledger = GroupRDPFilter(archive.protocol)
    for t in range(1, protocol.rounds+1):
        legacy.prepare(t)
        legacy.set_public_coefficients(t, archive.public(t))
        if not after_release:
            yield t, protocol, legacy, energy, ledger
        certified = archive.reserve(ledger, t)
        released = archive.release(t)
        energy.observe(t, energy_observations(released, certified, protocol.dimensions, archive.seed))
        bound = replace(certified.action, feedback_basis_hash=legacy.basis_hash)
        legacy.observe(t, released, ledger.certify(bound))
        if after_release:
            yield t, protocol, legacy, energy, ledger


def collect(archives):
    samples = []
    for archive in archives:
        targets = {}
        for t, protocol, state, energy, _ in replay(archive, after_release=True):
            targets[t] = state.pure_public_costs([t], t)[0]
            if t >= protocol.config["feedback"]["warmup_rounds"] + 1:
                ids = state.indices(t)
                direction = np.column_stack([state.mean[ids], np.sqrt(np.diag(state.cov)[ids])])
                for horizon in HORIZONS:
                    if t+horizon > protocol.rounds:
                        continue
                    offset, features = public_features(state, t, horizon)
                    samples.append(dict(archive=archive.name, seed=archive.seed, t=t, horizon=horizon,
                                        names=list(protocol.dimensions), floor=protocol.p_min,
                                        offset=offset, targets=targets, B0_public=features,
                                        B1_energy=np.column_stack([features, energy.features(energy.posterior)]),
                                        B2_legacy_direction=np.column_stack([features, direction])))
    for sample in samples:
        sample["target"] = sample.pop("targets")[sample["t"] + sample["horizon"]]
    return samples


def evaluate(samples):
    rows, folds = [], {}
    for seed in sorted({x["seed"] for x in samples}):
        train = [e for e in samples if e["seed"] != seed]
        held_out = [e for e in samples if e["seed"] == seed]
        if len({e["seed"] for e in train}) < 2:
            raise ValueError("At least three distinct seeds are required for leave-seed-out evaluation")
        folds[seed] = {}
        for variant in VARIANTS:
            model = CurvaturePredictor.fit(train, variant)
            if variant == "B1_energy":
                folds[seed] = model.record()
            for e in held_out:
                target = e["target"]
                median, lower, upper = model.predict(e["horizon"], e["offset"], e[variant])
                expected = model.expected(e["horizon"], e["offset"], e[variant])
                allocation = spatial_optimum(expected, e["floor"])
                oracle = spatial_optimum(target, e["floor"])
                oracle_loss = np.sum(target/oracle)
                regret = (np.sum(target/allocation) - oracle_loss)/oracle_loss
                for i, layer in enumerate(e["names"]):
                    rows.append(dict(trajectory=e["archive"], seed=seed, t=e["t"], horizon=e["horizon"],
                                     variant=variant, layer=layer, observed=target[i], prediction=median[i],
                                     lower90=lower[i], upper90=upper[i], covered=bool(lower[i]<=target[i]<=upper[i]),
                                     error=median[i]-target[i], log_error=np.log(median[i]/target[i]),
                                     p=allocation[i], oracle_p=oracle[i], p_error=abs(allocation[i]-oracle[i]),
                                     allocation_regret=regret))
    return rows, folds


def summarize(rows):
    result = []
    dimensions = ("seed", "horizon", "variant", "layer")
    for key in sorted({tuple(r[k] for k in dimensions) for r in rows}):
        block = [r for r in rows if tuple(r[k] for k in dimensions) == key]
        result.append(dict(zip(dimensions, key), n=len(block),
                           mae=float(np.mean([abs(r["error"]) for r in block])),
                           rmse=float(np.sqrt(np.mean([r["error"]**2 for r in block]))),
                           log_rmse=float(np.sqrt(np.mean([r["log_error"]**2 for r in block]))),
                           coverage90=float(np.mean([r["covered"] for r in block])),
                           p_mae=float(np.mean([r["p_error"] for r in block])),
                           allocation_regret=float(np.mean([r["allocation_regret"] for r in block]))))
    index = {(r["seed"], r["horizon"], r["layer"]): r for r in result if r["variant"] == VARIANTS[0]}
    for r in result:
        base = index[r["seed"], r["horizon"], r["layer"]]
        r["rmse_improvement_vs_b0"] = base["rmse"] - r["rmse"]
        r["log_rmse_improvement_vs_b0"] = base["log_rmse"] - r["log_rmse"]
        r["regret_improvement_vs_b0"] = base["allocation_regret"] - r["allocation_regret"]
        r["p_mae_improvement_vs_b0"] = base["p_mae"] - r["p_mae"]
    return result


def gate(rows):
    details = []
    for seed in sorted({r["seed"] for r in rows}):
        for horizon in HORIZONS:
            stats = []
            for variant in VARIANTS[:2]:
                sub = [r for r in rows if (r["seed"], r["horizon"], r["variant"]) == (seed, horizon, variant)]
                stats.append((float(np.sqrt(np.mean([r["log_error"]**2 for r in sub]))),
                              float(np.mean([r["allocation_regret"] for r in sub])),
                              float(np.mean([r["covered"] for r in sub]))))
            b0, b1 = stats
            details.append(dict(seed=seed, horizon=horizon, b0_log_rmse=b0[0], b1_log_rmse=b1[0],
                                b0_regret=b0[1], b1_regret=b1[1], b1_coverage=b1[2],
                                passed=b1[0] < .99*b0[0] and b1[1] < b0[1] and .8 <= b1[2] <= .98))
    return all(r["passed"] for r in details), details


def plot(rows, path):
    first = rows[0]["trajectory"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), layout="constrained")
    for axis, (horizon, layer) in zip(axes.flat, [(h, n) for h in HORIZONS
                                                 for n in dict.fromkeys(r["layer"] for r in rows)]):
        selected = [r for r in rows if (r["trajectory"], r["horizon"], r["layer"]) == (first, horizon, layer)]
        for variant in VARIANTS:
            block = [r for r in selected if r["variant"] == variant]
            time = [r["t"]+horizon for r in block]
            axis.plot(time, [r["prediction"] for r in block], label=variant, linewidth=1)
            if variant == VARIANTS[1]:
                axis.fill_between(time, [r["lower90"] for r in block], [r["upper90"] for r in block], alpha=.15)
        axis.scatter(time, [r["observed"] for r in block], c="black", s=7, label="Observed")
        axis.set(title=f"{layer}, t+{horizon}", xlabel="Target round", yscale="log")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(first)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs="+", default=TRAJECTORIES)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    samples = collect([Archive(path) for path in args.archive])
    rows, folds = evaluate(samples)
    passed, details = gate(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "energy_predictive_value.csv", rows)
    write_csv(args.out / "energy_predictive_value_summary.csv", summarize(rows))
    write_json(args.out / "energy_predictive_value.json", dict(passed=passed, details=details,
               archives=list(dict.fromkeys(e["archive"] for e in samples)), folds=folds,
               refit=CurvaturePredictor.fit(samples, "B1_energy").record()))
    plot(rows, args.out / "energy_predictive_value.png")
    print(f"Stage B gate: {passed}; passing seed/horizon pairs: {sum(r['passed'] for r in details)}/{len(details)}")


if __name__ == "__main__":
    main()
