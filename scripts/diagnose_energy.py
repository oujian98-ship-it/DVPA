"""Stage A. Replay published releases and check the energy observation model."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr
from scipy.stats import ncx2
import torch

from src.dvpa_config import ROOT, load_config, public_preflight
from src.dvpa_energy_feedback import LayerEnergyState, energy_observations, energy_variance
from src.dvpa_filter import GroupRDPFilter, PrecisionAction


OUT = ROOT / "ccfa-workfiles/experiments/dvpa-energy"
ARCHIVE = ROOT / "results/group05_paper_studies_20261110/retry_06/02_fmnist_information_value_ablation"
FRESH = ROOT / "ccfa-workfiles/experiments/precision-planning/runs/convex_frozen_info_seed42"
TRAJECTORIES = [ARCHIVE / f"fmnist_DVPA_seed{s}" for s in (42, 43, 44)] + [FRESH]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Archive:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.config = load_config(self.folder / "config.yaml")
        self.protocol = public_preflight(self.config)
        self.seed = self.protocol.base["seed"]
        self.name = self.folder.name
        self.actions = (self.folder / "actions.jsonl").read_text(encoding="utf-8").splitlines()
        if len(self.actions) != self.protocol.rounds:
            raise ValueError("A complete archived trajectory is required")

    def public(self, t):
        return read_json(self.folder / "public_proxy" / f"round_{t:03d}.json")

    def release(self, t):
        return torch.load(self.folder / "released" / f"round_{t:03d}.pt",
                          map_location="cpu", weights_only=True)

    def reserve(self, ledger, t):
        saved = json.loads(self.actions[t - 1])["certified"]
        fields = saved["action"]
        action = PrecisionAction(t, fields["precision"], tuple(map(tuple, fields["layer_precision"])),
                                 fields["feedback_basis_hash"], fields["history_hash"])
        certified = ledger.certify(action)
        if certified.plan_hash != saved["plan_hash"]:
            raise ValueError(f"Archived action certificate mismatch at {self.name} round {t}")
        ledger.reserve(certified)
        return certified


def calibrate(dimensions):
    rng = np.random.default_rng(43105)
    results = []
    for d in sorted(set(dimensions)):
        for energy in (0.0, 0.1 * np.sqrt(2*d), np.sqrt(2*d), d, 10.0*d):
            draws = rng.noncentral_chisquare(d, energy, 100_000) - d
            variance = energy_variance(d, 1, energy)
            q = np.array([.01, .05, .5, .95, .99])
            exact = ncx2.ppf(q, d, energy) - d
            err = np.max(np.abs(ndtr((exact - energy) / np.sqrt(variance)) - q))
            results.append(dict(d_l=d, true_energy=float(energy), mean=float(draws.mean()),
                                mean_error_se=float((draws.mean()-energy) / np.sqrt(variance/len(draws))),
                                expected_variance=float(variance), variance_ratio=float(draws.var()/variance),
                                max_cdf_error=float(err), fraction_negative=float(np.mean(draws < 0))))
    return results


def collect(archives):
    rows = []
    for archive in archives:
        protocol = archive.protocol
        state = LayerEnergyState(protocol)
        ledger = GroupRDPFilter(protocol)
        for t in range(1, protocol.rounds + 1):
            cert = archive.reserve(ledger, t)
            observations = energy_observations(archive.release(t), cert, protocol.dimensions, archive.seed)
            state.observe(t, observations)
            for row in observations:
                phase = int(t >= state.switch_round)
                row.update(trajectory=archive.name, phase=phase)
                rows.append(row)
    return rows


def summarize(rows):
    summaries = []
    by_layer_phase = sorted({(r["layer"], r["phase"]) for r in rows})
    for layer, phase in by_layer_phase:
        subset = [r for r in rows if (r["layer"], r["phase"]) == (layer, phase)]
        for row in subset:
            peers = [r for r in subset if r["seed"] != row["seed"] and r["t"] == row["t"]]
            row["other_seeds_same_round_mean"] = float(np.mean([r["e_hat"] for r in peers])) if peers else None
        positive_seeds = set()
        independent_gain = []
        for trajectory in sorted({r["trajectory"] for r in subset}):
            series = [r for r in subset if r["trajectory"] == trajectory]
            if np.median([r["e_hat"] / r["noise_only_sd"] for r in series]) > 1:
                positive_seeds.add(series[0]["seed"])
            scores = [[], [], []]
            for i, row in enumerate(series):
                previous = series[max(0, i-3):i]
                window = series[max(0, i-2):i+1]
                row["rolling3_e_hat"] = np.mean([r["e_hat"] for r in window])
                row["rolling3_noise_only_sd"] = np.linalg.norm([r["noise_only_sd"] for r in window]) / len(window)
                if len(previous) < 3 or row["other_seeds_same_round_mean"] is None:
                    continue
                if any(r["other_seeds_same_round_mean"] is None for r in previous):
                    continue
                baseline = row["other_seeds_same_round_mean"]
                increments = [r["e_hat"] - r["other_seeds_same_round_mean"] for r in previous]
                for score, delta in zip(scores, (0, increments[-1], np.mean(increments)), strict=True):
                    score.append(abs(row["e_hat"] - baseline - delta))
            if scores[0]:
                # Remove the public round/phase trend using other seeds and see
                # whether within-trajectory history still helps.
                independent_gain.append((series[0]["seed"], *(float(np.mean(v)) for v in scores)))
        coverage = np.mean([.05 <= r["predictive_pit"] <= .95 for r in subset])
        fraction = float(np.mean([r["negative"] for r in subset]))
        for row in subset:
            row["fraction_negative"] = fraction
        summaries.append(dict(layer=layer, phase=phase, rows=len(subset), fraction_negative=fraction,
                              median_e_hat=float(np.median([r["e_hat"] for r in subset])),
                              median_noise_only_sd=float(np.median([r["noise_only_sd"] for r in subset])),
                              coverage90=float(coverage), signal_seeds=len(positive_seeds),
                              rolling_gain_seeds=len({seed for seed, base, one, three in independent_gain if three < base}),
                              time_adjusted_mae=independent_gain))
    return summaries


def plot(rows, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, layer in zip(axes.flat, dict.fromkeys(r["layer"] for r in rows)):
        for name in dict.fromkeys(r["trajectory"] for r in rows):
            series = [r for r in rows if (r["layer"], r["trajectory"]) == (layer, name)]
            rounds = [r["t"] for r in series]
            line, = ax.plot(rounds, [r["rolling3_e_hat"] for r in series], label=name)
            ax.scatter(rounds, [r["e_hat"] for r in series], s=9, alpha=.3, color=line.get_color())
        ax.axhline(0, linewidth=.7, color="grey")
        phase_switches = {r["t"] for r in rows if r["phase"] == 1}
        if phase_switches:
            ax.axvline(min(phase_switches)-.5, linewidth=.7, color="grey", linestyle="--")
        ax.set(title=layer, xlabel="Round", ylabel="Debiased energy")
    axes.flat[0].legend(fontsize=7)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs="+", default=TRAJECTORIES)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    archives = [Archive(path) for path in args.archive]
    rows = collect(archives)
    summary = summarize(rows)
    calibration = calibrate({row["d_l"] for row in rows})
    calibrated = all(abs(r["mean_error_se"]) < 5 and abs(r["variance_ratio"]-1) < .025
                     and r["max_cdf_error"] < .02 for r in calibration)
    # At least one layer/phase must show replicated signal and held-out
    # temporal prediction beyond a matched round/phase baseline.
    qualified = [r for r in summary if r["signal_seeds"] >= 2 and r["rolling_gain_seeds"] >= 2
                 and .8 <= r["coverage90"] <= 1.0]
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "energy_observability.csv", rows)
    write_csv(args.out / "energy_observation_calibration.csv", calibration)
    write_json(args.out / "energy_observability.json", dict(passed=calibrated and bool(qualified),
               calibrated=calibrated, qualified=qualified, summary=summary, archives=[a.name for a in archives]))
    plot(rows, args.out / "energy_observability.png")
    print(f"Stage A gate: {calibrated and bool(qualified)}; qualified layer/phase pairs: {len(qualified)}")


if __name__ == "__main__":
    main()
