"""Stage C. Fixed-z matched replay and synthetic VOI stress tests."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from scripts.compare_energy_predictors import analysis_protocol, replay
from scripts.diagnose_energy import Archive, OUT, TRAJECTORIES, read_json, write_csv, write_json
from src.dvpa_controller import DecisionValueController
from src.dvpa_convex import ConvexPrecision
from src.dvpa_energy_forecast import CurvaturePredictor
from src.dvpa_energy_voi import (choose_energy, compare_candidates, energy_options, gate_scope,
                                 layer_candidates)


def prepare(t, protocol, legacy, ledger):
    controller = DecisionValueController(protocol)
    remaining = ledger.remaining()
    future = np.arange(t, protocol.rounds+1)
    coefficients = legacy.pure_public_costs(future, t)
    planner = ConvexPrecision(protocol, future, remaining)
    schedule = planner.solve(np.sum(coefficients/controller.layer_weights(coefficients), axis=-1))
    return controller, remaining, coefficients, float(schedule[0]), controller.layer_weights(coefficients[0]), planner.fallbacks


def flattened(record, **labels):
    base = {k: v for k, v in record.items() if k != "candidates"}
    base.update(labels)
    return [dict(base, **row, is_selected=i == record["selected_index"])
            for i, row in enumerate(record["candidates"])] or [base]


def stress(archive):
    output = []
    for t, protocol, legacy, energy, ledger in replay(archive):
        if (t >= protocol.rounds
                or t not in {protocol.base["transition_round"]-1, protocol.base["transition_round"]+1}):
            continue
        controller, remaining, costs, z, p, _ = prepare(t, protocol, legacy, ledger)
        options = layer_candidates(p, controller.static_layer_weights(), protocol.p_min, costs[1:].mean(axis=0))
        for variation in (.1, .2, .3):
            log_sd = np.sqrt(np.log1p(variation**2))
            for quality, strength in (("zero", 0.0), ("weak", .25), ("strong", .85)):
                def forecast(share, draws):
                    _, normals, _ = draws
                    # Posterior predictive mean under a lognormal latent K and
                    # a standardized Gaussian channel with action-specific correlation.
                    correlation = np.clip(.35 + strength*(share-p)/max(p.max(), .1), .05, .98)
                    scaled = log_sd*correlation
                    update = np.exp(normals*scaled - scaled**2/2)
                    return costs[None, 1:, :]*update[:, None, :]

                selected, result = compare_candidates(controller, t, z, options, remaining, costs, legacy,
                                                       forecast, samples=16)
                if quality == "zero":
                    values = [r["voi_proposal"] for r in result["candidates"] if r["feasible"]]
                    if (selected or not values or any(v is None for v in values)
                            or not np.allclose(values, values[0], rtol=0, atol=1e-10)):
                        raise RuntimeError("Equal observation quality produced action-dependent information")
                output.extend(flattened(result, variant="synthetic_stress", trajectory=archive.name,
                                        seed=archive.seed, future_k_cv=variation, quality=quality, lambda_voi=1))
    return output


def real_replay(archives, evidence, weights):
    output, confirmed = [], set()
    for archive in archives:
        predictor = CurvaturePredictor.from_record(evidence["folds"][str(archive.seed)])
        for t, protocol, legacy, energy, ledger in replay(archive):
            if not protocol.config["feedback"]["warmup_rounds"] < t < protocol.rounds:
                continue
            controller, remaining, costs, z, p, fallbacks = prepare(t, protocol, legacy, ledger)
            if fallbacks:
                output.append(dict(trajectory=archive.name, seed=archive.seed, round=t,
                                   status="baseline_fallback", z=z))
                continue
            options = energy_options(controller, t, p, costs, energy)
            future = np.arange(t+1, protocol.rounds+1)
            prior = legacy.expected_costs(legacy.pred_mean, legacy.pred_cov, future, t)[0]

            def direction(share, draws):
                _, normals, _ = draws
                mean, covariance, _ = legacy.posterior(t, z, share, standard_normals=normals)
                return costs[1:] * (legacy.expected_costs(mean, covariance, future, t) / prior)

            def no_feedback(share, draws):
                return np.broadcast_to(costs[1:], (len(draws[0]), *costs[1:].shape))

            functions = {"legacy_direction+convex": direction, "no_VOI+convex": no_feedback}
            for weight in weights:
                selected, result = choose_energy(controller, legacy, t, z, p, remaining, costs,
                                                 predictor, energy=energy, weight=weight)
                output.extend(flattened(result, variant="energy+convex", seed=archive.seed,
                                        trajectory=archive.name, lambda_voi=weight))
                if selected and weight == 1:
                    confirmed.add(archive.seed)
                for variant, forecast in functions.items():
                    if weight != 1 and variant == "no_VOI+convex":
                        continue
                    _, result = compare_candidates(controller, t, z, options, remaining, costs, legacy, forecast,
                                                   weight=weight, samples=controller.cfg["virtual_observations"])
                    output.extend(flattened(result, variant=variant, seed=archive.seed,
                                            trajectory=archive.name, lambda_voi=weight))
    return output, sorted(confirmed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, nargs="+", default=TRAJECTORIES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sensitivity", action="store_true", help="Additional explicitly labeled weights 3 and 15")
    parser.add_argument("--exploratory", action="store_true", help="Run archived replay even if A or B failed")
    args = parser.parse_args()
    torch.set_num_threads(1)
    archives = [Archive(path) for path in args.archive]
    file_a = args.out / "energy_observability.json"
    file_b = args.out / "energy_predictive_value.json"
    stage_a, stage_b = read_json(file_a), read_json(file_b)
    if sorted(a.name for a in archives) != sorted(stage_a["archives"]) or sorted(stage_a["archives"]) != sorted(stage_b["archives"]):
        raise ValueError("Replay must use the same trajectories as A and B")
    passed_ab = stage_a["passed"] and stage_b["passed"]
    rows, confirmed = [], []
    weights = (1, 3, 15) if args.sensitivity else (1,)
    if passed_ab or args.exploratory:
        rows, confirmed = real_replay(archives, stage_b, weights)
        for row in rows:
            row["scope"] = "held_out" if passed_ab else "exploratory_failed_A_or_B"
    else:
        rows.append(dict(variant="archived_replay", status="prerequisite_failed_A_or_B"))
    stress_rows = stress(archives[0])
    rows.extend(stress_rows)
    write_csv(args.out / "energy_voi_replay.csv", rows)
    passed = passed_ab and len(confirmed) >= 2
    result = dict(version=1, passed={"A": stage_a["passed"], "B": stage_b["passed"], "C": passed},
                  confirming_seeds=confirmed, archives=[a.name for a in archives],
                  stress_candidate_rows=len(stress_rows),
                  phase_evidence_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (file_a, file_b)})
    if passed:
        scopes = [gate_scope(analysis_protocol(a).config) for a in archives]
        if any(scopes[0] != other for other in scopes[1:]):
            raise ValueError("Gate data must share one calibrated online protocol")
        result.update(predictor=stage_b["refit"], scope=scopes[0])
    write_json(args.out / "energy_voi_gate.json", result)
    print(f"Stage C gate: {passed}; independent confirming seeds: {confirmed}; stress rows: {len(stress_rows)}")


if __name__ == "__main__":
    main()
