"""Empirical numerical-ranking checks, independent of the privacy filter.

Intervals here are diagnostics on a screened candidate set, NOT statistical
coverage guarantees, global optimization certificates, or privacy guarantees.
No real query is made during virtual replication or quadrature refinement.
"""

from collections import Counter
import time

import numpy as np


def blank_log(status):
    return dict(
        ranking_status=status,
        ranking_stable=1,
        ranking_escalated=0,
        ranking_shortlist=0,
        ranking_samples=0,
        ranking_repeats=0,
        ranking_gap=0.0,
        ranking_mc_gap_se=0.0,
        ranking_quad_gap_error=0.0,
        ranking_min_margin=0.0,
        ranking_coarse_agreement=1.0,
        ranking_changed_v1=0,
        ranking_value_changed_action=0,
    )


def paired_evidence(low, high, factor, tolerance):
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    if (
        low.shape != high.shape
        or high.ndim != 2
        or high.shape[0] < 2
        or not np.isfinite(high).all()
        or not np.isfinite(low).all()
    ):
        raise ValueError("Finite replicated candidate scores required")
    means = high.mean(axis=0)
    order = np.argsort(means, kind="stable")
    winner = int(order[0])
    if len(order) == 1:
        return dict(
            winner=winner,
            runner_up=-1,
            stable=True,
            gap=0.0,
            mc_gap_se=0.0,
            quad_gap_error=0.0,
            minimum_margin=0.0,
        )
    competitors = order[1:]
    differences = high[:, competitors] - high[:, winner, None]
    gaps = differences.mean(axis=0)
    errors = differences.std(axis=0, ddof=1) / np.sqrt(len(high))
    low_gaps = (low[:, competitors] - low[:, winner, None]).mean(axis=0)
    quad = np.abs(gaps - low_gaps)
    margins = gaps - factor * errors - quad - tolerance
    return dict(
        winner=winner,
        runner_up=int(competitors[0]),
        stable=bool(np.all(margins > 0)),
        gap=float(gaps[0]),
        mc_gap_se=float(errors[0]),
        quad_gap_error=float(quad[0]),
        minimum_margin=float(margins.min()),
    )


def frozen_scores(controller, t, feedback, actions, tails, nodes):
    current = feedback.expected_costs(
        feedback.pred_mean, feedback.pred_cov, [t], origin=t, quadrature_nodes=nodes
    )[0, 0]
    cache = {}
    scores = []
    for action in actions:
        z = action.precision
        future, plans = tails[z]
        if z not in cache:
            if len(future):
                k = feedback.expected_costs(
                    feedback.pred_mean,
                    feedback.pred_cov,
                    future,
                    origin=t,
                    quadrature_nodes=nodes,
                )
                cache[z] = float(controller.tail_value_from_costs(k, plans)[0][0])
            else:
                cache[z] = 0.0
        p = np.array([v for _, v in action.layer_precision])
        scores.append(float(np.sum(current / p) / z + cache[z]))
    return np.array(scores)


def replicated_scores(controller, t, feedback, actions, tails, samples, stream):
    """Paired MC/GH scores; caller chooses separate search/validation streams."""
    cfg = controller.cfg
    lower = []
    higher = []
    for repeat in range(cfg["ranking_repeats"]):
        lower.append(
            controller.score_actions(
                t,
                feedback,
                actions,
                tails,
                samples=samples,
                nodes=cfg["ranking_quadrature_nodes"],
                repeat=repeat,
                stream=stream,
            )
        )
        higher.append(
            controller.score_actions(
                t,
                feedback,
                actions,
                tails,
                samples=samples,
                nodes=cfg["ranking_validation_nodes"],
                repeat=repeat,
                stream=stream,
            )
        )
    lo = np.array([[r["objective"] for r in rows] for rows in lower])
    hi = np.array([[r["objective"] for r in rows] for rows in higher])
    evidence = paired_evidence(
        lo, hi, cfg["ranking_gap_factor"], cfg["ranking_tolerance"]
    )
    return lower, higher, lo, hi, evidence


def choose_stable(controller, t, feedback, remaining, started):
    cfg = controller.cfg
    actions, tails = controller.candidate_context(t, feedback, remaining)
    controller.last_candidate_context = (actions, tails)
    fine_nodes = cfg["ranking_quadrature_nodes"]
    validation_nodes = cfg["ranking_validation_nodes"]
    frozen = frozen_scores(controller, t, feedback, actions, tails, validation_nodes)
    frozen_index = int(np.argmin(frozen))
    if t == controller.protocol.rounds or cfg["mode"] in {"frozen_info", "static", "time_only", "layer_only"}:
        logs = controller.score_actions(
            t,
            feedback,
            [actions[frozen_index]],
            tails,
            samples=2,
            nodes=validation_nodes,
            stream=31,
        )
        log = logs[0]
        log.pop("frozen_objective")
        log.update(
            blank_log(
                "terminal"
                if t == controller.protocol.rounds
                else "deterministic_frozen"
            )
        )
        log.update(
            candidates=len(actions), controller_seconds=time.perf_counter() - started
        )
        controller.last_ranking = dict(
            round=t,
            status=log["ranking_status"],
            candidate_count=len(actions),
            selected_index=frozen_index,
            selected_action=actions[frozen_index].as_dict(),
        )
        return actions[frozen_index], log
    coarse = []
    for repeat in range(cfg["ranking_coarse_repeats"]):
        coarse.append(
            controller.score_actions(t, feedback, actions, tails, repeat=repeat)
        )
    matrix = np.array([[r["objective"] for r in rows] for rows in coarse])
    winners = np.argmin(matrix, axis=1)
    selected = {frozen_index}
    for row in matrix:
        selected.update(
            map(int, np.argsort(row, kind="stable")[: cfg["ranking_top_k"]])
        )
    shortlist = sorted(selected)
    finalists = [actions[i] for i in shortlist]

    def stage(samples):
        return replicated_scores(controller, t, feedback, finalists, tails, samples, 31)

    samples = cfg["ranking_samples"]
    lower, higher, lo, hi, evidence = stage(samples)
    first_evidence = dict(evidence)
    escalated = not evidence["stable"] and cfg["ranking_max_samples"] > samples
    if escalated:
        samples = cfg["ranking_max_samples"]
        lower, higher, lo, hi, evidence = stage(samples)
    winner = evidence["winner"]
    chosen_global = shortlist[winner] if evidence["stable"] else frozen_index
    chosen_local = shortlist.index(chosen_global)
    rows = [r[chosen_local] for r in higher]
    log = dict(rows[0])
    for key in (
        "current_loss",
        "tail_loss",
        "information_value",
        "weighted_information_value",
        "unweighted_objective",
        "raw_information_value",
        "tower_correction",
        "working_mutual_information",
        "objective",
    ):
        log[key] = float(np.mean([r[key] for r in rows]))
    log.pop("frozen_objective")
    log["selected_tail_profile"] = Counter(
        r["selected_tail_profile"] for r in rows
    ).most_common(1)[0][0]
    if not evidence["stable"]:
        log["fallback_reason"] = "unresolved_numerical_ranking_use_frozen_info"
        log["objective"] = float(frozen[chosen_global])
    log.update(
        ranking_status="empirically_separated"
        if evidence["stable"]
        else "unresolved_frozen_fallback",
        ranking_stable=int(evidence["stable"]),
        ranking_escalated=int(escalated),
        ranking_shortlist=len(shortlist),
        ranking_samples=samples,
        ranking_repeats=cfg["ranking_repeats"],
        ranking_gap=evidence["gap"],
        ranking_mc_gap_se=evidence["mc_gap_se"],
        ranking_quad_gap_error=evidence["quad_gap_error"],
        ranking_min_margin=evidence["minimum_margin"],
        ranking_coarse_agreement=float(max(Counter(winners).values()) / len(winners)),
        ranking_changed_v1=int(chosen_global != int(winners[0])),
        ranking_value_changed_action=int(
            evidence["stable"] and chosen_global != frozen_index
        ),
        candidates=len(actions),
        controller_seconds=time.perf_counter() - started,
    )
    controller.last_ranking = dict(
        round=t,
        status=log["ranking_status"],
        scope="empirical paired MC/GH diagnostic on screened candidates; not a global or coverage certificate",
        candidate_count=len(actions),
        coarse_winners=winners.tolist(),
        shortlist_indices=shortlist,
        shortlist_actions=[a.as_dict() for a in finalists],
        frozen_index=frozen_index,
        selected_index=chosen_global,
        selected_action=actions[chosen_global].as_dict(),
        samples_per_repeat=samples,
        repeats=cfg["ranking_repeats"],
        nodes=[fine_nodes, validation_nodes],
        first_stage=first_evidence,
        final_stage=evidence,
        low_scores=lo.tolist(),
        high_scores=hi.tolist(),
        coarse_winner_agreement=log["ranking_coarse_agreement"],
    )
    return actions[chosen_global], log


def ranking_summary(decisions):
    rows = [
        r for r in decisions if not r["warmup"] and r["ranking_status"] != "terminal"
    ]
    return dict(
        checked_nonterminal_rounds=len(rows),
        separated_rounds=sum(r["ranking_stable"] for r in rows),
        escalated_rounds=sum(r["ranking_escalated"] for r in rows),
        frozen_fallback_rounds=sum(not r["ranking_stable"] for r in rows),
        different_from_single_estimate_rounds=sum(
            r["ranking_changed_v1"] for r in rows
        ),
        value_changed_action_rounds=sum(
            r["ranking_value_changed_action"] for r in rows
        ),
        scope="empirical numerical diagnostics, not proof of task utility or optimality",
    )
