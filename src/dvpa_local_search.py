"""Local physical precision search with held-out paired objective checks.

Actions are selected on search streams and checked on separate public virtual
streams. Acceptance is empirical evidence of proxy-objective decrease, not a
formal probability guarantee, task-accuracy guarantee, or new privacy theorem.
"""

from collections import Counter
import itertools
import time

import numpy as np

from .dvpa_filter import PrecisionAction
from .dvpa_ranking import frozen_scores, replicated_scores


def action_key(action):
    return (
        round(action.precision, 18),
        *(round(v, 14) for _, v in action.layer_precision),
    )


def blank_local(status):
    return dict(
        local_status=status,
        local_candidates=0,
        local_accepted=0,
        local_changed_coarse=0,
        local_time_changed=0,
        local_layers_changed=0,
        local_proxy_gain=0.0,
        local_gain_mc_se=0.0,
        local_gain_quad_error=0.0,
        local_gain_margin=0.0,
        local_rank_stable=1,
        local_samples=0,
        local_escalated=0,
        local_information_changed_action=0,
        local_information_margin=0.0,
    )


def neighborhood(controller, center):
    cfg = controller.cfg
    p0 = np.array([v for _, v in center.layer_precision])
    weights = [p0]
    transfers = () if cfg["mode"] == "time_only" else cfg["local_layer_transfers"]
    for d in transfers:
        for i, j in itertools.permutations(range(len(p0)), 2):
            if p0[i] - d < controller.protocol.p_min:
                continue
            p = p0.copy()
            p[i] -= d
            p[j] += d
            weights.append(p)
    zs = [center.precision]
    offsets = () if cfg["mode"] == "layer_only" else cfg["local_time_offsets"]
    for fraction in offsets:
        zs.extend(
            [center.precision * (1 - fraction), center.precision * (1 + fraction)]
        )
    for z in zs:
        for p in weights:
            yield PrecisionAction(
                center.round_index,
                float(z),
                tuple(zip(controller.protocol.dimensions, map(float, p), strict=True)),
                center.feedback_basis_hash,
                center.history_hash,
            )


def local_pool(controller, t, remaining, centers, tails):
    future = np.arange(t + 1, controller.protocol.rounds + 1)
    reserve_precision = controller.protocol.z0 if controller.cfg["mode"] == "layer_only" else controller.protocol.z_floor
    reserve = (
        controller.protocol.charges(future, reserve_precision).sum(axis=0)
        if len(future)
        else np.zeros(3)
    )
    result = {action_key(a): a for a in centers}
    feasible = {}
    for center in centers:
        for a in neighborhood(controller, center):
            z = a.precision
            if z not in feasible:
                feasible[z] = z >= controller.protocol.z_floor and np.all(
                    controller.protocol.charges(t, z) + reserve <= remaining
                )
                if feasible[z] and z not in tails:
                    tails[z] = controller.tail_plans(
                        t, remaining - controller.protocol.charges(t, z)
                    )
            if feasible[z]:
                result.setdefault(action_key(a), a)
    return list(result.values())


def decrease_evidence(low, high, reference, proposal, factor, tolerance):
    """Positive means the proposal improves over its fixed reference."""
    low = np.asarray(low)
    high = np.asarray(high)
    if (
        low.shape != high.shape
        or high.ndim != 2
        or len(high) < 2
        or not np.isfinite(high).all()
        or not np.isfinite(low).all()
    ):
        raise ValueError("Finite replicated objective differences required")
    differences = high[:, reference] - high[:, proposal]
    gain = float(differences.mean())
    se = float(differences.std(ddof=1) / np.sqrt(len(differences)))
    quad = abs(gain - float((low[:, reference] - low[:, proposal]).mean()))
    margin = gain - factor * se - quad - tolerance
    return dict(
        gain=gain,
        mc_se=se,
        quad_error=quad,
        margin=float(margin),
        supported=bool(margin > 0),
    )


def accept_proposal(low, high, proposal, frozen, incumbent, factor, tolerance):
    improvement = decrease_evidence(low, high, incumbent, proposal, factor, tolerance)
    information = decrease_evidence(low, high, frozen, proposal, factor, tolerance)
    if proposal == frozen or information["supported"]:
        if improvement["supported"]:
            return proposal, improvement, information, proposal != frozen
    # Predetermined no-information fallback must ALSO improve the full proxy.
    fallback = decrease_evidence(low, high, incumbent, frozen, factor, tolerance)
    if fallback["supported"]:
        return frozen, fallback, information, False
    return incumbent, improvement, information, False


def refine(controller, t, feedback, remaining, incumbent, coarse_log, started):
    cfg = controller.cfg
    coarse_actions, tails = controller.last_candidate_context
    coarse_frozen = frozen_scores(
        controller, t, feedback, coarse_actions, tails, cfg["ranking_validation_nodes"]
    )
    centers = [incumbent, coarse_actions[int(np.argmin(coarse_frozen))]]
    pool = local_pool(controller, t, remaining, centers, tails)
    # Optimize no-information planning over the SAME expanded action set.
    whole = {action_key(a): a for a in coarse_actions}
    whole.update({action_key(a): a for a in pool})
    whole = list(whole.values())
    frozen = whole[
        int(
            np.argmin(
                frozen_scores(
                    controller,
                    t,
                    feedback,
                    whole,
                    tails,
                    cfg["ranking_validation_nodes"],
                )
            )
        )
    ]
    lookup = {action_key(a): a for a in pool}
    lookup.setdefault(action_key(frozen), frozen)
    lookup.setdefault(action_key(incumbent), incumbent)
    pool = list(lookup.values())
    screen = [
        controller.score_actions(t, feedback, pool, tails, repeat=r, stream=61)
        for r in range(cfg["ranking_coarse_repeats"])
    ]
    shortlisted = {action_key(incumbent), action_key(frozen)}
    for rows in screen:
        shortlisted.update(
            action_key(pool[i])
            for i in np.argsort([x["objective"] for x in rows], kind="stable")[
                : cfg["ranking_top_k"]
            ]
        )
    finalists = [a for a in pool if action_key(a) in shortlisted]

    def evaluate(samples, validation_stream):
        _, _, _, search_hi, rank = replicated_scores(
            controller, t, feedback, finalists, tails, samples, 67
        )
        proposed = finalists[int(np.argmin(search_hi.mean(axis=0)))]
        held = {action_key(a): a for a in [incumbent, frozen, proposed]}
        held = list(held.values())
        ids = {action_key(a): i for i, a in enumerate(held)}
        low_rows, high_rows, lo, hi, _ = replicated_scores(
            controller, t, feedback, held, tails, samples, validation_stream
        )
        selected, imp, info, supported_info = accept_proposal(
            lo,
            hi,
            ids[action_key(proposed)],
            ids[action_key(frozen)],
            ids[action_key(incumbent)],
            cfg["ranking_gap_factor"],
            cfg["local_min_decrease"],
        )
        return dict(
            proposed=proposed,
            held=held,
            selected=selected,
            improvement=imp,
            information=info,
            supported_info=supported_info,
            ranking=rank,
            lo=lo,
            hi=hi,
            high_rows=high_rows,
        )

    samples = cfg["ranking_samples"]
    assessment = evaluate(samples, 89)
    first = dict(
        improvement=assessment["improvement"],
        information=assessment["information"],
        ranking=assessment["ranking"],
    )
    # Increase effort only for an empirically positive but unresolved decrease.
    uncertain = (
        not assessment["improvement"]["supported"]
        and assessment["improvement"]["gain"] > 0
    )
    uncertain |= (
        action_key(assessment["proposed"]) != action_key(frozen)
        and assessment["information"]["gain"] > 0
        and not assessment["information"]["supported"]
    )
    escalated = bool(uncertain and samples < cfg["ranking_max_samples"])
    if escalated:
        samples = cfg["ranking_max_samples"]
        assessment = evaluate(samples, 97)
    chosen = assessment["held"][assessment["selected"]]
    accepted = action_key(chosen) != action_key(incumbent)
    frozen_held = next(
        i
        for i, a in enumerate(assessment["held"])
        if action_key(a) == action_key(frozen)
    )
    actual_info = decrease_evidence(
        assessment["lo"],
        assessment["hi"],
        frozen_held,
        assessment["selected"],
        cfg["ranking_gap_factor"],
        cfg["local_min_decrease"],
    )
    info_changed = action_key(chosen) != action_key(frozen) and actual_info["supported"]
    log = dict(coarse_log)
    if accepted:
        rows = [r[assessment["selected"]] for r in assessment["high_rows"]]
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
        log["selected_tail_profile"] = Counter(
            r["selected_tail_profile"] for r in rows
        ).most_common(1)[0][0]
        log["fallback_reason"] = (
            ""
            if assessment["supported_info"]
            else "local_gain_without_supported_information_advantage"
        )
    imp = assessment["improvement"]
    log.update(
        local_status="accepted_supported_decrease"
        if accepted
        else "retained_coarse_no_supported_decrease",
        local_candidates=len(pool),
        local_accepted=int(accepted),
        local_changed_coarse=int(accepted),
        local_time_changed=int(
            accepted
            and not np.isclose(
                chosen.precision, incumbent.precision, rtol=1e-12, atol=0
            )
        ),
        local_layers_changed=int(
            accepted
            and not np.allclose(
                [v for _, v in chosen.layer_precision],
                [v for _, v in incumbent.layer_precision],
                rtol=0,
                atol=1e-12,
            )
        ),
        local_proxy_gain=imp["gain"] if accepted else 0.0,
        local_gain_mc_se=imp["mc_se"],
        local_gain_quad_error=imp["quad_error"],
        local_gain_margin=imp["margin"],
        local_rank_stable=int(assessment["ranking"]["stable"]),
        local_samples=samples,
        local_escalated=int(escalated),
        local_information_changed_action=int(info_changed),
        local_information_margin=actual_info["margin"],
        controller_seconds=time.perf_counter() - started,
    )
    controller.last_ranking["selection_scope"] = (
        "coarse ranking before separately validated local refinement"
    )
    controller.last_local_search = dict(
        round=t,
        status=log["local_status"],
        scope="empirical proxy decrease on independent public virtual streams, not formal coverage or task gain",
        coarse_action=incumbent.as_dict(),
        frozen_action=frozen.as_dict(),
        selected_action=chosen.as_dict(),
        proposed_action=assessment["proposed"].as_dict(),
        candidate_count=len(pool),
        shortlist_actions=[a.as_dict() for a in finalists],
        validation_actions=[a.as_dict() for a in assessment["held"]],
        samples_per_repeat=samples,
        first_stage=first,
        final_improvement=imp,
        final_information=actual_info,
        proposed_information=assessment["information"],
        final_ranking=assessment["ranking"],
        accepted=accepted,
        information_supported=bool(info_changed),
        validation_low_scores=assessment["lo"].tolist(),
        validation_high_scores=assessment["hi"].tolist(),
        search_stream=67,
        validation_stream=97 if escalated else 89,
    )
    return chosen, log


def local_summary(decisions):
    rows = [r for r in decisions if not r["warmup"]]
    return dict(
        checked_rounds=len(rows),
        accepted_decrease_rounds=sum(r["local_accepted"] for r in rows),
        time_changed_rounds=sum(r["local_time_changed"] for r in rows),
        layer_changed_rounds=sum(r["local_layers_changed"] for r in rows),
        information_changed_action_rounds=sum(
            r["local_information_changed_action"] for r in rows
        ),
        escalated_rounds=sum(r["local_escalated"] for r in rows),
        scope="proxy decrease and conditional decision attribution; action count alone is not success",
    )
