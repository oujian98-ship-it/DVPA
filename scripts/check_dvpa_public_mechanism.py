"""Public mechanism checks and full-size controller timing; NO FL accuracy run."""

import argparse
import copy
import json
import time

import numpy as np
import torch

from src.brdp_source_runner import _json
from src.dvpa_config import ROOT, load_config, public_preflight
from src.dvpa_controller import DecisionValueController, allocation_risk
from src.dvpa_feedback import DPHistoryFeedback
from src.dvpa_filter import GroupRDPFilter


def check(config):
    torch.set_num_threads(1)
    protocol = public_preflight(config)
    feedback = DPHistoryFeedback(protocol)
    ledger = GroupRDPFilter(protocol)
    controller = DecisionValueController(protocol)
    # Explicitly public zero-contribution warmup, not a private FL execution.
    for t in range(1, config["feedback"]["warmup_rounds"] + 1):
        feedback.prepare(t)
        action, _ = controller.choose(t, feedback, ledger.remaining())
        cert = ledger.certify(action)
        ledger.reserve(cert)
        feedback.observe(
            t, {n: torch.zeros(d) for n, d in protocol.dimensions.items()}, cert
        )
    t = config["feedback"]["warmup_rounds"] + 1
    feedback.prepare(t)
    started = time.perf_counter()
    chosen, log = controller.choose(t, feedback, ledger.remaining())
    seconds = time.perf_counter() - started
    assert ledger.can_reserve(ledger.certify(chosen))
    terminal = copy.deepcopy(feedback)
    terminal.prepared_round = protocol.rounds
    _, last = controller.choose(protocol.rounds, terminal, ledger.remaining())
    assert last["information_value"] == 0
    fixed = copy.deepcopy(feedback)
    fixed.protocol = copy.deepcopy(protocol)
    fixed.protocol.config["feedback"]["lambda1"] = 0.0
    fixed.cfg = fixed.protocol.config["feedback"]
    _, no_learning = DecisionValueController(fixed.protocol).choose(
        t, fixed, ledger.remaining()
    )
    assert abs(no_learning["information_value"]) < 1e-8
    state = np.array([[1.0, 4.0, 9.0, 16.0], [9.0, 36.0, 81.0, 144.0]])
    common_value = float(
        allocation_risk(state.mean(axis=0), 0.02) - allocation_risk(state, 0.02).mean()
    )
    assert abs(common_value) < 1e-10
    rng = np.random.default_rng(42)
    normals = rng.standard_normal((64, 4))
    _, low, _ = feedback.posterior(
        t, protocol.z0 * 0.25, [0.25] * 4, standard_normals=normals
    )
    _, high, _ = feedback.posterior(
        t, protocol.z0 * 3.0, [0.25] * 4, standard_normals=normals
    )
    assert np.trace(high) < np.trace(low)
    # Public wrong-prior stress: even arbitrary utility beliefs cannot bypass
    # the independent actual-noise filter. This is NOT an empirical DP proof.
    rejected = ledger.certify(
        type(chosen)(
            chosen.round_index,
            chosen.precision * 10000,
            chosen.layer_precision,
            chosen.feedback_basis_hash,
            chosen.history_hash,
        )
    )
    assert not ledger.can_reserve(rejected)
    return dict(
        status="passed",
        scope="public mechanism/working-model checks only; no FL accuracy",
        candidate_count=log["candidates"],
        controller_seconds=seconds,
        selected_action=chosen.as_dict(),
        decision=log,
        common_scale_spatial_value=common_value,
        known_weights_information_value=no_learning["information_value"],
        terminal_information_value=last["information_value"],
        low_precision_posterior_trace=float(np.trace(low)),
        high_precision_posterior_trace=float(np.trace(high)),
        oversized_candidate_rejected=True,
        note="Information values use finite Monte Carlo and an approximate state model; no accuracy/optimality claim",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dvpa_fmnist.yaml")
    parser.add_argument("--output", default="reports/dvpa/public_mechanism.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = (ROOT / args.output).resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Output must stay inside BR_DP")
    out.parent.mkdir(parents=True, exist_ok=True)
    result = check(cfg)
    _json(out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
