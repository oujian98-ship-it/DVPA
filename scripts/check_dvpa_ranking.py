"""Offline numerical checks on frozen released histories; no FL retraining."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.brdp_source_runner import _json
from src.dvpa_config import ROOT, public_preflight, resolve
from src.dvpa_controller import DecisionValueController
from src.dvpa_feedback import DPHistoryFeedback
from src.dvpa_filter import GroupRDPFilter, PrecisionAction


def check(directory, rounds, output, local_search=False):
    root = Path(directory).resolve()
    out = Path(output).resolve()
    if not root.is_relative_to(ROOT) or not out.is_relative_to(ROOT):
        raise ValueError("DVPA diagnostics must stay inside BR_DP")
    cfg = resolve(yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")))
    cfg["controller"]["ranking_enabled"] = True
    cfg["controller"]["local_search_enabled"] = bool(local_search)
    protocol = public_preflight(cfg)
    feedback = DPHistoryFeedback(protocol)
    ledger = GroupRDPFilter(protocol)
    controller = DecisionValueController(protocol)
    torch.set_num_threads(1)
    saved = [
        json.loads(line)["certified"]["action"]
        for line in (root / "actions.jsonl").read_text().splitlines()
    ]
    if (
        not rounds
        or len(set(rounds)) != len(rounds)
        or any(t <= cfg["feedback"]["warmup_rounds"] or t > len(saved) for t in rounds)
    ):
        raise ValueError("Choose distinct saved non-warmup rounds")
    result = dict(
        status="running",
        scope="offline frozen histories/balances; no new private query or training accuracy",
        source_results=root.relative_to(ROOT).as_posix(),
        rounds=rounds,
        checks=[],
        changed_scope="local physical precision search plus numerical checks; feedback covariance, proxy and filter unchanged"
        if local_search
        else "numerical ranking only; physical candidates, feedback covariance, proxy and filter unchanged",
        numerical_config={
            k: v for k, v in cfg["controller"].items() if k.startswith("ranking_")
        },
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "src").glob("dvpa*.py")
        },
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    for t, item in enumerate(saved, 1):
        feedback.prepare(t)
        previous = PrecisionAction(
            item["round_index"],
            item["precision"],
            tuple(tuple(x) for x in item["layer_precision"]),
            item["feedback_basis_hash"],
            item["history_hash"],
        )
        if t in rounds:
            actions, tails = controller.candidate_context(
                t, feedback, ledger.remaining()
            )
            scores = controller.score_actions(t, feedback, actions, tails)
            old_index = int(np.argmin([r["objective"] for r in scores]))
            old = actions[old_index]
            z_error = abs(old.precision - previous.precision) / previous.precision
            p_error = float(
                np.max(
                    np.abs(
                        np.array([v for _, v in old.layer_precision])
                        - np.array([v for _, v in previous.layer_precision])
                    )
                )
            )
            if z_error > 1e-12 or p_error > 1e-12:
                raise ValueError(
                    "Legacy candidate refactor no longer matches saved v1 action"
                )
            chosen, log = controller.choose(t, feedback, ledger.remaining())
            row = dict(
                round=t,
                legacy_reproduced=True,
                legacy_action=previous.as_dict(),
                selected_action=chosen.as_dict(),
                decision=log,
                ranking=controller.last_ranking,
            )
            if local_search:
                row["local_search"] = controller.last_local_search
            result["checks"].append(row)
            _json(out, result)
            print(
                json.dumps(
                    {
                        "round": t,
                        **{
                            k: v
                            for k, v in log.items()
                            if k.startswith(("ranking_", "local_"))
                        },
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        cert = ledger.certify(previous)
        ledger.reserve(cert)
        released = torch.load(
            root / "released" / f"round_{t:03d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        feedback.observe(t, released, cert)
        if t >= max(rounds):
            break
    result["status"] = "completed"
    result["summary"] = dict(
        checked_rounds=len(rounds),
        separated_rounds=sum(x["decision"]["ranking_stable"] for x in result["checks"]),
        changed_v1_rounds=sum(
            x["decision"]["ranking_changed_v1"] for x in result["checks"]
        ),
        information_changed_action_rounds=sum(
            x["decision"]["ranking_value_changed_action"] for x in result["checks"]
        ),
    )
    if local_search:
        result["summary"].update(
            accepted_decrease_rounds=sum(
                x["decision"]["local_accepted"] for x in result["checks"]
            ),
            local_information_changed_action_rounds=sum(
                x["decision"]["local_information_changed_action"]
                for x in result["checks"]
            ),
        )
    _json(out, result)
    return result["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results/dvpa_vs_idp_sayg_seeds42_43_44_20261028/DVPA_seed42",
    )
    parser.add_argument("--rounds", nargs="+", type=int, default=[4, 8, 13, 20, 24])
    parser.add_argument("--output", default="reports/dvpa_ranking/offline_check.json")
    parser.add_argument("--local-search", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            check(
                ROOT / args.results_dir,
                args.rounds,
                ROOT / args.output,
                args.local_search,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
