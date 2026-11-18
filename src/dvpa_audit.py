"""Replay DVPA decisions from public configuration and released DP history.

Audits artifact/algorithm consistency, not the physical noise generator or the
contents of private client optimizers. Reads designated public calibration input
only for the public-curvature variant, never private training or retained test input.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from .brdp_group_accountant import sampled_gaussian_rdp_upper
from .data import split_checksum
from .dvpa_config import ROOT, public_preflight, resolve
from .dvpa_controller import DecisionValueController
from .dvpa_feedback import DPHistoryFeedback
from .dvpa_filter import GroupRDPFilter, PrecisionAction, VERSION, digest
from .brdp_source_protocol import SourceRoundPlan
from .layer_registry import LayerRegistry
from .upstream_runner import SOURCE, source_model
from .dvpa_public_training import client_training_count, evaluation_record as train_public_evaluation


def _equal(actual, expected, label="artifact"):
    if isinstance(expected, torch.Tensor):
        if (
            not isinstance(actual, torch.Tensor)
            or actual.shape != expected.shape
            or not torch.allclose(
                actual.double(), expected.double(), rtol=1e-9, atol=1e-11
            )
        ):
            raise ValueError(f"Audit tensor mismatch: {label}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Audit fields mismatch: {label}")
        for k, v in expected.items():
            _equal(actual[k], v, f"{label}.{k}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f"Audit length mismatch: {label}")
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _equal(a, e, f"{label}[{i}]")
    elif isinstance(expected, bool):
        if actual not in (expected, str(expected)) or (
            type(actual) is not bool and not isinstance(actual, str)
        ):
            raise ValueError(f"Audit boolean mismatch: {label}")
    elif isinstance(expected, (int, float, np.number)):
        if (
            isinstance(actual, bool)
            or not math.isfinite(float(actual))
            or not math.isclose(
                float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-11
            )
        ):
            raise ValueError(f"Audit numeric mismatch: {label}")
    elif actual != expected:
        raise ValueError(f"Audit mismatch: {label}")


def _csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit(directory):
    root = Path(directory)
    cfg = resolve(yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")))
    impl = json.loads((root / "implementation.json").read_text(encoding="utf-8"))
    if (
        impl["version"] != VERSION
        or impl["mechanism"] != "gaussian"
        or impl["method"] != "DVPA"
    ):
        raise ValueError("Wrong DVPA implementation/accountant")
    if (
        impl["config_sha256"]
        != hashlib.sha256((root / "config.yaml").read_bytes()).hexdigest()
    ):
        raise ValueError("Config hash mismatch")
    for folder, origin in [
        ("source_snapshot", ROOT / "src"),
        ("upstream_source", SOURCE),
    ]:
        expected = {p.relative_to(origin).as_posix() for p in origin.rglob("*.py")}
        manifest = impl["source_sha256"][folder]
        if set(manifest) != expected:
            raise ValueError(
                "Incomplete source manifest; audit with matching frozen version"
            )
        for name, sha in manifest.items():
            p = Path(name)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Unsafe snapshot path")
            if (
                hashlib.sha256((root / folder / p).read_bytes()).hexdigest() != sha
                or hashlib.sha256((origin / p).read_bytes()).hexdigest() != sha
            ):
                raise ValueError("Source changed; audit with matching frozen source")
    protocol = public_preflight(cfg)
    if (
        cfg["feedback"]["curvature_forecast"] != "hold_current"
        and cfg["base"].get("dataset_mode") != "synthetic"
        and "public_test_count" not in cfg["base"]
    ):
        if "public_train_file" in cfg["base"]:
            from .dvpa_public_training import verify_validation as verify_forecast_validation
        else:
            from .dvpa_dataset_support import verify_forecast_validation

        verify_forecast_validation(
            cfg,
            json.loads((root / "forecast_validation.json").read_text(encoding="utf-8")),
        )
    for file in ("privacy_filter.json", "resolved_protocol.json"):
        _equal(
            json.loads((root / file).read_text(encoding="utf-8")),
            protocol.as_dict(),
            file,
        )
    state = json.loads((root / "run_state.json").read_text(encoding="utf-8"))
    if (
        state.get("status") != "completed"
        or state.get("completed_rounds") != protocol.rounds
    ):
        raise ValueError(
            "Incomplete run; pending reservations cannot be refunded/resumed"
        )
    partition = json.loads((root / "partition.json").read_text(encoding="utf-8"))
    if partition["split_checksum"] != split_checksum(partition["parts"]):
        raise ValueError("Partition checksum mismatch")
    count = (
        max(40 * protocol.base["clients"], protocol.base.get("synthetic_samples", 600))
        if protocol.base.get("dataset_mode") == "synthetic"
        else client_training_count(cfg["base"])
    )
    if (
        len(partition["parts"]) != protocol.base["clients"]
        or sorted(i for part in partition["parts"] for i in part) != list(range(count))
        or len(partition["group_ids"]) != protocol.base["clients"]
        or not np.array_equal(
            np.bincount(partition["group_ids"], minlength=3), protocol.counts
        )
    ):
        raise ValueError("Public benchmark slot/group coverage mismatch")
    torch.set_num_threads(1)
    torch.manual_seed(protocol.base["seed"])
    for _ in range(protocol.base["clients"]):
        source_model()(protocol.base["dataset"], protocol.base["batch_size"])
    model = source_model()(protocol.base["dataset"], protocol.base["batch_size"])
    initial = torch.load(
        root / "initial_model.pt", map_location="cpu", weights_only=True
    )
    if set(initial) != set(model.state_dict()) or any(
        not torch.equal(initial[k], v) for k, v in model.state_dict().items()
    ):
        raise ValueError("Author initialization order/seed mismatch")
    registry = LayerRegistry(model)
    if registry.dimensions != protocol.dimensions:
        raise ValueError("Model signature mismatch")
    records = [
        json.loads(line)
        for line in (root / "actions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if (
        len(records) != protocol.rounds
        or len(list((root / "released").glob("*.pt"))) != protocol.rounds
    ):
        raise ValueError("Action/release count mismatch")
    filter_ = GroupRDPFilter(protocol)
    feedback = DPHistoryFeedback(protocol)
    predictor = None
    if cfg["controller"]["voi_explore_p_only"]:
        from .dvpa_energy_voi import load_energy_gate

        predictor, _ = load_energy_gate(cfg, root / "energy_gate.json")
        if len(list((root / "energy_ranking").glob("*.json"))) != protocol.rounds:
            raise ValueError("Energy decision record coverage mismatch")
    controller = DecisionValueController(protocol, predictor)
    public_proxy = None
    if "public_train_file" in cfg["base"]:
        from .dvpa_public_training import partition_record, verify_validation
        _equal(json.loads((root / "data_partition_protocol.json").read_text(encoding="utf-8")),
               partition_record(cfg["base"]), "training_public_split")
        verify_validation(cfg, json.loads((root / "public_validation.json").read_text(encoding="utf-8")))
    if cfg["feedback"]["utility_proxy"] == "public_curvature_signal":
        if "public_test_count" in cfg["base"]:
            from .dvpa_test_public64 import PublicCurvatureProxy
        elif "public_train_file" in cfg["base"]:
            from .dvpa_public_training import PublicCurvatureProxy
        else:
            from .dvpa_dataset_support import PublicCurvatureProxy

        public_proxy = PublicCurvatureProxy(cfg)
        if (
            hashlib.sha256((root / "public_proxy_data.pt").read_bytes()).hexdigest()
            != public_proxy.sha
        ):
            raise ValueError("Saved public calibration input mismatch")
    expected_groups = []
    ranking_decisions = []
    expected_feedback = []
    decisions = _csv(root / "decisions.csv")
    rounds = _csv(root / "rounds.csv")
    if len(decisions) != protocol.rounds or len(rounds) != protocol.rounds:
        raise ValueError("Decision/round record count mismatch")
    for t, record in enumerate(records, 1):
        payload = {k: v for k, v in record.items() if k != "record_hash"}
        if record["record_hash"] != digest(payload):
            raise ValueError("Action record hash mismatch")
        feedback.prepare(t)
        if public_proxy is not None:
            report = public_proxy.report(model, t)
            feedback.set_public_coefficients(t, report)
            report.update(feedback.public_forecast_record())
            _equal(
                json.loads((root / "public_proxy" / f"round_{t:03d}.json").read_text()),
                report,
                "public_proxy",
            )
        expected_action, decision = controller.choose(t, feedback, filter_.remaining())
        if predictor is not None:
            _equal(json.loads((root / "energy_ranking" / f"round_{t:03d}.json").read_text(encoding="utf-8")),
                   controller.last_energy_ranking, "energy_ranking")
        if cfg["controller"]["ranking_enabled"]:
            ranking_decisions.append(decision)
        if cfg["controller"]["ranking_enabled"]:
            _equal(
                json.loads(
                    (root / "ranking" / f"round_{t:03d}.json").read_text(
                        encoding="utf-8"
                    )
                ),
                controller.last_ranking,
                "numerical_ranking",
            )
        if cfg["controller"]["local_search_enabled"]:
            _equal(
                json.loads(
                    (root / "local_search" / f"round_{t:03d}.json").read_text(
                        encoding="utf-8"
                    )
                ),
                controller.last_local_search,
                "local_search",
            )
        saved = record["certified"]
        # Exact action hashes bind decisions to their preceding history.
        if digest(saved["action"]) != digest(expected_action.as_dict()):
            raise ValueError("Action is not replayable from previous DP history")
        ac = saved["action"]
        action = PrecisionAction(
            ac["round_index"],
            ac["precision"],
            tuple(tuple(x) for x in ac["layer_precision"]),
            ac["feedback_basis_hash"],
            ac["history_hash"],
        )
        sp = saved["source_plan"].copy()
        for key in (
            "group_rates",
            "group_radii",
            "layer_clip_weights",
            "layer_budget_weights",
            "layer_noise_std",
            "planned_shapes",
        ):
            sp[key] = tuple(tuple(v) if isinstance(v, list) else v for v in sp[key])
        plan = SourceRoundPlan(**sp)
        certified = filter_.certify(action, plan)
        _equal(saved, certified.as_dict(), "certified")
        # Independent shared primitive, recomputed from actual sigma, jointly.
        for g in range(3):
            square = sum(
                (
                    plan.final_scale
                    * plan.group_radii[g]
                    * math.sqrt(w)
                    / dict(plan.layer_noise_std)[n]
                )
                ** 2
                for n, w in plan.layer_clip_weights
            )
            cost = sampled_gaussian_rdp_upper(
                plan.group_rates[g], square, [int(protocol.orders[g])]
            )[0]
            _equal(certified.group_rdp_increment[g], cost, "independent SGM cost")
        receipt = filter_.reserve(certified)
        _equal(record["receipt"], receipt, "receipt")
        expected_groups.extend(filter_.rows())
        for key in set(decision) - {"controller_seconds"}:
            _equal(decisions[t - 1][key], decision[key], "decision." + key)
        if (
            set(decisions[t - 1]) != set(decision)
            or not math.isfinite(float(decisions[t - 1]["controller_seconds"]))
            or float(decisions[t - 1]["controller_seconds"]) < 0
        ):
            raise ValueError("Invalid controller timing/fields")
        released = torch.load(
            root / "released" / f"round_{t:03d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        expected_feedback.extend(feedback.observe(t, released, certified))
        registry.apply(model, released)
        for key, value in dict(
            round=t,
            precision=action.precision,
            denominator=plan.denominator,
            local_epochs_completed=t * protocol.base["local_epochs"],
            scheduler_last_epoch=t * protocol.base["local_epochs"],
            non_gaussian_layers=0,
            history_hash=feedback.history_hash,
        ).items():
            _equal(rounds[t - 1][key], value, "round." + key)
    _equal(_csv(root / "group_privacy.csv"), expected_groups, "group_privacy")
    _equal(_csv(root / "feedback.csv"), expected_feedback, "feedback")
    _equal(
        torch.load(root / "feedback_state.pt", map_location="cpu", weights_only=True),
        feedback.state_dict(),
        "feedback_state",
    )
    _equal(
        json.loads((root / "filter_state.json").read_text(encoding="utf-8")),
        filter_.state_dict(),
        "filter_state",
    )
    final = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    if set(final) != set(model.state_dict()) or any(
        not torch.equal(final[k], v) for k, v in model.state_dict().items()
    ):
        raise ValueError("Final model does not equal accumulated DP releases")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    from .dvpa_test_public64 import verify_evaluation

    verify_evaluation(cfg["base"], root, summary)
    if public_proxy is not None:
        expected_count = (
            (cfg["base"].get("validation_count", 10000) if "public_train_file" in cfg["base"] else 9000)
            if cfg["base"].get("dataset_mode") != "synthetic"
            and "public_test_count" not in cfg["base"] else None
        )
        if expected_count is not None:
            _equal(
                summary["evaluation_examples"], expected_count, "evaluation_examples"
            )
            _equal(
                json.loads((root / "evaluation_protocol.json").read_text()),
                train_public_evaluation(cfg["base"], public_proxy.sha)
                if "public_train_file" in cfg["base"] else dict(
                    dataset=cfg["base"]["dataset"].upper(),
                    public_examples=1000,
                    final_evaluation_examples=9000,
                    source_test_examples=10000,
                    public_data_sha256=public_proxy.sha,
                    original_test_has_historical_evaluations=True,
                    scope="9000 held out from this iteration proxy calibration; old full-test scores are historical",
                ),
                "evaluation_protocol",
            )
        _equal(summary["utility_proxy"], "public_curvature_signal", "utility_proxy")
        if "validation_count" in cfg["base"]:
            _equal(summary["evaluation_split"], "independent_validation", "evaluation_split")
        if cfg["feedback"]["curvature_forecast"] != "hold_current":
            _equal(
                summary["curvature_forecast"],
                cfg["feedback"]["curvature_forecast"],
                "curvature_forecast",
            )
        _equal(summary["public_data_sha256"], public_proxy.sha, "public_data_sha256")
    if cfg["controller"]["ranking_enabled"]:
        from .dvpa_ranking import ranking_summary

        _equal(
            summary["numerical_ranking"],
            ranking_summary(ranking_decisions),
            "ranking_summary",
        )
    if cfg["controller"]["local_search_enabled"]:
        from .dvpa_local_search import local_summary

        _equal(
            summary["local_search"],
            local_summary(ranking_decisions),
            "local_search_summary",
        )
    for key, value in dict(
        status="completed",
        method="DVPA",
        mechanism="gaussian",
        rounds=protocol.rounds,
        parameters=582026,
        seed=protocol.base["seed"],
        controller_mode=cfg["controller"]["mode"],
        group_privacy=filter_.rows(),
        final_history_hash=feedback.history_hash,
        final_test_accuracy=float(rounds[-1]["test_accuracy"]),
        best_test_accuracy=max(float(r["test_accuracy"]) for r in rounds),
        total_client_local_epochs=protocol.base["clients"]
        * protocol.rounds
        * protocol.base["local_epochs"],
    ).items():
        _equal(summary[key], value, "summary." + key)
    return dict(
        status="passed",
        rounds=protocol.rounds,
        method="DVPA",
        group_privacy=filter_.rows(),
        scope="replayed decisions/feedback/conditional costs/models; not physical noise or client-state attestation",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.results_dir), indent=2))


if __name__ == "__main__":
    main()
