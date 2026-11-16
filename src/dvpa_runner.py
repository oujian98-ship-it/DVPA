"""DVPA training adapter: all decisions/filtering precede current private updates."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml

from .brdp_source_protocol import (
    PersistentSourceClient,
    clip_source_update,
    load_datasets,
    partition_and_groups,
    release_source_sum,
)
from .brdp_source_runner import _json, _private_rngs
from .data import split_checksum
from .dvpa_config import ROOT, load_config, public_preflight, resolve, smoke_config
from .dvpa_controller import DecisionValueController
from .dvpa_feedback import DPHistoryFeedback
from .dvpa_filter import GroupRDPFilter, VERSION, digest
from .layer_registry import LayerRegistry
from .metrics import evaluate
from .st_runner import write_csv
from .upstream_runner import SOURCE, source_model
from .dvpa_public_training import evaluation_record as train_public_evaluation


def inside_output(output):
    path = Path(output).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("DVPA outputs must stay inside BR_DP")
    return path


def _snapshot(out):
    manifests = {}
    for folder, origin in [
        ("source_snapshot", ROOT / "src"),
        ("upstream_source", SOURCE),
    ]:
        manifest = {}
        for path in sorted(origin.rglob("*.py")):
            relative = path.relative_to(origin)
            dest = out / folder / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            manifest[relative.as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        manifests[folder] = manifest
    return manifests


def run(config, output):
    cfg = resolve(config)
    protocol = public_preflight(cfg)  # BEFORE dataset access.
    validation_proof = None
    forecast_proof = None
    if (
        cfg["feedback"]["utility_proxy"] == "public_curvature_signal"
        and cfg["base"].get("dataset_mode") != "synthetic"
    ):
        if "public_train_file" in cfg["base"]:
            from .dvpa_public_training import verify_validation as verify_public_validation
            from .dvpa_public_training import verify_validation as verify_forecast_validation
        else:
            from .dvpa_dataset_support import verify_public_validation, verify_forecast_validation

        validation_proof = verify_public_validation(cfg)
        if cfg["feedback"]["curvature_forecast"] != "hold_current":
            forecast_proof = verify_forecast_validation(cfg)
    out = inside_output(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    _json(out / "resolved_protocol.json", protocol.as_dict())
    _json(out / "privacy_filter.json", protocol.as_dict())
    manifests = _snapshot(out)
    _json(
        out / "implementation.json",
        dict(
            version=VERSION,
            method="DVPA",
            mechanism="gaussian",
            source_sha256=manifests,
            config_sha256=hashlib.sha256(
                (out / "config.yaml").read_bytes()
            ).hexdigest(),
            parameter_count=582026,
            trust_model="trusted_server_aggregate_only",
            update_units="raw=(local-global)/initial_lr*rate; release=raw_sum*initial_lr*rate/expected_count+noise",
            all_clients_train=True,
            private_randomness="independent OS seeds; never serialized",
            belief="approximate utility model, not a privacy assumption",
            guarantee_scope="ideal Gaussian real arithmetic; fixed-order conditional bounds on every history",
            resume_supported=False,
            torch=torch.__version__,
        ),
    )
    torch.set_num_threads(1)
    torch.manual_seed(cfg["base"]["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    base = cfg["base"]
    device = base["device"]
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Dataset/test metrics never enter feedback or choose().
    train, test = load_datasets(base)
    if "public_train_file" in base:
        from .dvpa_public_training import partition_record
        _json(out / "data_partition_protocol.json", partition_record(base))
    parts, group_ids = partition_and_groups(base, train.tensors[1])
    _json(
        out / "partition.json",
        dict(
            parts=parts,
            group_ids=group_ids.tolist(),
            split_checksum=split_checksum(parts),
            scope="frozen public benchmark partition only",
        ),
    )
    clients = [
        PersistentSourceClient(train, indices, base, device, slot=i)
        for i, indices in enumerate(parts)
    ]
    model = source_model()(base["dataset"], base["batch_size"]).to(device)
    registry = LayerRegistry(model)
    if registry.dimensions != protocol.dimensions or list(model.named_buffers()):
        raise ValueError("Author model signature/buffers changed")
    torch.save(
        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        out / "initial_model.pt",
    )
    ledger = GroupRDPFilter(protocol)
    feedback = DPHistoryFeedback(protocol)
    controller = DecisionValueController(protocol)
    public_proxy = None
    if cfg["feedback"]["utility_proxy"] == "public_curvature_signal":
        if "public_train_file" in base:
            from .dvpa_public_training import PublicCurvatureProxy
        else:
            from .dvpa_dataset_support import PublicCurvatureProxy
        from torch.utils.data import Subset

        public_proxy = PublicCurvatureProxy(cfg)
        if base.get("dataset_mode") != "synthetic":
            if len(test) != 10000:
                raise ValueError(
                    "Full official test split required before public reservation"
                )
            test = Subset(test, public_proxy.final_indices)
        shutil.copyfile(public_proxy.path, out / "public_proxy_data.pt")
        _json(
            out / "evaluation_protocol.json",
            train_public_evaluation(base, public_proxy.sha)
            if "public_train_file" in base else dict(
                dataset=base["dataset"].upper(),
                public_examples=1000,
                final_evaluation_examples=len(test),
                source_test_examples=10000,
                public_data_sha256=public_proxy.sha,
                original_test_has_historical_evaluations=True,
                scope="9000 held out from this iteration proxy calibration; old full-test scores are historical",
            ),
        )
        (out / "public_proxy").mkdir()
        if validation_proof is not None:
            _json(out / "public_validation.json", validation_proof)
        if forecast_proof is not None:
            _json(out / "forecast_validation.json", forecast_proof)
    sample_rng, scalar_rng, noise_rng = _private_rngs(device)
    (out / "released").mkdir()
    (out / "actions.jsonl").touch()
    rows = []
    groups = []
    decisions = []
    observations = []
    _json(
        out / "run_state.json",
        dict(status="initialized", completed_rounds=0, resume_supported=False),
    )
    try:
        for t in range(1, base["rounds"] + 1):
            feedback.prepare(t)
            if public_proxy is not None:
                proxy_report = public_proxy.report(model, t)
                feedback.set_public_coefficients(t, proxy_report)
                proxy_report.update(feedback.public_forecast_record())
                _json(out / "public_proxy" / f"round_{t:03d}.json", proxy_report)
            action, decision = controller.choose(t, feedback, ledger.remaining())
            certified = ledger.certify(action)
            receipt = ledger.reserve(certified)
            # Write-ahead receipt. A crash must never refund/reuse this charge.
            record = dict(certified=certified.as_dict(), receipt=receipt)
            record["record_hash"] = digest(record)
            with (out / "actions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            _json(
                out / "run_state.json",
                dict(
                    status="reserved",
                    completed_rounds=t - 1,
                    reserved_round=t,
                    spent=ledger.spent.tolist(),
                    resume_supported=False,
                ),
            )
            plan = certified.source_plan
            groups.extend(ledger.rows())
            write_csv(out / "group_privacy.csv", groups)
            private_sum = {
                n: torch.zeros(d, device=device) for n, d in registry.dimensions.items()
            }
            selected = (
                sample_rng.random(base["clients"])
                < np.asarray(plan.group_rates)[group_ids]
            )
            for i, client in enumerate(clients):
                update = client.train(model, registry, return_update=bool(selected[i]))
                if update is not None:
                    clipped = clip_source_update(
                        update, plan.group_radii[int(group_ids[i])], plan
                    )
                    for n in registry.names:
                        private_sum[n].add_(clipped[n])
                    del clipped, update
            directions = dict.fromkeys(registry.names, None)
            shapes = {n: (0.0, 1.0) for n in registry.names}
            released = release_source_sum(
                private_sum, plan, directions, shapes, scalar_rng, noise_rng
            )
            del private_sum, selected
            published = {n: v.detach().cpu().clone() for n, v in released.items()}
            torch.save(published, out / "released" / f"round_{t:03d}.pt")
            registry.apply(model, released)
            observations.extend(feedback.observe(t, published, certified))
            decisions.append(decision)
            loss, acc = evaluate(model, test, device)
            rows.append(
                dict(
                    round=t,
                    test_accuracy=acc,
                    test_loss=loss,
                    precision=action.precision,
                    denominator=plan.denominator,
                    local_epochs_completed=clients[0].epochs_completed,
                    scheduler_last_epoch=clients[0].scheduler.last_epoch,
                    next_local_lr=clients[0].optimizer.param_groups[0]["lr"],
                    non_gaussian_layers=0,
                    history_hash=feedback.history_hash,
                )
            )
            write_csv(out / "rounds.csv", rows)
            write_csv(out / "decisions.csv", decisions)
            if observations:
                write_csv(out / "feedback.csv", observations)
            else:
                (out / "feedback.csv").write_text("round,layer\n", encoding="utf-8")
            torch.save(
                {k: v.detach().cpu() for k, v in model.state_dict().items()},
                out / "model.pt",
            )
            torch.save(feedback.state_dict(), out / "feedback_state.pt")
            if cfg["controller"]["ranking_enabled"]:
                folder = out / "ranking"
                folder.mkdir(exist_ok=True)
                _json(folder / f"round_{t:03d}.json", controller.last_ranking)
            if cfg["controller"]["local_search_enabled"]:
                folder = out / "local_search"
                folder.mkdir(exist_ok=True)
                _json(folder / f"round_{t:03d}.json", controller.last_local_search)
            _json(out / "filter_state.json", ledger.state_dict())
            _json(
                out / "run_state.json",
                dict(
                    status="running",
                    completed_rounds=t,
                    spent=ledger.spent.tolist(),
                    resume_supported=False,
                ),
            )
            print(
                f"DVPA {cfg['controller']['mode']} seed={base['seed']} round={t}/{base['rounds']} "
                f"accuracy={acc:.2f}% z={action.precision:.8g} "
                f"path_eps={[round(r['epsilon_path_diagnostic'], 4) for r in ledger.rows()]}",
                flush=True,
            )
    except Exception as error:
        _json(
            out / "run_state.json",
            dict(
                status="failed_no_resume",
                completed_rounds=len(rows),
                reserved_round=len(ledger.receipts),
                spent=ledger.spent.tolist(),
                failure_type=type(error).__name__,
                reserved_costs_not_refunded=True,
                resume_supported=False,
            ),
        )
        raise
    summary = dict(
        status="completed",
        method="DVPA",
        mechanism="gaussian",
        controller_mode=cfg["controller"]["mode"],
        seed=base["seed"],
        rounds=len(rows),
        parameters=582026,
        dataset_scope="synthetic engineering"
        if base.get("dataset_mode") == "synthetic"
        else f"full public {base['dataset'].upper()} benchmark",
        group_privacy=ledger.rows(),
        final_test_accuracy=rows[-1]["test_accuracy"],
        best_test_accuracy=max(row["test_accuracy"] for row in rows),
        total_client_local_epochs=base["clients"]
        * base["rounds"]
        * base["local_epochs"],
        controller_seconds=sum(d["controller_seconds"] for d in decisions),
        guarantee="single-run fixed group caps for all histories; path values are diagnostics",
        final_history_hash=feedback.history_hash,
    )
    if public_proxy is not None:
        summary["evaluation_examples"] = len(test)
        summary["utility_proxy"] = "public_curvature_signal"
        summary["public_data_sha256"] = public_proxy.sha
        if "public_train_file" in base:
            summary["client_training_examples"] = (60000 - base.get("public_train_count", 1000))
            summary["public_source"] = "official_training_split"
        if cfg["feedback"]["curvature_forecast"] != "hold_current":
            summary["curvature_forecast"] = cfg["feedback"]["curvature_forecast"]
    if cfg["controller"]["ranking_enabled"]:
        from .dvpa_ranking import ranking_summary

        summary["numerical_ranking"] = ranking_summary(decisions)
    if cfg["controller"]["local_search_enabled"]:
        from .dvpa_local_search import local_summary

        summary["local_search"] = local_summary(decisions)
    _json(out / "summary.json", summary)
    _json(
        out / "run_state.json",
        dict(status="completed", completed_rounds=len(rows), resume_supported=False),
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume:
        parser.error(
            "DVPA v1 cannot resume persistent private clients; use a fresh output"
        )
    cfg = load_config(args.config)
    if args.smoke:
        cfg = smoke_config(cfg)
    if args.preview:
        print(json.dumps(public_preflight(cfg).as_dict(), indent=2))
        return
    if not args.output:
        parser.error("--output required except for --preview")
    print(json.dumps(run(cfg, args.output), indent=2))


if __name__ == "__main__":
    main()
