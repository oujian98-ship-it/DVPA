"""Screenshot-aligned, three-budget DP-FL runner; no trynew imports/writes."""

import argparse
import hashlib
import json
from pathlib import Path
import secrets
import shutil
import time

import numpy as np
import torch
import yaml

from .brdp_direction import DirectionState
from .brdp_group_accountant import (
    ACCOUNTANT_VERSION,
    GroupPrivacyLedger,
)
from .brdp_source_protocol import (
    PersistentSourceClient,
    actual_round_costs,
    clip_source_update,
    effective_shapes,
    layer_log_rows,
    load_datasets,
    partition_and_groups,
    public_training_data_config,
    preflight,
    release_source_sum,
    validate_config,
)
from .data import split_checksum
from .layer_registry import LayerRegistry
from .metrics import evaluate
from .st_runner import write_csv
from .upstream_runner import SOURCE, source_model


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.02)


def _private_rngs(device):
    return (
        np.random.default_rng(secrets.randbits(128)),
        np.random.default_rng(secrets.randbits(128)),
        torch.Generator(device=device).manual_seed(secrets.randbits(63)),
    )


def run(config, output, operational_progress=None):
    cfg = validate_config(config)
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = cfg["device"]
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    train, test = load_datasets(cfg)
    if "public_train_file" in cfg:
        from .dvpa_public_training import partition_record, evaluation_record
        data_cfg = public_training_data_config(cfg)
        _json(out / "data_partition_protocol.json", partition_record(data_cfg))
        evaluation = evaluation_record(data_cfg, cfg["public_train_sha256"])
        if len(train) != (60000 - cfg.get("public_train_count", 1000)) or len(test) != 10000:
            raise ValueError("Training-public baseline has unexpected client / test sizes")
    else:
        from .dvpa_dataset_support import baseline_evaluation
        test, evaluation = baseline_evaluation(cfg, test)
    if evaluation is not None:
        _json(out / "evaluation_protocol.json", evaluation)
    parts, group_ids = partition_and_groups(cfg, train.tensors[1])
    _json(
        out / "partition.json",
        dict(
            parts=parts,
            group_ids=group_ids.tolist(),
            split_checksum=split_checksum(parts),
            scope="frozen public benchmark partition; neighbor comparisons start after assignment",
        ),
    )
    # Author initialization order: initialize every persistent client, then the
    # global model. Model broadcasts do NOT reset client optimizers/schedulers.
    clients = [
        PersistentSourceClient(train, indices, cfg, device, slot=i)
        for i, indices in enumerate(parts)
    ]
    model = source_model()(cfg["dataset"], cfg["batch_size"]).to(device)
    if list(model.named_buffers()):
        raise ValueError("Only the author buffer-free model is supported")
    registry = LayerRegistry(model)
    if sum(registry.dimensions.values()) != 582026:
        raise ValueError("Author model parameter signature changed")
    public = preflight(cfg, registry.dimensions)
    ledger = GroupPrivacyLedger(
        public["orders"], cfg["epsilons"], cfg["delta"], public["envelope"]
    )
    direction = DirectionState(registry.dimensions)
    sample_rng, scalar_rng, noise_rng = _private_rngs(device)
    torch.save(
        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        out / "initial_model.pt",
    )
    source_snapshot = out / "source_snapshot"
    source_snapshot.mkdir()
    manifest = {}
    for p in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(p, source_snapshot / p.name)
        manifest[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    author_manifest = {}
    for p in SOURCE.rglob("*.py"):
        relative = p.relative_to(SOURCE)
        dest = out / "upstream_source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        author_manifest[relative.as_posix()] = hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
    _json(
        out / "implementation.json",
        dict(
            accountant=ACCOUNTANT_VERSION,
            source_sha256=manifest,
            author_source_sha256=author_manifest,
            config_sha256=hashlib.sha256(
                (out / "config.yaml").read_bytes()
            ).hexdigest(),
            torch=torch.__version__,
            device=str(device),
            parameter_count=582026,
            trust_model="trusted_server_aggregate_only",
            privacy_unit="fixed client slot add/remove contribution",
            sampling_bound="Gaussian joint sampled mechanism; monotone D2 bound below order2"
            if cfg["method"] in {"SPEND_AS_YOU_GO", "IDP_FEDAVG"}
            else "minimum of SGM-reference density sandwich and joint Bernoulli convexity",
            opacus_usage="Gaussian-only SGM RDP plus generic conversion"
            if cfg["method"] in {"SPEND_AS_YOU_GO", "IDP_FEDAVG"}
            else "generic RDP conversion only; no direct SGM billing for non-Gaussian noise",
            optimizer="persistent SGD(momentum=0.9, weight_decay=5e-4)",
            lr_schedule="persistent CosineAnnealingLR(T_max=200); step once per local epoch",
            training_population="all clients train each round, including unselected clients",
            shuffle_rng="public per-slot persistent generators; isolates other clients from private-data-dependent RNG consumption",
            update_units="u=(local-global)/initial_lr*rate; release=sum(clipped u)*initial_lr*rate/sum(q)+noise",
            guarantee_scope="ideal real-valued distributions; research floating-point implementation",
            data_split="clean official train/test; source train/test merging is not reproduced",
        ),
    )
    _json(
        out / "privacy_preflight.json",
        dict(
            accountant=ACCOUNTANT_VERSION,
            group_counts=public["counts"].tolist(),
            targets=cfg["epsilons"],
            delta=cfg["delta"],
            orders=public["orders"].tolist(),
            beta=public["betas"].tolist(),
            kappa=public["kappa"],
            group_clip_raw=public["group_radii"].tolist(),
            weighted_mean_clip=float(
                np.dot(public["counts"] / cfg["clients"], public["group_radii"])
            ),
            final_epsilon_uniform_envelope=public["final_epsilon"],
            plans=[p.as_dict() for p in public["plans"]],
            schedule_metadata=public.get("schedule_metadata"),
        ),
    )
    np.savez(
        out / "privacy_envelope.npz",
        orders=public["orders"],
        envelope=public["envelope"],
    )
    (out / "released").mkdir()
    rows, layer_rows, group_rows = [], [], []
    actual_increments = []
    started = time.perf_counter()
    for t, plan in enumerate(public["plans"], 1):
        directions = direction.for_round(t)
        shapes = effective_shapes(plan, directions)
        cost = actual_round_costs(
            cfg,
            public["orders"],
            plan.group_radii,
            plan.final_scale,
            dict(plan.layer_clip_weights),
            dict(plan.layer_noise_std),
            plan.group_rates,
            shapes,
        )
        charged = ledger.accept(t, cost)
        actual_increments.append(cost)
        layer_rows.extend(layer_log_rows(plan, registry.dimensions, directions, shapes))
        for g, row in enumerate(charged):
            group_rows.append(
                {
                    **row,
                    "clients": int(public["counts"][g]),
                    "sample_rate": plan.group_rates[g],
                    "clip_raw": plan.group_radii[g],
                    "denominator": plan.denominator,
                }
            )
        write_csv(out / "group_privacy.csv", group_rows)
        write_csv(out / "layers.csv", layer_rows)
        private_sum = {
            g: torch.zeros(d, device=device) for g, d in registry.dimensions.items()
        }
        probabilities = np.asarray(plan.group_rates)[group_ids]
        selected = sample_rng.random(cfg["clients"]) < probabilities
        for i, client in enumerate(clients):
            update = client.train(model, registry, return_update=bool(selected[i]))
            if update is not None:
                clipped = clip_source_update(
                    update, plan.group_radii[int(group_ids[i])], plan
                )
                for g in registry.names:
                    private_sum[g].add_(clipped[g])
                del update, clipped
            if operational_progress is not None:
                progress_path = Path(operational_progress)
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    _json(
                        progress_path,
                        dict(
                            scope="trusted local operational telemetry; exclude from DP transcript",
                            output=str(out),
                            round=t,
                            rounds=cfg["rounds"],
                            phase="local_training",
                            clients_processed=i + 1,
                            clients_total=cfg["clients"],
                            elapsed_seconds=time.perf_counter() - started,
                        ),
                    )
                except PermissionError:
                    pass  # A status reader must not abort a sensitive run.
        released = release_source_sum(
            private_sum, plan, directions, shapes, scalar_rng, noise_rng
        )
        del private_sum, selected
        registry.apply(model, released)
        direction.observe(t, released)
        published = {g: v.detach().cpu().clone() for g, v in released.items()}
        torch.save(published, out / "released" / f"round_{t:03d}.pt")
        torch.save(model.state_dict(), out / "model.pt")
        loss, acc = evaluate(model, test, device)
        rows.append(
            dict(
                round=t,
                test_accuracy=acc,
                test_loss=loss,
                denominator=plan.denominator,
                mean_sample_rate=plan.denominator / cfg["clients"],
                local_epochs_completed=clients[0].epochs_completed,
                scheduler_last_epoch=clients[0].scheduler.last_epoch,
                next_local_lr=clients[0].optimizer.param_groups[0]["lr"],
                non_gaussian_layers=sum(r > 0 for r, _ in shapes.values()),
                epsilon_group_10=charged[0]["epsilon_guarantee"],
                epsilon_group_20=charged[1]["epsilon_guarantee"],
                epsilon_group_30=charged[2]["epsilon_guarantee"],
            )
        )
        write_csv(out / "rounds.csv", rows)
        torch.save(direction.state_dict(), out / "directions_state.pt")
        np.savez(
            out / "accountant.npz",
            orders=public["orders"],
            actual_increments=np.asarray(actual_increments),
            actual_rdp=ledger.actual,
            uniform_rdp=ledger.uniform,
            targets=np.asarray(cfg["epsilons"]),
        )
        _json(
            out / "run_state.json",
            dict(status="running", completed_rounds=t, total_rounds=cfg["rounds"]),
        )
        print(
            f"SOURCE-BRDP {cfg['mechanism']} seed={cfg['seed']} round={t}/{cfg['rounds']} "
            f"accuracy={acc:.2f}% groups_eps={[round(r['epsilon_guarantee'], 6) for r in charged]}",
            flush=True,
        )
    summary = dict(
        status="completed",
        seed=cfg["seed"],
        mechanism=cfg["mechanism"],
        method=cfg["method"],
        rounds=len(rows),
        parameters=582026,
        dataset_scope="synthetic engineering"
        if cfg.get("dataset_mode") == "synthetic"
        else f"full public {cfg['dataset'].upper()} benchmark",
        privacy_unit="fixed client slot add/remove contribution",
        trust_model="trusted_server_aggregate_only",
        accountant=ACCOUNTANT_VERSION,
        group_counts=public["counts"].tolist(),
        group_privacy=group_rows[-3:],
        split_checksum=split_checksum(parts),
        initial_model_sha256=hashlib.sha256(
            (out / "initial_model.pt").read_bytes()
        ).hexdigest(),
        evaluation_examples=len(test),
        final_test_accuracy=rows[-1]["test_accuracy"],
        best_test_accuracy=max(r["test_accuracy"] for r in rows),
        non_gaussian_layer_releases=sum(r["recycle_prob"] > 0 for r in layer_rows),
        total_client_local_epochs=cfg["clients"] * cfg["rounds"] * cfg["local_epochs"],
        guarantee_scope="single-run uniform privacy envelope; actual-path conversion is a diagnostic",
    )
    client_rows = [
        dict(
            client_slot=i,
            group=int(group_ids[i]),
            epsilon_target=cfg["epsilons"][int(group_ids[i])],
            delta=cfg["delta"],
            epsilon_guarantee=group_rows[-3 + int(group_ids[i])]["epsilon_guarantee"],
            epsilon_actual_path_accounting=group_rows[-3 + int(group_ids[i])][
                "epsilon_actual_path_accounting"
            ],
        )
        for i in range(cfg["clients"])
    ]
    if "public_train_file" in cfg:
        summary["client_training_examples"] = (60000 - cfg.get("public_train_count", 1000))
        summary["public_source"] = "official_training_split"
    write_csv(out / "client_privacy.csv", client_rows)
    _json(out / "summary.json", summary)
    _json(
        out / "run_state.json",
        dict(
            status="completed", completed_rounds=len(rows), total_rounds=cfg["rounds"]
        ),
    )
    if operational_progress is not None:
        _json(
            Path(operational_progress),
            dict(
                scope="trusted local telemetry",
                status="completed",
                round=len(rows),
                elapsed_seconds=time.perf_counter() - started,
            ),
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--operational-progress")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(run(cfg, args.output, args.operational_progress), indent=2))


if __name__ == "__main__":
    main()
