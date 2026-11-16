"""FMNIST-only controlled benefit verification using already released models."""

import json
import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

from src.brdp_source_runner import _json
from src.dvpa_config import ROOT, load_config, public_preflight
from src.dvpa_controller import spatial_optimum
from src.dvpa_feedback import DPHistoryFeedback
from src.dvpa_filter import GroupRDPFilter, PrecisionAction
from src.dvpa_public_proxy import curvature_traces, make_public_fmnist
from src.layer_registry import LayerRegistry
from src.upstream_runner import source_model


def candidates(old, new):
    vectors = [np.full(4, 0.25), old, new]
    for i in range(4):
        for j in range(4):
            if i == j or old[i] - 0.03 < 0.02:
                continue
            p = old.copy()
            p[i] -= 0.03
            p[j] += 0.03
            vectors.append(p)
    return np.array(vectors)


def measure(model, x, y, allocations, std_base, draws, seed):
    """Each +/- Gaussian pair is ONE independent Monte Carlo sample."""
    net = model.eval()
    reg = LayerRegistry(net)
    params = dict(net.named_parameters())
    original = {k: v.detach().clone() for k, v in net.state_dict().items()}
    gen = torch.Generator(device=x.device).manual_seed(seed)
    losses = np.zeros((2, draws, len(allocations)))
    with torch.no_grad():
        logits = net(x)
        teacher = logits.softmax(-1)
        clean = float(F.cross_entropy(logits, y))
        for d in range(draws):
            noise = {
                n: torch.randn(v.shape, device=v.device, generator=gen)
                for n, v in params.items()
            }
            for i, p in enumerate(allocations):
                for sign in (-1, 1):
                    for n, v in params.items():
                        v.copy_(
                            original[n]
                            + sign
                            * noise[n]
                            * std_base
                            / np.sqrt(p[reg.names.index(n.split(".")[0])])
                        )
                    logits = net(x)
                    losses[0, d, i] += float(F.cross_entropy(logits, y) - clean) / 2
                    losses[1, d, i] += (
                        float(
                            F.kl_div(
                                logits.log_softmax(-1), teacher, reduction="batchmean"
                            )
                        )
                        / 2
                    )
        net.load_state_dict(original)
    return losses


def run():
    folder = ROOT / "reports/dvpa_proxy"
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(20261031)
    cfg = load_config(ROOT / "configs/dvpa_fmnist_local_seed42.yaml")
    protocol = public_preflight(cfg)
    path = ROOT / "data/public_fmnist_proxy.pt"
    data = (
        torch.load(path, weights_only=True)
        if path.exists()
        else make_public_fmnist(path, cfg["base"]["data_root"])
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = source_model()("fmnist", 125)
    previous = ROOT / "results/dvpa_local_seed42_20261029"
    model.load_state_dict(torch.load(previous / "initial_model.pt", weights_only=True))
    reg = LayerRegistry(model)
    feedback = DPHistoryFeedback(protocol)
    ledger = GroupRDPFilter(protocol)
    saved = [
        json.loads(v)["certified"]["action"]
        for v in (previous / "actions.jsonl").read_text().splitlines()
    ]
    checks = []
    for t in range(1, 22):
        feedback.prepare(t)
        if t in (4, 13, 21):
            report = curvature_traces(
                model, data["images"][data["indices"]["calibration"]], probes=16
            )
            dim = np.array(list(protocol.dimensions.values()), dtype=float)
            importance = feedback.expected_costs(
                feedback.pred_mean, feedback.pred_cov, [t], origin=t
            )[0, 0] / (dim * protocol.scales[t - 1] ** 2 / 4)
            costs = {
                "legacy_dimension_signal": dim * importance,
                "public_curvature_signal": np.array(report["traces"]) * importance,
            }
            old, new = [spatial_optimum(v, 0.02) for v in costs.values()]
            allocations = candidates(old, new)
            predictions = {
                k: np.sum(v[None, :] / allocations, axis=1) for k, v in costs.items()
            }
            model = model.to(device)
            std = protocol.scales[t - 1] * 0.5 / np.sqrt(protocol.z0)
            for split in ("validation", "confirmation"):
                ids = data["indices"][split]
                measured = measure(
                    model,
                    data["images"][ids].to(device),
                    data["labels"][ids].to(device),
                    allocations,
                    std,
                    64,
                    20261031 + t,
                )
                for metric, losses in zip(
                    ("cross_entropy", "predictive_kl"), measured, strict=True
                ):
                    gain = losses[:, 1] - losses[:, 2]
                    se = float(gain.std(ddof=1) / len(gain) ** 0.5)
                    row = dict(
                        round=t,
                        metric=metric,
                        split=split,
                        draws=64,
                        antithetic=True,
                        noise_std_base=std,
                        signal_importance=importance.tolist(),
                        coefficients=report,
                        allocations=allocations.tolist(),
                        correlations={
                            k: float(spearmanr(v, losses.mean(0)).statistic)
                            for k, v in predictions.items()
                        },
                        new_vs_old_gain=float(gain.mean()),
                        paired_se=se,
                        margin=float(gain.mean() - 3 * se),
                        measured_means=losses.mean(0).tolist(),
                        paired_losses=losses.tolist(),
                    )
                    checks.append(row)
                    _json(
                        folder / "fmnist_validation.json",
                        dict(status="running", checks=checks),
                    )
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in row.items()
                                if k
                                in [
                                    "round",
                                    "metric",
                                    "split",
                                    "correlations",
                                    "new_vs_old_gain",
                                    "paired_se",
                                    "margin",
                                ]
                            }
                        ),
                        flush=True,
                    )
            model = model.cpu()
        item = saved[t - 1]
        action = PrecisionAction(
            t,
            item["precision"],
            tuple(tuple(v) for v in item["layer_precision"]),
            item["feedback_basis_hash"],
            item["history_hash"],
        )
        cert = ledger.certify(action)
        ledger.reserve(cert)
        released = torch.load(
            previous / "released" / f"round_{t:03d}.pt", weights_only=True
        )
        reg.apply(model, released)
        feedback.observe(t, released, cert)
    held = [
        v
        for v in checks
        if v["split"] == "confirmation" and v["metric"] == "cross_entropy"
    ]
    passed = len(held) == 3 and all(
        v["correlations"]["public_curvature_signal"] >= 0.5 and v["margin"] > 0
        for v in held
    )
    result = dict(
        status="completed",
        passed=passed,
        public_data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        proxy_module_sha256=hashlib.sha256(
            (ROOT / "src/dvpa_public_proxy.py").read_bytes()
        ).hexdigest(),
        validation_script_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        protocol=dict(
            dataset=data["metadata"],
            calibration=64,
            validation=128,
            confirmation=128,
            checkpoints=[4, 13, 21],
            draws=64,
            antithetic=True,
            probes=16,
            signal_settings={
                k: cfg["feedback"][k] for k in ("lambda0", "lambda1", "signal_scale")
            },
            public_seed=20261031,
            acceptance="All three CE confirmation cases: Spearman >=0.5 and gain-3SE>0; KL secondary.",
            scope="FMNIST-only public Gaussian channel benefit at fixed total precision, no raw private client query; does not prove FL accuracy or time benefit",
        ),
        checks=checks,
    )
    _json(folder / "fmnist_validation.json", result)
    print(json.dumps(dict(status="completed", passed=passed)), flush=True)


if __name__ == "__main__":
    run()
