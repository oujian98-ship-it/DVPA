"""Rolling forecasts: develop on old seeds42/43, confirm on44 and proxy42."""

import hashlib
import itertools
import json

import numpy as np
import torch

from src.brdp_source_runner import _json
from src.dvpa_config import ROOT, load_config
from src.dvpa_controller import spatial_optimum
from src.dvpa_forecast import centered_log, forecast_coefficients
from src.dvpa_public_proxy import PublicCurvatureProxy, curvature_traces
from src.layer_registry import LayerRegistry
from src.upstream_runner import source_model


def collect():
    proxy = PublicCurvatureProxy(
        load_config(ROOT / "configs/dvpa_fmnist_proxy_seed42.yaml")
    )
    sources = {
        f"old_seed{s}": ROOT
        / f"results/dvpa_vs_idp_sayg_seeds42_43_44_20261028/DVPA_seed{s}"
        for s in (42, 43, 44)
    }
    sources["proxy_seed42"] = ROOT / "results/dvpa_proxy_seed42_20261029"
    cache = ROOT / "reports/dvpa_forecast/curvature_histories.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / "src/dvpa_public_proxy.py", proxy.path]
    for r in sources.values():
        paths += [r / "initial_model.pt"] + [
            r / "released" / f"round_{t:03d}.pt" for t in range(1, 25)
        ]
    paths += sorted((sources["proxy_seed42"] / "public_proxy").glob("round_*.json"))
    manifest = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["manifest"] == manifest:
            return saved
    torch.set_num_threads(1)
    torch.manual_seed(20261031)
    histories = {}
    for name, r in sources.items():
        model = source_model()("fmnist", 125)
        model.load_state_dict(torch.load(r / "initial_model.pt", weights_only=True))
        registry = LayerRegistry(model)
        values = []
        for t in range(1, 26):
            if name == "proxy_seed42":
                report = json.loads(
                    (r / "public_proxy" / f"round_{t:03d}.json").read_text()
                )
            else:
                report = curvature_traces(model, proxy.images, probes=16)
            values.append(report["coefficients"])
            if t < 25:
                registry.apply(
                    model,
                    torch.load(r / "released" / f"round_{t:03d}.pt", weights_only=True),
                )
        histories[name] = values
        print(json.dumps(dict(collected=name, rounds=25)), flush=True)
    saved = dict(manifest=manifest, histories=histories, public_data_sha256=proxy.sha)
    _json(cache, saved)
    return saved


def evaluate(values, params):
    values = np.array(values)
    rows = []
    for origin in range(3, len(values) - 1):
        available = min(3, len(values) - 1 - origin)
        steps = np.arange(1, available + 1)
        predicted = forecast_coefficients(values[: origin + 1], steps, **params)
        actual = values[origin + steps]
        frozen = np.broadcast_to(values[origin], actual.shape)
        oracle_p = spatial_optimum(actual, 0.02)
        oracle = (actual / oracle_p).sum(axis=1)

        def regret(x):
            return (actual / spatial_optimum(x, 0.02)).sum(axis=1) / oracle - 1

        for j, step in enumerate(steps):
            rows.append(
                dict(
                    origin_round=origin + 1,
                    horizon=int(step),
                    forecast_regret=float(regret(predicted)[j]),
                    hold_regret=float(regret(frozen)[j]),
                    forecast_clr_mse=float(
                        np.mean(
                            (centered_log(predicted)[j] - centered_log(actual)[j]) ** 2
                        )
                    ),
                    hold_clr_mse=float(
                        np.mean(
                            (centered_log(frozen)[j] - centered_log(actual)[j]) ** 2
                        )
                    ),
                )
            )

    def aggregate(items):
        return {
            key: float(np.mean([v[key] for v in items]))
            for key in [
                "forecast_regret",
                "hold_regret",
                "forecast_clr_mse",
                "hold_clr_mse",
            ]
        }

    return dict(
        one_step=aggregate([v for v in rows if v["horizon"] == 1]),
        three_step_pool=aggregate(rows),
        rows=rows,
    )


def run():
    data = collect()
    hist = data["histories"]
    candidates = []
    for ema, damping in itertools.product(
        (0.25, 0.5, 0.75, 1.0), (0.25, 0.5, 0.75, 1.0)
    ):
        params = dict(ema=ema, damping=damping, clip=0.1, horizon=3)
        dev = [evaluate(hist[k], params) for k in ["old_seed42", "old_seed43"]]
        score = float(np.mean([v["three_step_pool"]["forecast_regret"] for v in dev]))
        candidates.append(dict(parameters=params, development_score=score))
    chosen = min(candidates, key=lambda x: x["development_score"])
    # Parameters are fixed before accessing either confirmation score.
    result = dict(
        status="running",
        parameters=chosen["parameters"],
        development=candidates,
        development_sources=["old_seed42", "old_seed43"],
        confirmation_sources=["old_seed44", "proxy_seed42"],
        public_data_sha256=data["public_data_sha256"],
        history_manifest=data["manifest"],
        checks={},
    )
    output = ROOT / "reports/dvpa_forecast/validation.json"
    _json(output, result)
    for key in result["confirmation_sources"]:
        result["checks"][key] = evaluate(hist[key], chosen["parameters"])
    passed = True
    for name, v in result["checks"].items():
        a, b = v["one_step"], v["three_step_pool"]
        accepted = (
            a["forecast_regret"] < a["hold_regret"]
            and a["forecast_clr_mse"] < a["hold_clr_mse"]
            and b["forecast_regret"] <= b["hold_regret"]
        )
        passed &= accepted
        print(
            json.dumps(
                dict(source=name, one_step=a, three_step_pool=b, passed=accepted)
            ),
            flush=True,
        )
    result.update(
        status="completed",
        passed=bool(passed),
        acceptance="Both confirmation histories: lower one-step relative allocation regret and CLR MSE, no worse pooled horizons1..3 regret.",
        forecast_module_sha256=hashlib.sha256(
            (ROOT / "src/dvpa_forecast.py").read_bytes()
        ).hexdigest(),
        script_sha256=hashlib.sha256(
            __import__("pathlib").Path(__file__).read_bytes()
        ).hexdigest(),
        scope="Conditional prediction of public relative curvature using previous released models only; not true private utility or task gain.",
    )
    _json(output, result)
    print(
        json.dumps(
            dict(
                status="completed", passed=bool(passed), parameters=result["parameters"]
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
