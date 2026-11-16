"""Dataset routing for the frozen DVPA mechanism and common evaluation splits.

The original FMNIST empirical certificates remain unchanged. MNIST is an
explicit fixed-hyperparameter transfer: numerical/data-isolation validation
is recorded separately and is never called an empirical benefit certificate.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from .dvpa_public_proxy import PublicCurvatureProxy as FMNISTProxy
from .dvpa_public_proxy import curvature_traces, tensor_sha256


def read_package(dataset, filename, expected_sha):
    from .dvpa_config import ROOT

    path = (ROOT / filename).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Public package must stay inside the DVPA project")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != expected_sha:
        raise ValueError("Public data hash mismatch")
    data = torch.load(path, weights_only=True, map_location="cpu")
    ids = data["public_source_indices"].tolist()
    final = data["final_indices"].tolist()
    if (
        dataset not in {"fmnist", "mnist"}
        or data["metadata"]["dataset"] != dataset.upper() + " official test split"
        or data["metadata"]["private_training_examples_used"]
        or len(ids) != 1000
        or len(final) != 9000
        or sorted(ids + final) != list(range(10000))
        or data["indices"]["calibration"].tolist() != list(range(64))
        or data["images"].shape != (1000, 1, 28, 28)
        or not torch.isfinite(data["images"]).all()
        or data["labels"].shape != (1000,)
    ):
        raise ValueError("Public dataset identity or disjoint split mismatch")
    return path, sha, data


def make_public_mnist(path, data_root):
    from .data import idx, normalize_images

    folder = Path(data_root) / "fashion-mnist-master/data/mnist"
    if not (folder / "t10k-images-idx3-ubyte.gz").exists() and not (folder / "t10k-images-idx3-ubyte").exists():
        folder = Path(data_root) / "MNIST/raw"
    image = folder / "t10k-images-idx3-ubyte.gz"
    label = folder / "t10k-labels-idx1-ubyte.gz"
    if not image.exists():
        image = image.with_suffix("")
    if not label.exists():
        label = label.with_suffix("")
    x = normalize_images(idx(image, 16).reshape(-1, 1, 28, 28)) * 2 - 1
    y = torch.from_numpy(idx(label, 8)).long()
    if len(x) != 10000 or len(y) != 10000:
        raise ValueError("Full official MNIST test split required")
    order = np.random.default_rng(20261031).permutation(10000)
    public, final = order[:1000], order[1000:]
    data = dict(
        images=x[public], labels=y[public],
        indices=dict(calibration=torch.arange(64), validation=torch.arange(64, 192),
                     confirmation=torch.arange(192, 320)),
        public_source_indices=torch.tensor(public), final_indices=torch.tensor(final),
        metadata=dict(
            dataset="MNIST official test split", split_seed=20261031,
            public_examples=1000, final_examples=9000,
            image_file_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            label_file_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
            input_sha256=tensor_sha256(x[public]), private_training_examples_used=False,
            original_test_has_historical_evaluations=True,
        ),
    )
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data


class PublicCurvatureProxy(FMNISTProxy):
    def __init__(self, config):
        if config["base"]["dataset"] == "fmnist":
            super().__init__(config)
            return
        self.cfg = config["feedback"]
        self.path, self.sha, data = read_package(
            config["base"]["dataset"], self.cfg["public_proxy_file"],
            self.cfg["public_proxy_sha256"],
        )
        self.images = data["images"][data["indices"]["calibration"]].clone()
        self.final_indices = data["final_indices"].tolist()
        self.metadata = data["metadata"]


def baseline_evaluation(cfg, test):
    """Same held-out 9k as DVPA; never change the training dataset."""
    if "evaluation_public_file" not in cfg or cfg.get("dataset_mode") == "synthetic":
        return test, None
    _, sha, data = read_package(
        cfg["dataset"], cfg["evaluation_public_file"], cfg["evaluation_public_sha256"]
    )
    if len(test) != 10000:
        raise ValueError("Full test set required before public reservation")
    ids = data["public_source_indices"]
    if not torch.equal(test.tensors[0][ids], data["images"]) or not torch.equal(
        test.tensors[1][ids], data["labels"]
    ):
        raise ValueError("Reserved public examples differ from the actual test dataset")
    return Subset(test, data["final_indices"].tolist()), dict(
        dataset=cfg["dataset"].upper(), public_examples=1000,
        final_evaluation_examples=9000, source_test_examples=10000,
        public_data_sha256=sha, original_test_has_historical_evaluations=True,
        scope="same 9000 held-out examples for all three methods; public data available to all",
    )


def create_transfer_validation(config):
    """Numerical validation only; freeze the FMNIST-chosen parameters on MNIST."""
    from .dvpa_config import ROOT
    from .dvpa_forecast import forecast_coefficients, parameters_from_config
    from .upstream_runner import source_model

    if config["base"]["dataset"] != "mnist":
        raise ValueError("This certificate is explicitly for MNIST transfer")
    torch.set_num_threads(1)
    proxy = PublicCurvatureProxy(config)
    histories = []
    for seed in (42, 43, 44):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = source_model()("mnist", 125)
        histories.append(curvature_traces(model, proxy.images, probes=16)["coefficients"])
    parameters = parameters_from_config(config["feedback"])
    predictions = forecast_coefficients(histories, np.arange(6), **parameters)
    # Prefix causality: a future appended observation cannot alter a stored forecast.
    prefix = forecast_coefficients(histories[:2], np.arange(4), **parameters)
    repeated = forecast_coefficients(histories[:2], np.arange(4), **parameters)
    checks = dict(
        positive_curvature=bool((np.array(histories) > 0).all()),
        scale_preserved=bool(np.allclose(predictions.sum(1), 582026)),
        current_exact=bool(np.array_equal(predictions[0], histories[-1])),
        finite_positive_forecast=bool(np.isfinite(predictions).all() and (predictions > 0).all()),
        horizon_endpoint_held=bool(np.array_equal(predictions[3], predictions[5])),
        prefix_deterministic=bool(np.array_equal(prefix, repeated)),
    )
    if not all(checks.values()):
        raise ValueError("MNIST fixed-transfer numerical validation failed")
    return dict(
        status="completed", validation_kind="frozen_mnist_transfer_numerical",
        dataset="mnist", public_data_sha256=proxy.sha,
        support_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        forecast_module_sha256=hashlib.sha256((ROOT / "src/dvpa_forecast.py").read_bytes()).hexdigest(),
        proxy_module_sha256=hashlib.sha256((ROOT / "src/dvpa_public_proxy.py").read_bytes()).hexdigest(),
        parameters=parameters, probes=16,
        signal_settings={k: config["feedback"][k] for k in ("lambda0", "lambda1", "signal_scale")},
        checks=checks, empirical_benefit_validated=False,
        scope="Frozen FMNIST hyperparameters transferred to MNIST; public-input/numerical checks only, no claim of proxy benefit or forecast superiority on MNIST",
    )


def _verify_transfer(config, proof=None):
    from .dvpa_config import ROOT
    from .dvpa_forecast import parameters_from_config

    f = config["feedback"]
    if proof is None:
        proof = json.loads((ROOT / f["public_proxy_validation"]).read_text(encoding="utf-8"))
    expected = dict(
        dataset="mnist", validation_kind="frozen_mnist_transfer_numerical",
        public_data_sha256=f["public_proxy_sha256"],
        support_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        forecast_module_sha256=hashlib.sha256((ROOT / "src/dvpa_forecast.py").read_bytes()).hexdigest(),
        proxy_module_sha256=hashlib.sha256((ROOT / "src/dvpa_public_proxy.py").read_bytes()).hexdigest(),
        parameters=parameters_from_config(f), probes=f["public_proxy_probes"],
        signal_settings={k: f[k] for k in ("lambda0", "lambda1", "signal_scale")},
    )
    if (
        config["base"]["dataset"] != "mnist" or proof.get("status") != "completed"
        or any(proof.get(k) != v for k, v in expected.items())
        or set(proof.get("checks", {})) != {"positive_curvature", "scale_preserved", "current_exact",
            "finite_positive_forecast", "horizon_endpoint_held", "prefix_deterministic"}
        or not all(proof["checks"].values())
        or proof.get("empirical_benefit_validated") is not False
        or f["extra_covariance"] != "dp_innovation"
        or expected["parameters"] != dict(ema=0.25, damping=0.75, clip=0.1, horizon=3)
    ):
        raise ValueError("Current fixed-transfer verification required")
    PublicCurvatureProxy(config)
    return proof


def verify_public_validation(config):
    if config["base"]["dataset"] == "fmnist":
        from .dvpa_public_proxy import verify_public_validation as original
        return original(config)
    return _verify_transfer(config)


def verify_forecast_validation(config, proof=None):
    if config["base"]["dataset"] == "fmnist":
        from .dvpa_forecast import verify_forecast_validation as original
        return original(config, proof)
    return _verify_transfer(config, proof)
