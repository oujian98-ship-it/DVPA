"""Public-input predictive curvature, independent of private client statistics.

The PSD softmax Fisher/Gauss--Newton trace measures local prediction sensitivity.
It is a utility approximation, not a task-accuracy or privacy guarantee.
"""

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch


def tensor_sha256(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def make_public_fmnist(path, data_root):
    """Reserve 1,000 official test examples; remaining 9,000 are evaluation only."""
    from .data import idx, normalize_images

    folder = Path(data_root) / "fashion-mnist-master/data/fashion"
    image_path = folder / "t10k-images-idx3-ubyte.gz"
    label_path = folder / "t10k-labels-idx1-ubyte.gz"
    x = normalize_images(idx(image_path, 16).reshape(-1, 1, 28, 28)) * 2 - 1
    y = torch.from_numpy(idx(label_path, 8)).long()
    if len(x) != 10000 or len(y) != 10000:
        raise ValueError("Full official FMNIST test split required")
    order = np.random.default_rng(20261031).permutation(10000)
    public_ids, final_ids = order[:1000], order[1000:]
    splits = dict(
        calibration=torch.arange(64),
        validation=torch.arange(64, 192),
        confirmation=torch.arange(192, 320),
    )
    package = dict(
        images=x[public_ids],
        labels=y[public_ids],
        indices=splits,
        public_source_indices=torch.tensor(public_ids),
        final_indices=torch.tensor(final_ids),
        metadata=dict(
            dataset="FMNIST official test split",
            split_seed=20261031,
            public_examples=1000,
            final_examples=9000,
            image_file_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
            label_file_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(),
            input_sha256=tensor_sha256(x[public_ids]),
            private_training_examples_used=False,
            original_test_has_historical_evaluations=True,
        ),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Public data artifact already exists")
    torch.save(package, path)
    return package


def curvature_traces(model, images, probes=16, seed=20261030):
    """E||J_l^T v||^2 = tr(J_l^T (diag(p)-pp^T) J_l).

    Float64 CPU and a local generator give reproducible replay without touching
    training RNG state. No labels, raw gradients, or private examples are read.
    """
    if images.ndim != 4 or not len(images) or probes < 2:
        raise ValueError("Nonempty public images and >=2 probes required")
    net = copy.deepcopy(model).cpu().double().eval()
    names, params = zip(*net.named_parameters(), strict=True)
    groups = list(dict.fromkeys(n.split(".")[0] for n in names))
    values = np.zeros((probes, len(groups)))
    logits = net(images.detach().cpu().double())
    probabilities = logits.detach().softmax(-1)
    rng = torch.Generator().manual_seed(seed)
    for k in range(probes):
        noise = torch.randn(logits.shape, dtype=torch.float64, generator=rng)
        v = probabilities.sqrt() * noise
        v = (v - probabilities * v.sum(-1, keepdim=True)) / len(images) ** 0.5
        gradients = torch.autograd.grad(
            logits, params, grad_outputs=v, retain_graph=k < probes - 1
        )
        for name, grad in zip(names, gradients, strict=True):
            values[k, groups.index(name.split(".")[0])] += float(grad.square().sum())
    means = np.maximum(values.mean(0), 1e-12)
    dimensions = {
        g: sum(
            p.numel()
            for n, p in zip(names, params, strict=True)
            if n.split(".")[0] == g
        )
        for g in groups
    }
    # Keep the total coefficient scale unchanged; alter allocation across layers.
    coefficients = means / means.sum() * sum(dimensions.values())
    return dict(
        names=groups,
        traces=means.tolist(),
        standard_errors=(values.std(0, ddof=1) / probes**0.5).tolist(),
        coefficients=coefficients.tolist(),
        probes=probes,
        seed=seed,
        model_sha256={n: tensor_sha256(p) for n, p in net.state_dict().items()},
    )


class PublicCurvatureProxy:
    def __init__(self, config):
        from .dvpa_config import ROOT

        self.cfg = config["feedback"]
        self.path = (ROOT / self.cfg["public_proxy_file"]).resolve()
        if not self.path.is_relative_to(ROOT):
            raise ValueError("Public input outside BR_DP")
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.sha != self.cfg["public_proxy_sha256"]:
            raise ValueError("Public data hash mismatch")
        data = torch.load(self.path, map_location="cpu", weights_only=True)
        ids = data["public_source_indices"].tolist()
        final = data["final_indices"].tolist()
        if (
            sorted(ids + final) != list(range(10000))
            or len(ids) != 1000
            or len(final) != 9000
        ):
            raise ValueError(
                "Public/final evaluation split is not disjoint and complete"
            )
        if (
            data["metadata"]["dataset"] != "FMNIST official test split"
            or data["metadata"]["private_training_examples_used"]
        ):
            raise ValueError("Only designated public FMNIST input is allowed")
        if (
            data["images"].shape != (1000, 1, 28, 28)
            or not torch.isfinite(data["images"]).all()
        ):
            raise ValueError("Invalid public input")
        calibration = data["indices"]["calibration"].tolist()
        if calibration != list(range(64)):
            raise ValueError("Fixed public calibration IDs required")
        self.images = data["images"][calibration].clone()
        self.final_indices = final
        self.metadata = data["metadata"]

    def report(self, model, round_index):
        result = curvature_traces(model, self.images, self.cfg["public_proxy_probes"])
        result.update(
            round=round_index,
            public_data_sha256=self.sha,
            public_examples=64,
            forecast="hold current relative curvature over remaining rounds",
            coefficient_scale="sum equals total parameter count",
        )
        return result


def verify_public_validation(config):
    """Require the actual FMNIST controlled study before a formal variant run."""
    import json
    from .dvpa_config import ROOT

    cfg = config["feedback"]
    if cfg["utility_proxy"] != "public_curvature_signal":
        return None
    path = (ROOT / cfg["public_proxy_validation"]).resolve()
    proof = json.loads(path.read_text(encoding="utf-8"))
    expected = dict(
        public_data_sha256=cfg["public_proxy_sha256"],
        proxy_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        validation_script_sha256=hashlib.sha256(
            (ROOT / "scripts/check_dvpa_proxy.py").read_bytes()
        ).hexdigest(),
    )
    if (
        proof.get("status") != "completed"
        or proof.get("passed") is not True
        or any(proof.get(k) != v for k, v in expected.items())
        or proof["protocol"]["probes"] != cfg["public_proxy_probes"]
        or cfg["extra_covariance"] != "dp_innovation"
        or any(
            cfg[k] != proof["protocol"]["signal_settings"][k]
            for k in ("lambda0", "lambda1", "signal_scale")
        )
    ):
        raise ValueError(
            "Current FMNIST proxy benefit validation required before formal training"
        )
    held = [
        v
        for v in proof["checks"]
        if v["split"] == "confirmation" and v["metric"] == "cross_entropy"
    ]
    if sorted(v["round"] for v in held) != [4, 13, 21] or any(
        v["correlations"]["public_curvature_signal"] < 0.5 or v["margin"] <= 0
        for v in held
    ):
        raise ValueError("FMNIST proxy failed prespecified confirmation conditions")
    return proof
