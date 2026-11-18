"""Reserve 64 official test images for curvature and score the complement."""

import hashlib
import json
import os
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch.utils.data import Subset

from .data import load_mnist_split
from .dvpa_public_proxy import PublicCurvatureProxy as LegacyProxy


PUBLIC_COUNT = 64
TEST_COUNT = 10000
SPLIT_SEED = 20261106
EVALUATION_SPLIT = "official_test_holdout"


def split_indices():
    order = np.random.default_rng(SPLIT_SEED).permutation(TEST_COUNT)
    return order[:PUBLIC_COUNT], np.sort(order[PUBLIC_COUNT:])


def read_package(base):
    from .dvpa_config import ROOT

    path = (ROOT / base["evaluation_public_file"]).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Public package must stay inside the DVPA project")
    payload = path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    if sha != base["evaluation_public_sha256"]:
        raise ValueError("Public test package hash mismatch")
    data = torch.load(BytesIO(payload), weights_only=True, map_location="cpu")
    public, held = split_indices()
    if (data["dataset"] != base["dataset"] or data["split_seed"] != SPLIT_SEED
            or not torch.equal(data["public_source_indices"], torch.from_numpy(public))
            or not torch.equal(data["final_indices"], torch.from_numpy(held))
            or data["images"].shape != (PUBLIC_COUNT, 1, 28, 28)
            or not torch.isfinite(data["images"]).all()):
        raise ValueError("Public test64 identity or disjoint split mismatch")
    return path, sha, data


def configure(config):
    """Bind a fresh configuration to the fixed public package for its dataset."""
    from .dvpa_config import ROOT

    base, feedback = config["base"], config["feedback"]
    filename = f"data/public_{base['dataset']}_test64.pt"
    path = ROOT / filename
    if not path.exists():
        test = load_mnist_split(base["data_root"], base["dataset"], "test")
        if len(test) != TEST_COUNT:
            raise ValueError("Full official test set required for public reservation")
        public, held = split_indices()
        data = dict(dataset=base["dataset"], split_seed=SPLIT_SEED,
                    public_source_indices=torch.from_numpy(public), final_indices=torch.from_numpy(held),
                    images=test.tensors[0][public] * 2 - 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, suffix=".pt.tmp", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with temporary.open("wb") as handle:
                torch.save(data, handle)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    for key in ("public_train_count", "public_train_file", "public_train_sha256", "validation_count"):
        base.pop(key, None)
    base.update(public_test_count=PUBLIC_COUNT, evaluation_public_file=filename,
                evaluation_public_sha256=sha)
    feedback.update(public_proxy_file=filename, public_proxy_sha256=sha,
                    public_proxy_validation=None, forecast_validation=None)
    read_package(base)
    return config


class PublicCurvatureProxy(LegacyProxy):
    def __init__(self, config):
        self.cfg = config["feedback"]
        self.path, self.sha, data = read_package(config["base"])
        self.images = data["images"]


def is_official(base):
    return "public_test_count" in base and base.get("dataset_mode") != "synthetic"


def evaluation_record(base):
    _, sha, data = read_package(base)
    return dict(dataset=base["dataset"].upper(), evaluation_split=EVALUATION_SPLIT,
                public_source="official_test_split", public_examples=PUBLIC_COUNT,
                client_training_examples=60000, source_test_examples=TEST_COUNT,
                evaluation_examples=len(data["final_indices"]), public_data_sha256=sha,
                public_test_indices=data["public_source_indices"].tolist())


def held_out_test(base, test):
    _, _, data = read_package(base)
    if len(test) != TEST_COUNT or not torch.equal(test.tensors[0][data["public_source_indices"]], data["images"]):
        raise ValueError("Public test64 images differ from the official test dataset")
    return Subset(test, data["final_indices"].tolist())


def evaluation_data(base, test):
    return (held_out_test(base, test), evaluation_record(base)) if is_official(base) else (test, None)


def verify_evaluation(base, root, summary):
    if not is_official(base):
        return
    expected = evaluation_record(base)
    if json.loads((root / "evaluation_protocol.json").read_text(encoding="utf-8")) != expected:
        raise ValueError("Official test64 split provenance mismatch")
    for key in ("evaluation_split", "public_source", "client_training_examples", "evaluation_examples"):
        if summary.get(key) != expected[key]:
            raise ValueError(f"Official test64 summary mismatch: {key}")
