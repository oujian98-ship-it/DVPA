"""Public proxy correctness, data isolation, sequencing and privacy invariants."""

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from src.dvpa_audit import audit
from src.dvpa_config import ROOT, load_config, public_preflight, resolve, smoke_config
from src.dvpa_controller import DecisionValueController
from src.dvpa_feedback import DPHistoryFeedback
from src.dvpa_filter import GroupRDPFilter
from src.dvpa_public_proxy import (
    PublicCurvatureProxy,
    curvature_traces,
    verify_public_validation,
)
from src.dvpa_runner import run


def configuration():
    return load_config(ROOT / "configs/dvpa_fmnist_proxy_seed42.yaml")


def test_random_trace_matches_exact_predictive_fisher_without_mutating_model_or_rng():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Linear(2, 2)
            self.fc = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.fc(torch.tanh(self.conv1(x.flatten(1))))

    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(9)
        net = Tiny().double()
    x = torch.tensor([[[[0.1, 0.8]]], [[[-0.4, 0.3]]]], dtype=torch.double)
    exact = np.zeros(2)
    groups = ["conv1", "fc"]
    for sample in x:
        logits = net(sample[None])
        p = logits.detach().softmax(-1)[0]
        for label in range(2):
            gradients = torch.autograd.grad(
                logits.log_softmax(-1)[0, label],
                tuple(net.parameters()),
                retain_graph=label == 0,
            )
            for (name, _), g in zip(net.named_parameters(), gradients, strict=True):
                exact[groups.index(name.split(".")[0])] += float(
                    p[label] * g.square().sum()
                ) / len(x)
    before = {n: v.clone() for n, v in net.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    got = curvature_traces(net, x, probes=4096, seed=42)
    assert got["traces"] == pytest.approx(exact, rel=0.04)
    assert sum(got["coefficients"]) == pytest.approx(
        sum(v.numel() for v in net.parameters())
    )
    assert torch.equal(rng, torch.random.get_rng_state()) and net.training
    assert all(torch.equal(before[n], v) for n, v in net.state_dict().items())
    assert all(v.grad is None for v in net.parameters())


def test_public_final_split_and_data_hash_are_pinned():
    cfg = configuration()
    proxy = PublicCurvatureProxy(cfg)
    data = torch.load(proxy.path, weights_only=True)
    assert len(proxy.final_indices) == 9000 and len(proxy.images) == 64
    assert not set(data["public_source_indices"].tolist()) & set(proxy.final_indices)
    assert sorted(data["public_source_indices"].tolist() + proxy.final_indices) == list(
        range(10000)
    )
    assert not data["metadata"]["private_training_examples_used"]
    cfg["feedback"]["public_proxy_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        PublicCurvatureProxy(cfg)


def test_proxy_changes_coefficients_and_noise_without_changing_covariance_or_fixed_action_cost():
    cfg = smoke_config(configuration())
    p = public_preflight(cfg)
    f = DPHistoryFeedback(p)
    f.prepare(1)
    with pytest.raises(ValueError, match="curvature"):
        f.expected_costs(f.pred_mean, f.pred_cov, [1], 1)
    cov = f.extra.copy()
    coefficients = np.array([0.7, 0.1, 0.1, 0.1]) * sum(p.dimensions.values())
    report = dict(
        round=1,
        names=list(p.dimensions),
        coefficients=coefficients,
        public_data_sha256=cfg["feedback"]["public_proxy_sha256"],
    )
    f.set_public_coefficients(1, report)
    with pytest.raises(ValueError):
        f.set_public_coefficients(1, report)
    oldcfg = smoke_config(load_config(ROOT / "configs/dvpa_fmnist_local_seed42.yaml"))
    old = public_preflight(oldcfg)
    of = DPHistoryFeedback(old)
    of.prepare(1)
    cost = f.expected_costs(f.pred_mean, f.pred_cov, [1], 1)[0, 0]
    baseline = of.expected_costs(of.pred_mean, of.pred_cov, [1], 1)[0, 0]
    assert cost / baseline == pytest.approx(
        coefficients / np.array(list(p.dimensions.values()))
    )
    assert np.array_equal(cov, f.extra) and np.array_equal(f.extra, of.extra)
    ledger = GroupRDPFilter(p)
    a, _ = DecisionValueController(p).choose(1, f, ledger.remaining())
    assert ledger.certify(a).as_dict() == GroupRDPFilter(old).certify(a).as_dict()


@pytest.mark.parametrize(
    "key,value",
    [
        ("utility_proxy", "unknown"),
        ("public_proxy_probes", 0),
        ("public_proxy_sha256", "abc"),
        ("public_proxy_file", "../../outside.pt"),
        ("extra_covariance", "public_bound"),
    ],
)
def test_invalid_public_proxy_settings_rejected(key, value):
    cfg = configuration()
    cfg["feedback"][key] = value
    with pytest.raises(ValueError):
        resolve(cfg)


def test_formal_validation_gate_rejects_unvalidated_input():
    cfg = configuration()
    cfg["feedback"]["public_proxy_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="validation"):
        verify_public_validation(cfg)


def test_training_replay_keeps_public_history_before_private_training_and_rejects_tampering(
    monkeypatch,
):
    from src.brdp_source_protocol import PersistentSourceClient

    calls = []
    old_set = DPHistoryFeedback.set_public_coefficients
    old_train = PersistentSourceClient.train

    def set_coeff(self, t, report):
        result = old_set(self, t, report)
        calls.append(t)
        return result

    def train(self, *args, **kwargs):
        assert calls and calls[-1] == self.epochs_completed + 1
        return old_train(self, *args, **kwargs)

    monkeypatch.setattr(DPHistoryFeedback, "set_public_coefficients", set_coeff)
    monkeypatch.setattr(PersistentSourceClient, "train", train)
    cfg = smoke_config(configuration())
    cfg["feedback"]["public_proxy_probes"] = 2
    parent = ROOT / "reports/dvpa"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="public-proxy-", dir=parent) as directory:
        out = Path(directory) / "run"
        result = run(cfg, out)
        assert result["rounds"] == 4 and result["local_search"]["checked_rounds"] == 3
        assert audit(out)["status"] == "passed"
        report_path = out / "public_proxy/round_002.json"
        report = json.loads(report_path.read_text())
        report["coefficients"][0] += 1.0
        report_path.write_text(json.dumps(report))
        with pytest.raises(ValueError, match="public_proxy"):
            audit(out)
