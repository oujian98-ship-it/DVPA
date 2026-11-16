import contextlib
import copy
import csv
import io
import math
import hashlib
import json
from pathlib import Path

import numpy as np
from opacus.accountants.analysis import rdp as opacus_rdp
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.brdp_group_accountant import (
    GroupPrivacyLedger,
    joint_round_rdp,
    sampled_gaussian_rdp_upper,
    source_orders,
)
from src.brdp_noise import log_density
from src.brdp_source_audit import audit
from src.brdp_source_protocol import (
    PersistentSourceClient,
    clip_source_update,
    load_datasets,
    partition_and_groups,
    preflight,
    validate_config,
)
from src.brdp_source_runner import run
from src.layer_registry import LayerRegistry
from src.upstream_runner import build_source_classes, source_model


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def config(smoke=True):
    name = "brdp_source_smoke.yaml" if smoke else "brdp_source_brdp_joint_seed42.yaml"
    return yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / name).read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("q,squared", [(0.1, 0.2), (0.5, 1.0), (0.9, 2.0)])
def test_reference_integer_moments_and_fractional_upper_bound(q, squared):
    orders = np.asarray([1.1, 1.5, 2.0, 2.5, 3.0, 4.5, 8.0, 16.0])
    expected = opacus_rdp.compute_rdp(
        q=q, noise_multiplier=1 / math.sqrt(squared), steps=1, orders=orders
    )
    computed = sampled_gaussian_rdp_upper(q, squared, orders)
    assert np.all(expected <= computed + 1e-9)
    integers = orders == np.floor(orders)
    assert np.allclose(expected[integers], computed[integers], rtol=1e-9, atol=1e-10)
    assert np.all(expected <= joint_round_rdp(q, squared, orders, [(0.0, 1.0)]) + 1e-9)


@pytest.mark.parametrize(
    "q,r,alpha", [(0.3, 0.02, 2.0), (0.7, 0.2, 3.0), (0.9, 0.1, 4.5)]
)
def test_non_gaussian_sampling_bound_covers_both_density_directions(q, r, alpha):
    shift, tau = 0.7, 1.0

    def logs(x):
        base = float(log_density(x, 1.0, tau, r))
        mix = float(
            logsumexp(
                [
                    math.log1p(-q) + base,
                    math.log(q) + float(log_density(x - shift, 1.0, tau, r)),
                ]
            )
        )
        return base, mix

    cuts = [-math.inf, *sorted({-tau, tau, shift - tau, shift + tau}), math.inf]
    for reverse in [False, True]:

        def integrand(x):
            b, p = logs(x)
            if reverse:
                b, p = p, b
            return math.exp(alpha * p + (1 - alpha) * b)

        integral = sum(
            quad(integrand, lo, hi, epsabs=1e-10)[0] for lo, hi in zip(cuts, cuts[1:])
        )
        actual = math.log(integral) / (alpha - 1)
        bound = joint_round_rdp(q, shift**2, np.asarray([alpha]), [(r, tau)])[0]
        assert actual <= bound + 1e-8


def test_author_model_and_screenshot_group_preflight():
    cfg = validate_config(config(False))
    model = source_model()("fmnist", 125)
    registry = LayerRegistry(model)
    assert sum(registry.dimensions.values()) == 582026
    assert model.conv1[0].padding == (0, 0)
    assert model.fc1[0].in_features == 1024
    public = preflight(cfg, registry.dimensions)
    assert len(public["orders"]) == 601
    assert public["counts"].tolist() == [34, 43, 23]
    assert public["plans"][11].group_rates == (0.5, 0.6, 0.7)
    assert public["plans"][12].group_rates == (0.9, 0.9, 0.9)
    assert public["plans"][11].denominator == pytest.approx(58.9)
    assert public["plans"][12].denominator == pytest.approx(90.0, rel=0, abs=1e-12)
    assert np.dot(public["counts"] / 100, public["group_radii"]) == pytest.approx(250.0)
    assert public["final_epsilon"] == pytest.approx(
        [9.99999, 19.99998, 29.99997], abs=1e-7
    )
    assert all(
        e <= target for e, target in zip(public["final_epsilon"], cfg["epsilons"])
    )
    # The same inclusion coin applies to all layers: charge their joint norm.
    joint = joint_round_rdp(0.5, 1.0, np.asarray([4.0]), [(0.0, 1.0)])[0]
    incorrectly_split = (
        2 * joint_round_rdp(0.5, 0.5, np.asarray([4.0]), [(0.0, 1.0)])[0]
    )
    assert joint > incorrectly_split


def test_group_filter_rejects_envelope_violation_without_mutation():
    cfg = config()
    public = preflight(cfg, LayerRegistry(source_model()("fmnist", 16)).dimensions)
    ledger = GroupPrivacyLedger(
        public["orders"], cfg["epsilons"], cfg["delta"], public["envelope"]
    )
    with pytest.raises(RuntimeError, match="envelope"):
        ledger.accept(1, public["envelope"][0] * 1.1)
    assert not ledger.actual.any() and not ledger.history
    rows = ledger.accept(1, public["envelope"][0])
    assert [row["epsilon_target"] for row in rows] == [10.0, 20.0, 30.0]
    assert all(row["actual_path_value_is_global_guarantee"] is False for row in rows)
    with pytest.raises(ValueError, match="sequential"):
        ledger.accept(1, public["envelope"][0])


def test_source_clipping_units_and_joint_weight_bias_ball():
    cfg = config()
    registry = LayerRegistry(source_model()("fmnist", 16))
    public = preflight(cfg, registry.dimensions)
    plan = public["plans"][0]
    raw = {g: torch.full((d,), 100.0) for g, d in registry.dimensions.items()}
    clipped = clip_source_update(raw, public["group_radii"][0], plan)
    assert (
        sum(float(v.double().square().sum()) for v in clipped.values())
        <= public["group_radii"][0] ** 2
    )
    delta = {g: torch.zeros(d) for g, d in registry.dimensions.items()}
    for g in delta:
        delta[g][0] = 1e-5
    transformed = {g: v / cfg["lr"] * cfg["rate"] for g, v in delta.items()}
    final = clip_source_update(transformed, 250.0, plan)
    assert all(
        torch.allclose(
            final[g] * cfg["lr"] * cfg["rate"], delta[g] * cfg["rate"] ** 2, atol=1e-10
        )
        for g in delta
    )


def test_training_matches_author_persistent_optimizer_for_two_rounds():
    cfg = config()
    cfg["clients"] = 3
    cfg["batch_size"] = 8
    dataset = TensorDataset(
        torch.linspace(-1, 1, 32 * 784).reshape(32, 1, 28, 28), torch.arange(32) % 10
    )
    global_model = source_model()("fmnist", 8)
    registry = LayerRegistry(global_model)
    own = PersistentSourceClient(dataset, list(range(32)), cfg, "cpu")
    C, _, _ = build_source_classes("time_adaptive")
    original = C(
        0,
        dataset,
        dataset,
        1,
        10.0,
        1e-5,
        3,
        3,
        0.001,
        8,
        0.9,
        "fmnist",
        1000000,
        [],
        0.5,
        "cpu",
    )
    original.test_metrics = lambda: (0, 1)
    original.train_metrics = lambda: (0.0, 1)
    original.evaluate = lambda: 0.0
    original.clip_layer_parameters = lambda *args: None
    original.add_noise = lambda *args: None
    original.local_clip_value = 1e30
    original.local_noise_multiplier = 1.0
    original.sample_rate = 0.5
    source_shuffle_rng = torch.Generator().manual_seed(cfg["seed"])
    original.train.__func__.__globals__["DataLoader"] = lambda *a, **kw: DataLoader(
        *a, **kw, generator=source_shuffle_rng
    )
    assert np.array_equal(np.asarray(original.alphas), source_orders())
    for t in range(2):
        torch.manual_seed(100 + t)
        with contextlib.redirect_stdout(io.StringIO()):
            expected = original.train(global_model, t)[0]
        torch.manual_seed(100 + t)
        actual = own.train(global_model, registry)
        for g in registry.names:
            packed = torch.cat([expected[n].flatten() for n in registry.parameters[g]])
            assert torch.allclose(
                actual[g] * cfg["lr"] * cfg["rate"], packed, rtol=1e-6, atol=1e-8
            )
        assert own.scheduler.last_epoch == original.scheduler.last_epoch == t + 1
        assert (
            own.optimizer.param_groups[0]["lr"]
            == original.optimizer.param_groups[0]["lr"]
        )
        for a, b in zip(
            own.model.parameters(), original.local_model.parameters(), strict=True
        ):
            assert torch.equal(
                own.optimizer.state[a]["momentum_buffer"],
                original.optimizer.state[b]["momentum_buffer"],
            )
    own.train(global_model, registry, return_update=False)
    assert own.epochs_completed == 3 and own.scheduler.last_epoch == 3


def test_client_shuffle_is_isolated_from_other_clients_rng_state():
    cfg = config()
    dataset = TensorDataset(torch.zeros(16, 1, 28, 28), torch.arange(16) % 10)
    model = source_model()("fmnist", 16)
    registry = LayerRegistry(model)
    client = PersistentSourceClient(dataset, list(range(16)), cfg, "cpu", slot=2)
    before = torch.get_rng_state().clone()
    client.train(model, registry, return_update=False)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("mechanism", ["gaussian", "brdp"])
def test_full_source_pipeline_and_group_audit(tmp_path, mechanism):
    cfg = config()
    cfg["mechanism"] = mechanism
    result = run(cfg, tmp_path / "run")
    assert result["parameters"] == 582026
    assert result["non_gaussian_layer_releases"] == (8 if mechanism == "brdp" else 0)
    assert audit(tmp_path / "run")["status"] == "passed"
    assert all(
        row["epsilon_guarantee"] <= row["epsilon_target"]
        for row in result["group_privacy"]
    )
    path = tmp_path / "run/layers.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["noise_std"] = str(float(rows[0]["noise_std"]) / 2)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Audit mismatch"):
        audit(tmp_path / "run")


def test_partition_frozen_reproducible_and_no_formal_truncation():
    cfg = config()
    train, _ = load_datasets(cfg)
    a, ga = partition_and_groups(cfg, train.tensors[1])
    b, gb = partition_and_groups(cfg, train.tensors[1])
    assert a == b and np.array_equal(ga, gb)
    formal = config(False)
    formal["limit_train"] = 600
    with pytest.raises(ValueError, match="full clean"):
        validate_config(formal)
    bad = copy.deepcopy(cfg)
    bad["adjacency"] = "replace_client"
    with pytest.raises(ValueError, match="add/remove"):
        validate_config(bad)


def test_empty_poisson_draw_still_trains_every_client_and_releases_noise(
    tmp_path, monkeypatch
):
    cfg = config()
    cfg.update(clients=3, rounds=2, transition_round=2)

    class NoSelection:
        def random(self, n):
            return np.ones(n)

    monkeypatch.setattr(
        "src.brdp_source_runner._private_rngs",
        lambda device: (
            NoSelection(),
            np.random.default_rng(33),
            torch.Generator(device=device).manual_seed(34),
        ),
    )
    calls = []
    train = PersistentSourceClient.train

    def traced(self, *args, **kwargs):
        result = train(self, *args, **kwargs)
        calls.append((self.epochs_completed, kwargs["return_update"]))
        return result

    monkeypatch.setattr(PersistentSourceClient, "train", traced)
    result = run(cfg, tmp_path / "empty")
    assert len(calls) == 6 and all(not selected for _, selected in calls)
    assert sorted(epoch for epoch, _ in calls) == [1, 1, 1, 2, 2, 2]
    assert result["non_gaussian_layer_releases"] == 4
    assert audit(tmp_path / "empty")["status"] == "passed"


def test_formal_suite_rejects_path_escape_and_changed_tested_code(
    tmp_path, monkeypatch
):
    import scripts.run_brdp_source_suite as suite

    monkeypatch.setattr(suite, "ROOT", tmp_path)
    manifest = {}
    for name in suite.REQUIRED_GATE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tested", encoding="utf-8")
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "gate.json").write_text(
        json.dumps(dict(status="passed", tests_passed=75, source_sha256=manifest)),
        encoding="utf-8",
    )
    assert suite.verify_gate("gate.json")["status"] == "passed"
    with pytest.raises(ValueError, match="inside BR_DP"):
        suite.workspace_path("../trynew")
    (tmp_path / "src/brdp_group_accountant.py").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        suite.verify_gate("gate.json")
