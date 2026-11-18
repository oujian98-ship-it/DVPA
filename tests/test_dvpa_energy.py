import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from scripts.diagnose_energy import calibrate
from src.dvpa_audit import audit
from src.dvpa_config import ROOT, load_config, public_preflight, resolve, smoke_config
from src.dvpa_controller import DecisionValueController
from src.dvpa_energy_feedback import LayerEnergyState, energy_observations, observation_geometry
from src.dvpa_energy_voi import compare_candidates, gate_scope, layer_candidates, load_energy_gate
from src.dvpa_feedback import DPHistoryFeedback
from src.dvpa_filter import GroupRDPFilter, PrecisionAction
from src.dvpa_public_training import partition_record
from src.dvpa_ranking import confirm_energy_gain
from src.dvpa_runner import run


def configuration():
    cfg = smoke_config(load_config(ROOT / "configs/dvpa_fmnist.yaml"))
    cfg["feedback"]["feedback_mode"] = "energy"
    cfg["controller"].update(mode="frozen_info", tail_solver="convex", curvature_only=True,
                             information_value_weight=0, ranking_enabled=False, local_search_enabled=False)
    return cfg


def test_noncentral_energy_moments_and_high_dimension_calibration():
    rows = calibrate([832, 51264, 524800])
    for row in rows:
        assert abs(row["mean_error_se"]) < 5
        assert row["variance_ratio"] == pytest.approx(1, abs=.025)
        assert row["max_cdf_error"] < .02
    assert all(.48 < r["fraction_negative"] < .52 for r in rows if r["true_energy"] == 0)


def test_signed_measurement_and_gaussian_scale():
    protocol = public_preflight(configuration())
    feedback = DPHistoryFeedback(protocol)
    feedback.prepare(1)
    ledger = GroupRDPFilter(protocol)
    action, _ = DecisionValueController(protocol).choose(1, feedback, ledger.remaining())
    cert = ledger.certify(action)
    published = {n: torch.zeros(d) for n, d in protocol.dimensions.items()}
    rows = energy_observations(published, cert, protocol.dimensions, protocol.base["seed"])
    for r in rows:
        assert r["e_hat"] < 0
        assert r["e_hat_raw"] == pytest.approx(r["e_hat"]*(r["s_t"]*r["D_t"])**2)
        assert r["nu2"] == pytest.approx(.25/(r["z_t"]*r["p_tl"]*r["D_t"]**2))
    state = LayerEnergyState(protocol)
    state.observe(1, rows)
    assert np.allclose(state.posterior.sum(axis=1), 1)
    assert np.all(np.isfinite(state.posterior))


def test_candidate_floor_and_sigma_recertification():
    protocol = public_preflight(configuration())
    ledger = GroupRDPFilter(protocol)
    floor = protocol.p_min
    z = protocol.z0
    centre = np.array([floor, .2, .3, .5-floor])
    options = layer_candidates(centre, np.full(4, .25), floor, np.ones(4))
    assert any(label == "1->2:0.05" for label, _ in options)
    assert any(label == "2->1:0.05" for label, _ in options)
    for _, p in options:
        assert p.min() >= floor and p.sum() == pytest.approx(1)
        a = PrecisionAction(1, z, tuple(zip(protocol.dimensions, p)), "test", "test")
        cert = ledger.certify(a)
        assert ledger.can_reserve(cert)
        assert np.allclose(cert.group_rdp_increment, protocol.charges(1, z), rtol=1e-12)
        assert np.all(np.asarray(cert.group_rdp_increment) <= protocol.caps)
        _, std2, _ = observation_geometry(cert.source_plan)
        assert np.allclose(cert.source_plan.final_scale**2 * .25/std2, z*p, rtol=1e-12)


def test_equal_observation_channel_and_independent_confirmation():
    protocol = public_preflight(configuration())
    controller = DecisionValueController(protocol)
    feedback = DPHistoryFeedback(protocol)
    feedback.prepare(1)
    # The baseline is the exact water-level optimum for these coefficients.
    p = np.full(4, .25)
    options = layer_candidates(p, p, protocol.p_min, np.ones(4))[:2]
    k = np.ones((protocol.rounds, 4))

    def forecasts(_share, draws):
        return np.exp(.1*draws[1][:, None, :]) * k[1:]

    winner, report = compare_candidates(controller, 1, protocol.z0, options, protocol.caps,
                                        k, feedback, forecasts, samples=4)
    assert winner == 0
    assert report["candidates"][0]["voi_proposal"] == report["candidates"][1]["voi_proposal"]
    streams = []

    def confirm(i):
        streams.append(i)
        return np.ones(4), np.full(4, 1.1)

    args = np.array([0., 2.]), confirm, np.array([0., .2]), 3, 1e-8
    selected, proposed, _, gain, error, lower = confirm_energy_gain(*args)
    assert streams == [1] and proposed == 1 and selected == 0 and lower < 0 and error == 0
    accepted = confirm_energy_gain(args[0], lambda _: (np.ones(4), np.full(4, 2.0)), *args[2:])
    assert accepted[0] == 1 and accepted[-1] > 0 and gain < 0


def test_energy_prefix_and_tamper_replay(monkeypatch):
    observations = DPHistoryFeedback.observe
    selections = DecisionValueController.choose
    events = []

    def choose(self, t, feedback, remaining):
        assert feedback.energy.round == t-1
        events.append(("choose", t))
        return selections(self, t, feedback, remaining)

    def observe(self, t, released, certified):
        assert events[-1] == ("choose", t)
        assert (folder / "released" / f"round_{t:03d}.pt").is_file()
        events.append(("observe", t))
        result = observations(self, t, released, certified)
        assert self.energy.round == t
        with pytest.raises(ValueError, match="Only previous DP history"):
            selections(controller, t, self, controller.protocol.caps)
        return result

    cfg = configuration()
    controller = DecisionValueController(public_preflight(cfg))
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as work:
        folder = Path(work) / "energy"
        with monkeypatch.context() as m:
            m.setattr(DecisionValueController, "choose", choose)
            m.setattr(DPHistoryFeedback, "observe", observe)
            run(cfg, folder)
        assert len(events) == 2*cfg["base"]["rounds"]
        assert audit(folder)["status"] == "passed"
        with (folder / "feedback.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == cfg["base"]["rounds"]*4
        state_path = folder / "feedback_state.pt"
        state = torch.load(state_path, weights_only=True)
        state["energy"]["posterior"][-1][0] = 1.0
        torch.save(state, state_path)
        with pytest.raises(ValueError, match="feedback_state.energy.posterior"):
            audit(folder)


def test_missing_evidence_and_disjoint_validation():
    with pytest.raises(ValueError, match="pinned A/B/C"):
        cfg = configuration()
        cfg["controller"]["voi_explore_p_only"] = True
        resolve(cfg)
    archive = load_config(ROOT / "configs/dvpa_fmnist_trainpublic64_weight10_seed42.yaml")
    base = dict(archive["base"], validation_count=5000)
    record = partition_record(base)
    subsets = [set(record[f"{name}_source_indices"]) for name in ("public", "private", "validation")]
    assert tuple(map(len, subsets)) == (64, 54936, 5000)
    assert set.union(*subsets) == set(range(60000))
    assert all(not a & b for i, a in enumerate(subsets) for b in subsets[i+1:])


def test_gated_fixed_z_decisions_are_replayed():
    cfg = configuration()
    proxy = load_config(ROOT / "configs/dvpa_fmnist_proxy_seed42.yaml")
    cfg["feedback"].update({key: proxy["feedback"][key] for key in (
        "utility_proxy", "public_proxy_file", "public_proxy_validation", "public_proxy_sha256")})
    cfg["feedback"].update(curvature_mode="absolute_scale", public_proxy_probes=2)
    cfg["controller"].update(voi_explore_p_only=True, information_value_weight=1)
    with tempfile.TemporaryDirectory(dir=ROOT / "reports") as work:
        path = Path(work) / "test_gate.json"
        # A zero-slope fixture exercises the deployment pipeline and has no
        # evidentiary value for the real protocol.
        model = dict(center=np.zeros((2, 4, 8)).tolist(), scale=np.ones((2, 4, 8)).tolist(),
                     slopes=np.zeros((2, 4, 8)).tolist(), intercept=np.zeros((2, 4)).tolist(),
                     residual_sd=np.full((2, 4), .05).tolist())
        record = dict(version=1, passed=dict(A=False, B=True, C=True), scope=gate_scope(cfg),
                      predictor=model, confirming_seeds=[42, 43])

        def save():
            path.write_text(json.dumps(record), encoding="utf-8")
            cfg["controller"].update(energy_gate_file=path.relative_to(ROOT).as_posix(),
                                     energy_gate_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

        save()
        with pytest.raises(ValueError, match="did not pass"):
            load_energy_gate(cfg)
        record["passed"]["A"] = True
        save()
        out = Path(work) / "run"
        run(cfg, out)
        path.unlink()
        assert audit(out)["status"] == "passed"
        ranking = json.loads((out / "energy_ranking/round_002.json").read_text(encoding="utf-8"))
        assert ranking["selected_index"] == 0
        assert ranking["candidates"] and all(r["feasible"] for r in ranking["candidates"])
        actions = [json.loads(line)["certified"]["action"] for line in
                   (out / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
        assert ranking["z"] == actions[1]["precision"]
