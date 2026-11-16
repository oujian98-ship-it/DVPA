"""Test-gated DVPA execution; all artifacts remain inside BR_DP.

Use --test to create a gate for the exact current code, then run --smoke or a
formal seed42 configuration. A completed run is independently replayed.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

from src.brdp_source_runner import _json
from src.dvpa_audit import audit
from src.dvpa_config import ROOT, load_config, public_preflight, smoke_config
from src.dvpa_runner import inside_output, run

TEST_FILES = [
    "test_dvpa_datasets.py",
    "test_dvpa_filter.py",
    "test_dvpa_controller.py",
    "test_dvpa_information_weight.py",
    "test_dvpa_train_public.py",
    "test_train_public64.py",
    "test_train_public_baselines.py",
    "test_dvpa_pipeline.py",
    "test_dvpa_ranking.py",
    "test_dvpa_local_search.py",
    "test_dvpa_public_proxy.py",
    "test_dvpa_forecast.py",
    "test_brdp_source_protocol.py",
    "test_brdp_noise.py",
    "test_brdp_rdp.py",
    "test_sayg_comparison.py",
    "test_three_methods_seeds.py",
    "test_brdp_net_gain.py",
]
MIN_TESTS = 136


def manifest():
    paths = set((ROOT / "src").rglob("*.py")) | set(
        (ROOT / "fl-experiments-code-submission").rglob("*.py")
    )
    paths |= {ROOT / "tests" / name for name in TEST_FILES}
    paths |= {
        ROOT / "scripts/run_dvpa_suite.py",
        ROOT / "scripts/run_dvpa_datasets.py",
        ROOT / "scripts/run_dvpa_information_weight.py",
        ROOT / "scripts/run_dvpa_train_public.py",
        ROOT / "scripts/run_train_public_baselines.py",
        ROOT / "scripts/run_train_public64.py",
        ROOT / "scripts/check_dvpa_public_mechanism.py",
        ROOT / "scripts/check_dvpa_ranking.py",
        ROOT / "scripts/check_dvpa_proxy.py",
        ROOT / "scripts/check_dvpa_forecast.py",
        ROOT / "configs/dvpa_fmnist.yaml",
        ROOT / "configs/brdp_source_gaussian_joint_seed42.yaml",
    }
    # Existing regression tests load additional public configurations.
    paths |= set((ROOT / "configs").rglob("*.yaml")) | set(
        (ROOT / "configs").rglob("*.json")
    )
    paths |= {
        p
        for p in [
            ROOT / "data/public_fmnist_proxy.pt",
            ROOT / "reports/dvpa_proxy/fmnist_validation.json",
            ROOT / "reports/dvpa_forecast/validation.json",
        ]
        if p.is_file()
    }
    paths |= set((ROOT / "data").glob("public_mnist_proxy.pt"))
    paths |= set((ROOT / "reports/dvpa_datasets").glob("*_transfer_validation.json"))
    paths |= {p for p in [ROOT / "data/public_fmnist_train1000.pt", ROOT / "reports/dvpa_trainpublic_20261106/public_validation.json"] if p.is_file()}
    paths |= set((ROOT / "reports/train_public_baselines_20261106/configs").glob("*.yaml"))
    paths |= {p for p in [ROOT / "data/public_fmnist_train64.pt", ROOT / "reports/train_public64_seed42_20261106/public_validation.json"] if p.is_file()}
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def test_gate():
    report = ROOT / "reports/dvpa"
    report.mkdir(parents=True, exist_ok=True)
    before = manifest()
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *[f"tests/{name}" for name in TEST_FILES],
            "--junitxml",
            str(report / "pytest.xml"),
            "--basetemp",
            str(report / "pytest_gate"),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (report / "pytest.log").write_text(process.stdout, encoding="utf-8")
    print(process.stdout, flush=True)
    if process.returncode:
        _json(
            report / "test_gate.json",
            dict(status="failed", exit_code=process.returncode),
        )
        raise RuntimeError("DVPA regression gate failed; see reports/dvpa/pytest.log")
    suites = ET.parse(report / "pytest.xml").getroot()
    suites = list(suites) if suites.tag == "testsuites" else [suites]
    tests = sum(int(s.attrib["tests"]) for s in suites)
    errors = sum(
        int(s.attrib.get("errors", 0))
        + int(s.attrib.get("failures", 0))
        + int(s.attrib.get("skipped", 0))
        for s in suites
    )
    modules = sorted(
        {
            case.attrib.get("classname", "")
            for suite in suites
            for case in suite.iter("testcase")
        }
    )
    expected_modules = {"tests." + name.removesuffix(".py") for name in TEST_FILES}
    if (
        errors
        or tests < MIN_TESTS
        or not expected_modules <= set(modules)
        or before != manifest()
    ):
        raise RuntimeError("Incomplete/changed test inputs; formal execution refused")
    gate = dict(
        status="passed",
        tests_passed=tests,
        source_sha256=before,
        test_modules=modules,
        scope="DVPA mechanisms, training/replay/tampering plus existing Gaussian/BRDP/source regressions",
    )
    _json(report / "test_gate.json", gate)
    return gate


def verify_gate():
    path = ROOT / "reports/dvpa/test_gate.json"
    if not path.is_file():
        raise RuntimeError("Run --test before training")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if (
        gate.get("status") != "passed"
        or gate.get("tests_passed", 0) < MIN_TESTS
        or not {"tests." + name.removesuffix(".py") for name in TEST_FILES}
        <= set(gate.get("test_modules", []))
        or gate.get("source_sha256") != manifest()
    ):
        raise RuntimeError(
            "Passed tests for exact current sources/configs required; rerun --test"
        )
    return gate


def verify_formal(cfg):
    base = cfg["base"]
    expected = dict(
        clients=100,
        rounds=25,
        local_epochs=30,
        batch_size=125,
        lr=0.001,
        rate=0.9,
        clip=250.0,
        delta=1e-5,
        epsilons=[10.0, 20.0, 30.0],
        group_proportions=[0.34, 0.43, 0.23],
        q_save=[0.5, 0.6, 0.7],
        q_spend=0.9,
        transition_round=13,
    )
    if (
        base["dataset"] not in {"fmnist", "mnist"}
        or any(base[k] != v for k, v in expected.items())
        or base.get("dataset_mode", "official") != "official"
        or cfg["feedback"]["warmup_rounds"] != 3
    ):
        raise ValueError(
            "Formal DVPA suite requires full author-matched screenshot protocol"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dvpa_fmnist_local_seed42.yaml")
    parser.add_argument("--output")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume:
        parser.error(
            "Resume unsupported: private persistent client states are not public artifacts"
        )
    cfg = load_config(args.config)
    if args.smoke:
        cfg = smoke_config(cfg)
    if args.preview:
        print(json.dumps(public_preflight(cfg).as_dict(), indent=2))
        return
    if args.test:
        test_gate()
    if not args.output:
        if args.test:
            return
        parser.error("--output required for training")
    gate = verify_gate()
    if not args.smoke:
        verify_formal(cfg)
    out = inside_output(args.output)
    result = run(cfg, out)
    replay = audit(out)
    _json(out / "audit.json", replay)
    _json(out / "test_gate.json", gate)
    _json(
        out / "suite_state.json",
        dict(
            status="completed",
            audit=replay["status"],
            dataset_scope=result["dataset_scope"],
            method="DVPA",
            seed=cfg["base"]["seed"],
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
