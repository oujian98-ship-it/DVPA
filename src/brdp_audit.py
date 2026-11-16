"""Replay accounting, directions and predictable plans from released DP artifacts.

Artifact consistency is not proof that the executed sampler matched its source,
nor a hardened floating-point privacy certificate.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
import yaml

from .brdp_config import validate_config
from .brdp_rdp import BOUND_MODE
from .brdp_runner import make_components, plan_rows
from .st_runner import round_inputs


def _close(a, b, label):
    if not math.isfinite(float(a)) or not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-13):
        raise ValueError(f'Audit mismatch: {label}')


def _rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def audit(directory):
    root = Path(directory)
    implementation = json.loads((root/'implementation.json').read_text(encoding='utf-8'))
    if implementation['config_sha256'] != hashlib.sha256((root/'config.yaml').read_bytes()).hexdigest():
        raise ValueError('Config hash mismatch')
    if implementation.get('accountant') != BOUND_MODE or implementation.get('sampling_amplification') is not False:
        raise ValueError('Unsupported artifact accountant/trust scope')
    manifest = implementation['source_sha256']
    if not {'brdp_noise.py', 'brdp_rdp.py', 'brdp_filter.py', 'brdp_runner.py'} <= set(manifest):
        raise ValueError('Incomplete source snapshot')
    for name, digest in manifest.items():
        if Path(name).name != name or not name.endswith('.py'):
            raise ValueError('Unsafe source manifest filename')
        if hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Source snapshot hash mismatch')
    cfg = validate_config(yaml.safe_load((root/'config.yaml').read_text(encoding='utf-8')))
    torch.set_num_threads(1)
    torch.manual_seed(cfg['experiment']['seed'])
    model, registry, ledger, direction_state, allocator = make_components(cfg)
    total_rounds = cfg['train']['rounds']
    saved_layers, saved_privacy = _rows(root/'layer_mechanisms.csv'), _rows(root/'privacy.csv')
    saved_rounds = _rows(root/'rounds.csv')
    index = {(int(row['round']), row['layer']): row for row in saved_layers}
    if len(index) != len(saved_layers) or len(index) != total_rounds*len(registry.names):
        raise ValueError('Incomplete or duplicate layer rows')
    if len(saved_privacy) != total_rounds or len(saved_rounds) != total_rounds:
        raise ValueError('Incomplete round/ledger rows')
    released = torch.load(root/'released_updates.pt', map_location='cpu', weights_only=True)
    if set(released) != {str(t) for t in range(1, total_rounds+1)}:
        raise ValueError('Released history coverage mismatch')
    count_soft = count_gaussian = 0
    for t in range(1, total_rounds+1):
        directions = direction_state.for_round(t)
        active, radii, denominator = round_inputs(cfg, registry.names, cfg['clipping']['weights'],
                                                  cfg['freezing']['freeze_rounds'], t)
        plan = allocator.plan(t, active, ledger.remaining, radii, denominator, directions)
        expected_rows = plan_rows(plan, cfg)
        for expected in expected_rows:
            actual = index.get((t, expected['layer']))
            if actual is None or set(actual) != set(expected):
                raise ValueError('Layer row fields mismatch')
            for key, value in expected.items():
                if isinstance(value, (int, float)):
                    _close(actual[key], value, f'{t}.{expected["layer"]}.{key}')
                elif actual[key] != value:
                    raise ValueError(f'Audit mismatch: {key}')
            count_soft += int(expected['active'] and expected['recycle_prob'] > 0)
            count_gaussian += int(expected['active'] and expected['recycle_prob'] == 0)
        row = ledger.accept(plan)
        if set(saved_privacy[t-1]) != set(row):
            raise ValueError('Ledger fields mismatch')
        for key, value in row.items():
            _close(saved_privacy[t-1][key], value, f'privacy.{key}')
        observed = released[str(t)]
        if set(observed) != set(registry.names):
            raise ValueError('Released layer coverage mismatch')
        for g in registry.names:
            x = observed[g]
            if x.ndim != 1 or x.numel() != registry.dimensions[g] or not torch.isfinite(x).all():
                raise ValueError('Invalid released tensor')
            if not active[g] and torch.count_nonzero(x):
                raise ValueError('Frozen layer has nonzero released increment')
        registry.apply(model, observed)
        direction_state.observe(t, observed)
        allocator.observe(t, observed, plan.expected_noise_energy)
        _close(saved_rounds[t-1]['round'], t, 'round order')
        for key, value in [('rdp_round', row['rdp_increment']), ('rdp_charged_total', ledger.spent),
                           ('epsilon_guarantee', ledger.target), ('epsilon_path_conversion', row['epsilon_path_conversion']),
                           ('expected_noise_energy', sum(plan.expected_noise_energy.values()))]:
            _close(saved_rounds[t-1][key], value, f'rounds.{key}')
    state = torch.load(root/'directions_state.pt', map_location='cpu', weights_only=True)
    expected_state = direction_state.state_dict()
    if state['last_observed'] != total_rounds or set(state['ema']) != set(expected_state['ema']):
        raise ValueError('Direction final state mismatch')
    if any(not torch.equal(state['ema'][g], expected_state['ema'][g]) for g in registry.names):
        raise ValueError('Direction replay mismatch')
    if json.loads((root/'allocator_state.json').read_text(encoding='utf-8')) != allocator.state_dict():
        raise ValueError('Allocator replay mismatch')
    final_model = torch.load(root/'model.pt', map_location='cpu', weights_only=True)
    if set(final_model) != set(model.state_dict()):
        raise ValueError('Model coverage mismatch')
    for name, value in model.state_dict().items():
        if not torch.allclose(value, final_model[name], rtol=1e-5, atol=1e-6):
            raise ValueError('Model does not match released updates')
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    expected = dict(alpha=ledger.alpha, delta=ledger.delta, rdp_filter_cap=ledger.cap,
                    rdp_charged_total=ledger.spent, rdp_remaining=ledger.remaining,
                    epsilon_target=ledger.target, epsilon_guarantee=ledger.target,
                    epsilon_path_conversion=ledger.history[-1]['epsilon_path_conversion'],
                    non_gaussian_layer_releases=count_soft, gaussian_layer_releases=count_gaussian,
                    rounds=total_rounds,
                    final_test_accuracy=float(saved_rounds[-1]['public_test_accuracy']),
                    best_test_accuracy=max(float(r['public_test_accuracy']) for r in saved_rounds))
    for key, value in expected.items():
        _close(summary[key], value, f'summary.{key}')
    for key, value in dict(status='completed', mechanism='directional_soft_recycling',
                           accountant=BOUND_MODE, trust_model='trusted_server_aggregate_only',
                           privacy_unit='one fixed client slot', adjacency=ledger.adjacency,
                           sampling_amplification=False,
                           selection_policy=cfg['mechanism']['selection']['policy']).items():
        if summary[key] != value:
            raise ValueError(f'Summary scope mismatch: {key}')
    metadata = json.loads((root/'certificate_metadata.json').read_text(encoding='utf-8'))
    if metadata['bound_mode'] != BOUND_MODE or metadata['grid_diagnostics_used_for_charging'] is not False or metadata['interval_arithmetic'] is not False:
        raise ValueError('Misleading certificate metadata')
    return dict(status='passed', rounds=total_rounds, layers=len(registry.names),
                rdp_charged_total=ledger.spent, epsilon_guarantee=ledger.target,
                non_gaussian_layer_releases=count_soft,
                scope='artifact/accounting/history replay; not a proof of runtime sampling or public evaluation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.results_dir), indent=2))


if __name__ == '__main__':
    main()
