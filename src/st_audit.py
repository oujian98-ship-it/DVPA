"""Recompute time/layer accounting from actual logged radii/noise, not labels."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .st_accountant import GaussianRDPFilter, gaussian_rho
from .st_runner import validate_config
from .models import build_model
from .layer_registry import LayerRegistry


def close(actual, expected, label):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-12):
        raise ValueError(f'Audit mismatch: {label}: {actual} versus {expected}')


def audit(directory):
    root = Path(directory)
    cfg = validate_config(yaml.safe_load((root/'config.yaml').read_text(encoding='utf-8')))
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    ledger = GaussianRDPFilter(cfg['privacy']['epsilon'], cfg['privacy']['delta'], cfg['privacy']['adjacency'])
    with (root/'layers.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    with (root/'privacy.csv').open(encoding='utf-8', newline='') as handle:
        privacy = list(csv.DictReader(handle))
    names = set(cfg['clipping']['weights'])
    dimensions = LayerRegistry(build_model(cfg)).dimensions
    if names != set(dimensions):
        raise ValueError('Configured layer names do not match the model')
    if len(privacy) != cfg['train']['rounds'] or len(rows) != cfg['train']['rounds']*len(names):
        raise ValueError('Incomplete or duplicate accounting rows')
    for t in range(1, cfg['train']['rounds']+1):
        group = [row for row in rows if int(row['round']) == t]
        if len(group) != len(names) or {row['layer'] for row in group} != names:
            raise ValueError('Layer coverage/uniqueness mismatch')
        sensitivity, noise = {}, {}
        total_allocated = math.fsum(float(row['rho_allocated']) for row in group)
        for row in group:
            g = row['layer']
            active = g not in cfg['freezing']['freeze_rounds'] or t < cfg['freezing']['freeze_rounds'][g]
            close(float(row['active']), int(active), 'active mask')
            denominator = cfg['data']['clients']*cfg['privacy']['sample_rate']
            close(float(row['denominator']), denominator, 'denominator')
            close(float(row['sample_rate']), cfg['privacy']['sample_rate'], 'sample rate')
            close(float(row['dimensions']), dimensions[g], 'layer dimensions')
            close(float(row['budget_fraction']), float(row['rho_allocated'])/total_allocated, 'budget fraction')
            enabled = [k for k in names if k not in cfg['freezing']['freeze_rounds'] or t < cfg['freezing']['freeze_rounds'][k]]
            radius = (cfg['privacy']['clip_norm']*math.sqrt(cfg['clipping']['weights'][g]/
                      math.fsum(cfg['clipping']['weights'][k] for k in enabled))) if active else 0.
            close(float(row['clip_radius']), radius, 'clip radius')
            sensitivity[g] = ledger.adjacency_factor*radius/denominator
            noise[g] = float(row['noise_std'])
            close(float(row['sensitivity']), sensitivity[g], 'sensitivity')
            close(float(row['noise_variance']), noise[g]**2, 'variance')
            close(float(row['expected_noise_energy']), int(row['dimensions'])*noise[g]**2, 'noise energy')
            if not active and noise[g] != 0:
                raise ValueError('Frozen block received noise')
            cost = gaussian_rho({g: sensitivity[g]}, {g: noise[g]})
            close(float(row['rho_allocated']), cost, 'layer rho')
        result = ledger.record(t, sensitivity, noise)
        logged = privacy[t-1]
        for key, value in result.items():
            close(float(logged[key]), value, f'privacy.{key}')
    for key, expected in [('epsilon_bound', ledger.epsilon_bound), ('rho_spent', ledger.spent),
                          ('epsilon_accounting_value', ledger.epsilon_accounting_value),
                          ('rho_cap', ledger.rho_cap), ('alpha', ledger.alpha),
                          ('epsilon_target', ledger.target), ('delta', ledger.delta)]:
        close(float(summary[key]), expected, key)
    with np.load(root/'accountant.npz') as saved:
        for key, expected in [('alpha', ledger.alpha), ('rho_cap', ledger.rho_cap),
                              ('rho_spent', ledger.spent), ('rdp_spent', ledger.alpha*ledger.spent),
                              ('rdp_filter_cap', ledger.alpha*ledger.rho_cap),
                              ('epsilon_target', ledger.target), ('delta', ledger.delta)]:
            close(float(saved[key]), expected, f'npz.{key}')
    return dict(status='passed', rounds=len(privacy), layers=len(names),
                epsilon_bound=ledger.epsilon_bound, epsilon_target=ledger.target,
                scope='artifact consistency audit; not a proof of runtime execution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.results_dir), indent=2))


if __name__ == '__main__':
    main()
