"""Stage A: public-noise benchmark, same certified bound and absolute threshold."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from src.brdp_noise import absolute_tail, moments, sample_scalar_noise
from src.brdp_rdp import BoundTooLoose, calibrate, conservative_rdp, joint_grid_diagnostic


def benchmark(config, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    alpha, sensitivity = float(config['alpha']), float(config['sensitivity'])
    rows = []
    feasible = competing = 0
    for dimension in config['dimensions']:
        for budget in config['rdp_budgets']:
            baseline = calibrate(alpha, sensitivity, dimension, budget)
            for r in config['recycle_prob_candidates']:
                for tau in config['tau_candidates']:
                    try:
                        c = calibrate(alpha, sensitivity, dimension, budget, r, tau)
                    except BoundTooLoose:
                        rows.append(dict(dimension=dimension, rdp_budget=budget, recycle_prob=r,
                            tau=tau, status='BOUND_TOO_LOOSE', certified_cost='', sigma='',
                            threshold='', tail_probability='', gaussian_tail_probability='',
                            parallel_variance_ratio='', energy_ratio='', fixed_sigma_tail='',
                            fixed_sigma_rdp_upper='', diagnostic_grid_max='', competitive=False))
                        continue
                    feasible += 1
                    threshold = baseline.sigma_parallel*config['tail_threshold_sigma']
                    tail = absolute_tail(threshold, c.sigma_parallel, tau, r)
                    base_tail = absolute_tail(threshold, baseline.sigma_parallel, 1., 0.)
                    variance_ratio = moments(c.sigma_parallel, tau, r)['variance']/baseline.sigma_parallel**2
                    energy_ratio = c.expected_noise_energy/baseline.expected_noise_energy
                    good = bool(r > 0 and tail < base_tail-1e-12
                                and variance_ratio <= config['max_parallel_variance_ratio']
                                and energy_ratio <= config['max_energy_ratio'])
                    competing += good
                    grid = joint_grid_diagnostic(alpha, sensitivity, dimension,
                            c.sigma_parallel, c.sigma_orth, tau, r, config['diagnostic_grid_points'])
                    rows.append(dict(dimension=dimension, rdp_budget=budget, recycle_prob=r,
                        tau=tau, status=c.status, certified_cost=c.rdp_upper,
                        sigma=c.sigma_parallel, threshold=threshold, tail_probability=tail,
                        gaussian_tail_probability=base_tail, parallel_variance_ratio=variance_ratio,
                        energy_ratio=energy_ratio,
                        fixed_sigma_tail=absolute_tail(threshold, baseline.sigma_parallel, tau, r),
                        fixed_sigma_rdp_upper=conservative_rdp(alpha, sensitivity, dimension,
                            baseline.sigma_parallel, baseline.sigma_orth, tau, r),
                        diagnostic_grid_max=grid['diagnostic_grid_max'], competitive=good))
    with (root/'mechanisms.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Public synthetic sampling check with a deliberately public reproducibility seed.
    rng = np.random.default_rng(config['public_seed'])
    sample = sample_scalar_noise(1., 1., .1, rng, config['sample_count'])
    sampling_check = dict(scope='public noise only; not private training',
        sample_count=len(sample), empirical_mean=float(sample.mean()),
        empirical_variance=float(sample.var()), theoretical=moments(1., 1., .1))
    (root/'sampling_check.json').write_text(json.dumps(sampling_check, indent=2), encoding='utf-8')
    summary = dict(status='completed', scope='public synthetic noise benchmark; no FL accuracy evidence',
        candidates=len(rows), feasible_under_conservative_bound=feasible,
        competitive_non_gaussian_candidates=competing,
        stage_a_status='candidate_found_needs_validation' if competing else 'no_competitive_candidate_under_conservative_bound',
        grid_certified=False, grid_used_for_charging=False,
        limitation='No tight continuous-offset certificate implemented; negative bound result is not an impossibility result')
    (root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    print(json.dumps(benchmark(config, args.output), indent=2))


if __name__ == '__main__':
    main()
