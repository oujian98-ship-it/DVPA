"""Dimension counts and expected additive DP-noise energy from saved masks.

Counterfactual: all coordinates active with the SAME per-round variance.
This is a noise-only reference, not an unfrozen-training accuracy experiment.
Neither clipped signal energy nor total released-update energy is noise norm.
"""
import math


def summarize_noise_savings(rounds,total_dimension):
    if type(total_dimension) is not int or total_dimension<1 or not rounds:raise ValueError('Invalid dimension/horizon')
    trace=[]
    for t,row in enumerate(rounds,1):
        if int(row['round'])!=t:raise ValueError('Incomplete or unordered rounds')
        d=int(row['active_params']);v=float(row['noise_var_per_coordinate'])
        if not 0<d<=total_dimension or not math.isfinite(v) or v<0:raise ValueError('Invalid active dimension or noise variance')
        expected=d*v;full=total_dimension*v
        if not math.isclose(float(row['noise_energy_expected']),expected,rel_tol=1e-9,abs_tol=1e-12):raise ValueError('Saved noise energy mismatch')
        trace.append(dict(round=t,active_dimension=d,active_fraction=d/total_dimension,
                          dimension_saved_fraction=1-d/total_dimension,noise_variance_per_coordinate=v,
                          expected_noise_energy=expected,full_active_expected_noise_energy=full,
                          expected_noise_energy_saved=full-expected,
                          expected_noise_saved_fraction=(1-d/total_dimension) if full>0 else None))
    actual=math.fsum(r['expected_noise_energy'] for r in trace)
    reference=math.fsum(r['full_active_expected_noise_energy'] for r in trace)
    dimensions=sum(r['active_dimension'] for r in trace)
    return dict(total_dimension=total_dimension,rounds=len(trace),final_active_dimension=trace[-1]['active_dimension'],
                final_dimension_saved_fraction=trace[-1]['dimension_saved_fraction'],
                cumulative_active_coordinate_rounds=dimensions,
                cumulative_full_active_coordinate_rounds=len(trace)*total_dimension,
                cumulative_dimension_saved_fraction=1-dimensions/(len(trace)*total_dimension),
                cumulative_expected_noise_energy=actual,cumulative_full_active_expected_noise_energy=reference,
                cumulative_expected_noise_saved_fraction=1-actual/reference if reference>0 else None,
                realized_noise_energy=None,privacy_epsilon_savings=None,
                scope='Actual masks times known variance; expected additive noise only. Fixed-variance full-active reference; no realized noise-vector measurement, accuracy or epsilon saving inference.'),trace
