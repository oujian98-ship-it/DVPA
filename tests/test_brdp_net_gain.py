import copy
import math

import numpy as np
import pytest

from src.brdp_net_gain import (
    FixedGeometryAccountant,
    gain_metrics,
    profile_metrics,
    public_mc_case,
    static_geometry,
)


def test_gaussian_limit_and_geometry_never_changes():
    p = static_geometry("control", 0.9, 3, 10.0, 4, 128)
    before = copy.deepcopy(p)
    account = FixedGeometryAccountant(
        p, orders=np.array([2.0, 2.5, 3.0, 4.0, 8.0, 16.0, 32.0])
    )
    g = account.calibrate(p.targets, 0.0, 1.0, 1e-5, mode="gaussian")
    z = account.calibrate(p.targets, 0.0, 1.0, 1e-5, mode="brdp")
    assert g["multiplier"] == z["multiplier"]
    assert max(g["epsilon"]) <= 10 and max(g["epsilon"]) > 9.999
    for field in ["q", "radii", "clip_weights", "base_stds", "scales"]:
        np.testing.assert_array_equal(getattr(p, field), getattr(before, field))


def test_infinite_noise_bracket_is_stable_and_nonnegative():
    a = FixedGeometryAccountant(static_geometry("floor", 0.9, 25, 10.0))
    values = a.costs(1e12, 0.0, 1.0, "gaussian")
    assert np.isfinite(values).all() and (values >= 0).all()


def test_negative_high_fractional_library_costs_use_valid_upper_not_zero():
    from src.brdp_net_gain import _gaussian_detail
    from src.brdp_group_accountant import sampled_gaussian_rdp_upper

    orders = (2.0, 3.0, 50.5, 60.5, 100.5)
    values, fallback = _gaussian_detail(0.5, 1 / 256, orders)
    upper = sampled_gaussian_rdp_upper(0.5, 1 / 256, np.array(orders))
    assert len(fallback) > 0 and np.all(values > 0)
    for i, order in enumerate(orders):
        if order in fallback:
            assert values[i] >= upper[i]


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9, 1.0])
def test_mse_and_absolute_threshold_tail_reduce_to_gaussian(q):
    m = gain_metrics(1.0, 0.0, 1.0, 128)
    assert m["ideal_parallel_mse_ratio"] == 1
    assert m["energy_ratio"] == 1
    assert m["tail_ratio_at_2_gaussian_std"] == pytest.approx(1)
    p = static_geometry("g", q, 1, 10.0)
    a = FixedGeometryAccountant(p, orders=np.array([2.0, 3.0, 4.0, 8.0, 16.0]))
    np.testing.assert_array_equal(
        a.costs(2.0, 0.0, 1.0, "gaussian"), a.costs(2.0, 0.0, 1.0, "brdp")
    )


def test_net_condition_and_alignment_from_independent_covariance():
    m = gain_metrics(1.02, 0.5, 1.0, 128)
    assert m["ideal_parallel_mse_ratio"] < 1
    assert m["energy_ratio"] > 1
    cutoff = m["minimum_cosine_squared_for_direction_gain"]
    assert 0 < cutoff < 1
    for a in [0.25, 0.5, 0.75, 1.0]:
        direction = np.array([math.sqrt(a), math.sqrt(1 - a)])
        covariance = np.diag([1.02**2 * m["fixed_sigma_variance_factor"], 1.02**2])
        value = float(direction @ covariance @ direction)
        assert value == pytest.approx(m[f"mse_ratio_cos2_{a:g}"])
        assert (value < 1) == (a > cutoff)


def test_sigma_inflation_can_destroy_fixed_sigma_recycling_benefit():
    same = gain_metrics(1.0, 0.005, 1.0, 832)
    adjusted = gain_metrics(1.0263090803520427, 0.005, 1.0, 832)
    assert same["ideal_parallel_mse_ratio"] < 1
    assert adjusted["ideal_parallel_mse_ratio"] > 1.05
    assert adjusted["even_perfect_alignment_fails"]
    assert adjusted["minimum_cosine_squared_for_direction_gain"] is None


def test_current_like_composition_fails_but_high_single_release_can_gain():
    orders = np.array([1.1, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    for q, t, eps, k, r, expected in [
        (1.0, 1, 100.0, 1, 0.5, True),
        (0.9, 25, 10.0, 4, 0.005, False),
    ]:
        p = static_geometry("case", q, t, eps, k, 128)
        account = FixedGeometryAccountant(p, orders)
        g = account.calibrate(p.targets, 0.0, 1.0, 1e-5, "gaussian")
        b = account.calibrate(g["epsilon"], r, 1.0, 1e-5)
        assert b["status"] == "FEASIBLE"
        assert np.all(np.array(b["epsilon"]) <= g["epsilon"])
        good = (
            gain_metrics(b["multiplier"] / g["multiplier"], r, 1.0, 128)[
                "ideal_parallel_mse_ratio"
            ]
            < 1
        )
        assert good == expected


def test_bound_infeasible_is_not_silent_gaussian_fallback():
    a = FixedGeometryAccountant(
        static_geometry("tight", 0.9, 25, 1.0, 4),
        np.array([2.0, 3.0, 4.0, 8.0, 16.0, 64.0]),
    )
    result = a.calibrate(np.array([1.0]), 0.8, 1.0, 1e-5)
    assert result["status"] == "BOUND_INFEASIBLE" and "multiplier" not in result


def test_fixed_sensitivity_scaling_does_not_change_relative_gain():
    sigma = []
    orders = np.array([2.0, 3.0, 4.0, 8.0, 16.0, 32.0])
    for d in [1.0, 3.0]:
        a = FixedGeometryAccountant(
            static_geometry("scale", 0.5, 2, 10.0, sensitivity=d), orders
        )
        sigma.append(
            a.calibrate(np.array([10.0]), 0.0, 1.0, 1e-5, "gaussian")["multiplier"]
        )
    assert sigma[1] / sigma[0] == pytest.approx(3.0, rel=1e-8)


def test_public_sampler_checks_moments_tails_energy_and_alignment():
    state = np.random.get_state()
    m = public_mc_case(1.0, 1.02, 0.5, 1.0, 128, 100000, 42)
    after = np.random.get_state()
    assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    assert m["moment_check_passed"] and m["energy_check_passed"]
    assert all(m[f"tail_{k}_check_passed"] for k in [1, 2, 3])
    assert m["empirical_mse_ratio_cos2_1"] == pytest.approx(
        m["theoretical_mse_ratio_cos2_1"], rel=0.025
    )


def test_warmup_and_full_dimensions_affect_trajectory_not_ideal_axis():
    p = static_geometry("warm", 0.9, 3, 10.0, 4, 128)
    p.warmup = 1
    m = profile_metrics(p, 1.02, 0.5, 1.0)
    assert m["trajectory_protected_axes_mse_ratio"] > m["ideal_parallel_mse_ratio"]
    assert m["trajectory_total_energy_ratio"] > 1


@pytest.mark.parametrize("ratio,dimension", [(0.0, 1), (float("nan"), 2), (1.0, 0)])
def test_invalid_metrics_rejected(ratio, dimension):
    with pytest.raises(ValueError):
        gain_metrics(ratio, 0.1, 1.0, dimension)
