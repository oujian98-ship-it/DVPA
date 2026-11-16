import pytest
import torch
from src.noise_savings import summarize_noise_savings
from src.p1_freezing import StaticMask,group_counts


def test_cumulative_noise_uses_variance_weights_not_mean_dimension_fraction():
    rows=[dict(round=1,active_params=10,noise_var_per_coordinate=9.,noise_energy_expected=90.),
          dict(round=2,active_params=5,noise_var_per_coordinate=1.,noise_energy_expected=5.)]
    summary,trace=summarize_noise_savings(rows,10)
    assert summary['final_dimension_saved_fraction']==.5
    assert summary['cumulative_dimension_saved_fraction']==.25
    assert summary['cumulative_expected_noise_saved_fraction']==pytest.approx(.05)
    assert trace[1]['expected_noise_energy_saved']==5.
    assert summary['realized_noise_energy'] is None and summary['privacy_epsilon_savings'] is None


def test_noise_savings_zero_variance_and_invalid_ledger():
    rows=[dict(round=1,active_params=5,noise_var_per_coordinate=0.,noise_energy_expected=0.)]
    summary,trace=summarize_noise_savings(rows,10)
    assert summary['cumulative_expected_noise_saved_fraction'] is None
    assert trace[0]['expected_noise_saved_fraction'] is None
    for change in [dict(round=2),dict(active_params=11),dict(noise_var_per_coordinate=-1),dict(noise_energy_expected=1)]:
        with pytest.raises(ValueError):summarize_noise_savings([{**rows[0],**change}],10)


def test_public_static_order_matches_two_prespecified_delayed_schedules():
    model=torch.nn.Sequential(*[torch.nn.Linear(2,2) for _ in range(4)])
    for order,first,second in [(['0','1','2','3'],'0','1'),(['1','0','2','3'],'1','0')]:
        scheduler=StaticMask(model,'BU',[15,17],42,order)
        for t in range(1,26):
            count=group_counts(scheduler.decide(t))
            assert (count[first]==0)==(t>=15)
            assert (count[second]==0)==(t>=17)
            assert count['2']>0 and count['3']>0
    for invalid in [['0','0','2','3'],['0'],['0','1','2','other']]:
        with pytest.raises(ValueError):StaticMask(model,'BU',[15,17],42,invalid)
