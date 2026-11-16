import copy
import json
from pathlib import Path

import pytest
import yaml

from src.st_fourway import METHODS, configurations, prepare, verify_lock


def base():
    return yaml.safe_load((Path(__file__).parents[1]/'configs/st_budget_fmnist.yaml').read_text())


def test_factorial_changes_only_intended_axes():
    cfg = base()
    cells = configurations(cfg)
    assert tuple(cells) == METHODS
    for method, cell in cells.items():
        assert cell['experiment']['seed'] == 42
        assert cell['budget']['feedback'] is False
        for section in ('privacy', 'clipping', 'freezing', 'data', 'model', 'train'):
            assert cell[section] == cfg[section]
        temporal = cell['budget']['time_weights']
        assert temporal == (cfg['budget']['time_weights'] if method in {'TIME_ONLY', 'JOINT'} else [1.]*25)
        if method in {'UNIFORM', 'TIME_ONLY'}:
            assert set(cell['budget']['layer_decay'].values()) == {0.}
        else:
            assert cell['budget']['layer_decay'] == cfg['budget']['layer_decay']
    assert cfg == base()


def test_lock_checks_configs_and_same_constraints_without_training(tmp_path):
    root = tmp_path/'fourway'
    prepare(base(), root)
    verify_lock(root)
    plans = json.loads((root/'preflight.json').read_text())
    for method, plan in plans.items():
        assert plan['privacy'][-1]['epsilon_accounting_value'] <= 10.
        assert sum(r['rho_increment'] for r in plan['privacy']) == pytest.approx(plans['UNIFORM']['rho_cap'])
    path = root/'configs/JOINT.yaml'
    path.write_text(path.read_text()+'\n# unexpected edit\n')
    with pytest.raises(RuntimeError, match='config changed'):
        verify_lock(root)


@pytest.mark.parametrize('seed', [42, 43, 44])
def test_requested_seeds_and_no_unregistered_feedback(seed):
    cfg = base()
    cfg['experiment']['seed'] = seed
    assert all(cell['experiment']['seed'] == seed for cell in configurations(cfg).values())
    wrong = copy.deepcopy(cfg)
    wrong['data']['dataset'] = 'synthetic'
    with pytest.raises(ValueError, match='FMNIST'):
        configurations(wrong)
    cfg['budget']['feedback'] = True
    with pytest.raises(ValueError, match='feedback'):
        configurations(cfg)
