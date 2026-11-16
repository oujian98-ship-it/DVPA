import pytest
from src.main import run

def test_later_stage_requires_p0_gate_before_loading_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config={'experiment':{'seed':42},'method':{'layer_freeze':'static_bottom_up','layer_clip':'equal_active'}}
    with pytest.raises(RuntimeError,match='P0 gate not passed'):
        run(config,tmp_path/'must_not_exist')
    assert not (tmp_path/'must_not_exist').exists()
