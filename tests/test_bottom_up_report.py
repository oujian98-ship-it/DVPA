import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('bottom_up_report',Path(__file__).parents[1]/'scripts'/'report_bottom_up.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_saved_privacy_audit_rejects_missing_or_changed_rows(tmp_path):
    preflight=dict(orders=np.array([2.,3.]),targets=np.array([10.]),
                   rdp=np.array([[[.1,.2]],[[.2,.4]]]),
                   params=np.array([[[.5,2.,1.]],[[.6,3.,1.]]]),epsilon=np.array([[1.],[2.]]))
    np.savez(tmp_path/'accountant.npz',orders=preflight['orders'],targets=[10.],rdp=[[.2,.4]])
    rows=[dict(round=1,client_slot=0,q=.5,sigma=2.,clip=1.,target=10.,epsilon=1.),
          dict(round=2,client_slot=0,q=.6,sigma=3.,clip=1.,target=10.,epsilon=2.)]
    def save():
        with (tmp_path/'privacy.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(epsilon_n_final=[2.])
    save();module.audit_run(tmp_path,summary,preflight,2,1)
    rows[1]['sigma']=3.01;save()
    with pytest.raises(AssertionError):module.audit_run(tmp_path,summary,preflight,2,1)
    rows.pop();save()
    with pytest.raises(ValueError,match='Incomplete'):module.audit_run(tmp_path,summary,preflight,2,1)
