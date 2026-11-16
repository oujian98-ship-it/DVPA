"""Locked user-authorized NF equal vs NF 4A, full data, all registered seeds."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from scripts.validate_bottom_up import preflight
from src.upstream_runner import SOURCE
from src.p1_runner import run
from src.p4_evaluation import evaluate_manifest, write_evaluation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',required=True)
    args=parser.parse_args();root=Path(args.results_dir).resolve()
    root.mkdir(parents=True,exist_ok=False)
    equal=yaml.safe_load(Path('configs/p2_equal_nf.yaml').read_text())
    prior=yaml.safe_load(Path('configs/p2_4a_nf.yaml').read_text())
    excluded={'seed','mask_seed','allocation'}
    fixed={k:v for k,v in equal.items() if k not in excluded}
    if fixed!={k:v for k,v in prior.items() if k not in excluded}:raise ValueError('Unmatched experimental configurations')
    allocations={'EQUAL':equal['allocation'],'4A':prior['allocation']}
    seeds=[42,43,44]
    (root/'status.json').write_text(json.dumps(dict(state='privacy_preflight',completed=[],total=6),indent=2))
    preflight(fixed,root/'privacy_preflight.npz')
    sources=[p for p in SOURCE.rglob('*') if p.is_file()]
    sources.extend(Path('src').glob('*.py'))
    sources.extend([Path(__file__),Path('scripts/validate_bottom_up.py')])
    protocol=dict(kind='controlled_allocation_comparison_v1',
        authorization='User explicitly requested E2 NF+4A independent allocation validation; P0 gate unchanged',
        fixed_config=fixed,allocations=allocations,training_seeds=seeds,
        privacy_preflight='privacy_preflight.npz',
        privacy_preflight_sha256=hashlib.sha256((root/'privacy_preflight.npz').read_bytes()).hexdigest(),
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        primary_metric='final_test_acc',delta=1e-5,
        hypothesis='NF 4A minus NF equal_active; no layer freezing; independent allocation benefit',
        selection_policy='all six registered runs; no prior tuning, seed replacement or peak picking',
        privacy_scope='public FMNIST; per-run guarantees, not the combined privacy of publishing every run')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    manifest=dict(version=1,data_scope='public_benchmark',allow_engineering=False,include_local_cost=False,
        cases=[dict(id='E2_NF_4A',baseline='EQUAL',budget_matching='same_rdp',accuracy_target=70.,
            methods=[dict(id=name,runs=[dict(seed=s,replicate='default',path=f'{name}_seed{s}') for s in seeds]) for name in allocations])])
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    completed=[]
    for seed in seeds:
        for name,allocation in allocations.items():
            folder=root/f'{name}_seed{seed}'
            (root/'status.json').write_text(json.dumps(dict(state='running',current=folder.name,completed=completed,total=6),indent=2))
            cfg=copy.deepcopy(fixed);cfg.update(seed=seed,mask_seed=seed,allocation=allocation,validation_protocol=str(root/'protocol.json'))
            try:run(cfg,folder)
            except Exception as exc:
                (root/'status.json').write_text(json.dumps(dict(state='failed',current=folder.name,error=str(exc),completed=completed,total=6),indent=2));raise
            completed.append(folder.name)
    (root/'status.json').write_text(json.dumps(dict(state='training_completed_auditing',completed=completed,total=6),indent=2))
    result=evaluate_manifest(manifest,root)
    write_evaluation(result,root/'manifest.json',root/'analysis')
    case=result['cases'][0];pair=case['paired_comparisons'][0]
    conclusion=dict(completed_runs=6,method_table=case['main_table'],paired_comparison=pair,
                    privacy='All runs independently audited; full paired per-client RDP matched',
                    interpretation='Positive mean is a descriptive effect; three seeds cannot establish general or exact p<0.05 superiority')
    (root/'comparison.json').write_text(json.dumps(conclusion,indent=2))
    (root/'status.json').write_text(json.dumps(dict(state='completed',completed=completed,total=6),indent=2))
    print(json.dumps(conclusion,indent=2),flush=True)


if __name__=='__main__':main()
