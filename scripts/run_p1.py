"""P1 matrix: default engineering smoke; --full requires the P0 gate."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from src.p1_runner import run

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--full',action='store_true')
    p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--mask-seeds',nargs='+',type=int,default=[11,23,37])
    p.add_argument('--modes',nargs='+',choices=['NF','BU','TD','RAND','SAME-DIM','NO-DP'],default=['NF','BU','TD','RAND','SAME-DIM','NO-DP'])
    p.add_argument('--results-dir')
    args=p.parse_args()
    if args.full:
        gate=Path('reports/p0_gate.json')
        if not gate.exists() or json.loads(gate.read_text()).get('status')!='passed':p.error('P0 gate not passed; omit --full for engineering verification')
    seeds=args.seeds or ([42,43,44] if args.full else [42])
    root=Path(args.results_dir or f'results/p1_{"full" if args.full else "smoke"}_{time.time_ns()}')
    root.mkdir(parents=True,exist_ok=False)
    completed=[]
    for seed in seeds:
        for mode in args.modes:
            config=yaml.safe_load(Path('configs/p1_'+mode.lower().replace('-','_')+'.yaml').read_text())
            for mask_seed in args.mask_seeds if mode=='RAND' else [seed]:
                cfg=copy.deepcopy(config);cfg.update(seed=seed,mask_seed=mask_seed)
                if not args.full:cfg.update(engineering_only=True,clients=6,rounds=4,local_epochs=1,limit_train=600,limit_test=200,transition_round=3,freeze_rounds=[2,3])
                name=f'{mode}_seed{seed}_mask{mask_seed}'
                try:
                    path=run(cfg,root/name)
                    completed.append(dict(path=str(path),status='completed'))
                except Exception as exc:
                    completed.append(dict(path=str(root/name),status='failed',error=str(exc)))
                    (root/'matrix.json').write_text(json.dumps(completed,indent=2))
                    raise
                (root/'matrix.json').write_text(json.dumps(completed,indent=2))
    print(root)

if __name__=='__main__':main()
