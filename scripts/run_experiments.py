"""Explicit experiment runner; defaults to smoke, full runs require --full."""
import argparse
import copy
from pathlib import Path
import sys
import json
import subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.main import run

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['p0','p1','p2','p3'],default='p0')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--full',action='store_true')
    args=p.parse_args()
    if args.full and args.stage != 'p0':
        gate=Path('reports/p0_gate.json')
        if not gate.exists() or json.loads(gate.read_text()).get('status') != 'passed':
            p.error('P0 gate has not passed. Run scripts/check_p0_gate.py; only smoke is allowed for later stages.')
    if args.stage=='p1':
        command=[sys.executable,str(Path(__file__).with_name('run_p1.py')),'--seeds',*[str(s) for s in args.seeds]]
        if args.full:command.append('--full')
        subprocess.run(command,check=True)
        return
    base=yaml.safe_load(Path('configs/reference_original_fmnist.yaml').read_text())
    variants={
        'p0':[dict(temporal=m,layer_freeze='none',layer_clip='original') for m in ['fedavg','dp_fedavg','idp_fedavg','paper_spend_as_you_go']],
        'p2':[dict(layer_freeze='static_bottom_up',freeze_rounds=[10,18,None,None],layer_clip=m) for m in ['equal_active','param_count','exponential_depth_time','prior_decay_4A']],
        'p3':[dict(layer_freeze='dp_feedback',layer_clip='equal_active',feedback_score=m) for m in ['raw','debiased','ema']]
    }
    for seed in args.seeds:
        for i,variant in enumerate(variants[args.stage]):
            c=copy.deepcopy(base);c['experiment'].update(name=f'{args.stage}_{i}',seed=seed)
            c['method'].update(variant)
            if not args.full:
                c['experiment']['name']+='_smoke'
                c['experiment']['engineering_only']=True
                c['data'].update(clients=6,limit_train=600,limit_test=200)
                c['train'].update(rounds=4,local_epochs_or_iterations=1)
                c['privacy']['transition_round']=3
                c['method']['freeze_rounds']=[2,3,None,None]
                c['method'].update(warmup_rounds=1,patience=1)
            run(c)

if __name__=='__main__':main()
