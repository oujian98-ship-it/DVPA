"""Preview P2 by default. Training requires explicit --execute and existing gates."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from src.p2_allocation import allocation_plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/p2_4a_bu.yaml')
    parser.add_argument('--output',help='Preview JSON path, or a new training directory with --execute')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if cfg.get('validation_protocol'):parser.error('P1 locked protocols cannot authorize new P2 configurations')
    plan=allocation_plan(cfg)  # Validate all rounds before any dataset access.
    if not args.execute:
        if args.smoke:parser.error('--smoke requires --execute; default is preview only')
        content=json.dumps(plan,indent=2,allow_nan=False)
        if args.output:
            with Path(args.output).open('x',encoding='utf-8') as f:f.write(content)
            print(args.output)
        else:print(content)
        return
    if not args.output:parser.error('--execute requires a new output directory')
    if args.smoke:
        cfg.update(engineering_only=True,clients=6,rounds=4,local_epochs=1,
                   limit_train=600,limit_test=200,transition_round=3,freeze_rounds=[2,3])
    from src.p1_runner import run
    print(run(cfg,args.output))


if __name__=='__main__':main()
