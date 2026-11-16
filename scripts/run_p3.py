"""Validate P3 configuration by default. No adaptive trajectory is invented."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from src.p3_freezing import feedback_preview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/p3_ema.yaml')
    parser.add_argument('--output',help='New preview JSON file, or training directory with --execute')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    preview=feedback_preview(cfg)
    if not args.execute:
        if args.smoke:parser.error('--smoke requires --execute; default is configuration only')
        content=json.dumps(preview,indent=2,allow_nan=False)
        if args.output:
            with Path(args.output).open('x',encoding='utf-8') as f:f.write(content)
            print(args.output)
        else:print(content)
        return
    if not args.output:parser.error('--execute requires a new output directory')
    if args.smoke:
        cfg.update(engineering_only=True,clients=6,rounds=4,local_epochs=1,
                   limit_train=600,limit_test=200,transition_round=3,freeze_rounds=[])
    from src.p1_runner import run
    print(run(cfg,args.output))


if __name__=='__main__':main()
