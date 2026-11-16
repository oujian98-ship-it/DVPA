"""Plot a completed P4 evaluation artifact; never initiate evaluation/training."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.p4_plots import plot_evaluation, plot_oracle_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--evaluation');source.add_argument('--oracle')
    p.add_argument('--offline-only',action='store_true');p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.oracle:
        if not args.offline_only:p.error('--oracle requires --offline-only')
        result=plot_oracle_report(json.loads(Path(args.oracle).read_text(encoding='utf-8')),args.output,offline_only=True)
    else:result=plot_evaluation(json.loads(Path(args.evaluation).read_text(encoding='utf-8')),args.output)
    print(f'Created {len(result["produced"])} plots; unavailable evidence is listed in plot_status.json')


if __name__=='__main__':main()
