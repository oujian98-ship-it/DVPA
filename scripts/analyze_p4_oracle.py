"""Separate trusted offline process; never read oracle files in DP training."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.p4_oracle import analyze_oracle, conditional_decomposition, oracle_gap
from src.p4_metrics import client_utility


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,help='Private offline JSON, not a public training transcript')
    parser.add_argument('--output',required=True,help='New private offline report file')
    parser.add_argument('--offline-only',action='store_true',required=True)
    args=parser.parse_args();data=json.loads(Path(args.input).read_text(encoding='utf-8'))
    report={'scope':'private offline oracle diagnostics; do not publish as a DP transcript'}
    if 'oracle' in data:
        report['oracle']=analyze_oracle(data['feedback'],data['oracle'],data['horizon'],offline_only=True)
    if 'conditional' in data:report['conditional']=conditional_decomposition(**data['conditional'],offline_only=True)
    if 'clients' in data:report['clients']=client_utility(data['clients'],offline_only=True)
    if 'oracle_baseline' in data:report['oracle_gap']=oracle_gap(data['oracle_baseline'],data['method_baseline'],offline_only=True)
    if len(report)==1:parser.error('No supported offline diagnostic payload')
    with Path(args.output).open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(args.output)


if __name__=='__main__':main()
