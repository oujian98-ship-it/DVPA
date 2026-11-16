"""Evaluate saved runs only; manifest checking never opens run directories."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.p4_evaluation import validate_manifest, evaluate_manifest, write_evaluation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output')
    parser.add_argument('--check-manifest',action='store_true')
    args=parser.parse_args();path=Path(args.manifest)
    manifest=validate_manifest(json.loads(path.read_text(encoding='utf-8')))
    if args.check_manifest:
        print('Manifest valid; no runs read, no evaluation or training performed.');return
    if not args.output:parser.error('--output must name a new analysis directory')
    result=evaluate_manifest(manifest,path.resolve().parent)
    print(write_evaluation(result,path,args.output))


if __name__=='__main__':main()
