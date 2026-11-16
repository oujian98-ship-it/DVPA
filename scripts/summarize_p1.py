"""Aggregate actual P1 results and verify BU/SAME-DIM coordinate counts."""
import argparse
import csv
import json
from pathlib import Path
import statistics

def main():
    p=argparse.ArgumentParser();p.add_argument('--results-dir',required=True);args=p.parse_args()
    root=Path(args.results_dir);summaries=[];groups={};counts={}
    for path in sorted(root.glob('*/summary.json')):
        s=json.loads(path.read_text())
        with (path.parent/'rounds.csv').open() as f:rows=list(csv.DictReader(f))
        condition=s['mode'] if s['privacy_enabled'] or s['mode']=='NO-DP' else s['mode']+'_NO_DP'
        counts[(s['mode'],s['seed'],s['privacy_enabled'])]=[int(r['active_params']) for r in rows]
        record=dict(mode=s['mode'],seed=s['seed'],mask_seed=s['mask_seed'],final_test_acc=s['final_test_acc'],best_test_acc=s['best_test_acc'],budget_violation_rate=s['budget_violation_rate'],privacy_enabled=s['privacy_enabled'],status=s['status'],path=str(path.parent))
        summaries.append(record);groups.setdefault(condition,[]).append(s)
    for (mode,seed,private),values in counts.items():
        if mode=='SAME-DIM' and ('BU',seed,private) in counts and values!=counts[('BU',seed,private)]:
            raise RuntimeError('SAME-DIM does not match BU active dimensions')
    def write(file,rows):
        if rows:
            with file.open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(root/'table_runs.csv',summaries)
    aggregate=[]
    for mode,items in groups.items():
        # Average RAND mask replicates within each training seed FIRST; they
        # are not independent training seeds for confidence intervals.
        by_seed={}
        for s in items:by_seed.setdefault(s['seed'],[]).append(s['final_test_acc'])
        values=[statistics.mean(v) for v in by_seed.values()]
        aggregate.append(dict(mode=mode,training_seeds=len(values),runs=len(items),mean_final_acc=statistics.mean(values),std_across_training_seeds=statistics.stdev(values) if len(values)>1 else None,status='engineering_only' if any(s['status']=='engineering_only' for s in items) else 'experiment'))
    write(root/'table_ablation.csv',aggregate)
    print(json.dumps(dict(runs=len(summaries),matched_dimensions_checked=True,table=str(root/'table_ablation.csv')),indent=2))

if __name__=='__main__':main()
