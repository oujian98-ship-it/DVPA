"""Engineering plots from actual saved public outputs only."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--results-dir',default='results');args=p.parse_args()
    for file in Path(args.results_dir).rglob('rounds.csv'):
        with file.open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
        folder=file.parent/'figures';folder.mkdir(exist_ok=True)
        for column in ['test_acc','epsilon_max','active_param_fraction','noise_energy_expected']:
            if not rows or column not in rows[0]:
                continue
            fig,ax=plt.subplots(figsize=(6,4))
            ax.plot([int(r['round']) for r in rows],[float(r[column]) for r in rows],marker='o')
            ax.set(xlabel='Round',ylabel=column,title=file.parent.name)
            fig.tight_layout();fig.savefig(folder/f'{column}.png',dpi=150);plt.close(fig)
    summaries=[]
    import json
    for file in Path(args.results_dir).rglob('summary.json'):
        s=json.loads(file.read_text());summaries.append(dict(experiment=file.parent.name,seed=s['seed'],final_test_acc=s['final_test_acc'],best_test_acc=s['best_test_acc'],privacy_enabled=s.get('privacy_enabled'),normalization=s.get('data_normalization_version','source_minus_one_one' if 'source_training' in s.get('status','') else 'legacy_invalid_normalization'),validation_status=s.get('validation_status',s.get('status','unverified'))))
    if summaries:
        with (Path(args.results_dir)/'table_main.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)

if __name__=='__main__':main()
