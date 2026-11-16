"""Fail closed before full P1/P2/P3 experiments; smoke remains available."""
import json
import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml

def main():
    methods={'fedavg','dp_fedavg','idp_fedavg','paper_spend_as_you_go'}
    accepted={m:[] for m in methods}
    for file in Path('results').glob('*/summary.json'):
        summary=json.loads(file.read_text())
        config=yaml.safe_load((file.parent/'config.yaml').read_text())
        if 'method' not in config or 'data' not in config:
            # The author's literal tensor clipping cannot pass the joint-c gate.
            if (config.get('profile')!='joint_corrected' or config.get('clients')!=100
                or config.get('rounds')!=25 or config.get('clip')!=250
                or config.get('limit_train') or config.get('limit_test')
                or summary.get('parameters')!=582026 or summary.get('final_test_acc',0)<=15):
                continue
            method='paper_spend_as_you_go' if config.get('scheme')=='time_adaptive' else ('dp_fedavg' if len(set(config['epsilons']))==1 else 'idp_fedavg')
            rows=list(csv.DictReader((file.parent/'privacy.csv').open()))
            final=[r for r in rows if int(r['round'])==25]
            if len(final)!=100 or any(float(r['epsilon_reported'])>float(r['target'])+1e-6 for r in final):
                continue
            accepted[method].append({'seed':summary['seed'],'path':str(file.parent),'accuracy':summary['final_test_acc']})
            continue
        method=config['method']['temporal']
        if method not in accepted:
            continue
        metadata_path=file.parent/'implementation.json'
        metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if method!='fedavg' and metadata.get('sampling_rng')!='private_os_seed_v2':
            continue
        if (summary.get('data_normalization_version') != 'numpy_float32_v2'
            or config['data']['dataset'] != 'fmnist'
            or config['data'].get('limit_train') or config['data'].get('limit_test')
            or config['data']['clients'] != 100 or config['train']['rounds'] != 25
            or config['privacy']['average_clip_c'] != 250
            or config['model']['name'] != 'upstream_cnn'
            or config['method']['layer_freeze'] != 'none'
            or config['experiment'].get('purpose')):
            continue
        if method!='fedavg' and (summary.get('budget_violation_rate') != 0 or summary['final_test_acc']<=15):
            continue
        if method=='fedavg' and summary['final_test_acc']<60:
            continue
        accepted[method].append({'seed':summary['seed'],'path':str(file.parent),'accuracy':summary['final_test_acc']})
    missing={m:max(0,3-len({r['seed'] for r in runs})) for m,runs in accepted.items()}
    result={'status':'passed' if not any(missing.values()) else 'not_passed',
            'scope':'FMNIST P0 engineering gate; passage alone is not paper-equivalence certification',
            'accepted':accepted,'missing_independent_seeds':missing,
            'note':'Requires actual source architecture. Literal source tensor clipping, changed clip, reduced data, legacy normalization and failed runs cannot pass the joint-sensitivity gate.'}
    Path('reports/p0_gate.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
