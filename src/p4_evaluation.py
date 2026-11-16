"""Manifest-driven evaluation; explicit registration prevents silent seed selection."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import math
from .p4_audit import audit_run, match_runs
from .p4_metrics import mean_interval, paired_comparison, holm_adjust


def validate_manifest(manifest):
    if type(manifest.get('version')) is not int or manifest['version']!=1 or not isinstance(manifest.get('cases'),list) or not manifest['cases']:raise ValueError('Expected nonempty P4 manifest v1')
    if set(manifest)-{'version','cases','data_scope','include_local_cost','allow_engineering'}:raise ValueError('Unknown manifest fields')
    for key in ['allow_engineering','include_local_cost']:
        if key in manifest and type(manifest[key]) is not bool:raise ValueError('Manifest flags must be boolean')
    cases=set()
    for case in manifest['cases']:
        all_paths=set()
        if not isinstance(case.get('id'),str) or not case['id'] or not isinstance(case.get('methods'),list) or not case['methods']:
            raise ValueError('Expected a named case with methods')
        if set(case)-{'id','baseline','budget_matching','accuracy_target','methods','privacy_curve_group'}:raise ValueError('Unknown case fields')
        if 'privacy_curve_group' in case and (not isinstance(case['privacy_curve_group'],str) or not case['privacy_curve_group']):raise ValueError('Invalid privacy curve group')
        if case.get('accuracy_target') is not None:
            target=case['accuracy_target']
            if not isinstance(target,(int,float)) or not math.isfinite(target) or not 0<=target<=100:raise ValueError('Invalid accuracy target')
        if case['id'] in cases:raise ValueError('Duplicate case id')
        cases.add(case['id']);ids=set();seed_sets=[]
        if case.get('budget_matching','same_rdp') not in {'same_rdp','same_epsilon','same_cap'}:raise ValueError('Unknown budget matching')
        for method in case['methods']:
            if not isinstance(method.get('id'),str) or not method['id']:raise ValueError('Expected a method id')
            if set(method)-{'id','runs','replicate_type'}:raise ValueError('Unknown method fields')
            if method.get('replicate_type','run') not in {'run','mask_seed'}:raise ValueError('Unknown replicate type')
            if method['id'] in ids or not method.get('runs'):raise ValueError('Duplicate/empty method')
            ids.add(method['id']);seen=set();replicates={};mask_seeds={}
            for run in method['runs']:
                if set(run)-{'seed','replicate','path','mask_seed'}:raise ValueError('Unknown run registration fields')
                seed=run['seed'];rep=run.get('replicate','default');path=run['path']
                if type(seed) is not int or not isinstance(rep,str) or not isinstance(path,str) or not path:
                    raise ValueError('Invalid registered seed/replicate/path')
                if (seed,rep) in seen or path in all_paths:raise ValueError('Duplicate run/replicate')
                seen.add((seed,rep));all_paths.add(path);replicates.setdefault(seed,set()).add(rep)
                if 'mask_seed' in run and type(run['mask_seed']) is not int:raise ValueError('Invalid registered mask seed')
                if method.get('replicate_type')=='mask_seed':
                    if 'mask_seed' not in run or run['mask_seed'] in mask_seeds.setdefault(seed,set()):raise ValueError('Mask replicates require distinct registered mask seeds')
                    mask_seeds[seed].add(run['mask_seed'])
            if any(v!=next(iter(replicates.values())) for v in replicates.values()):raise ValueError('Replicates must be registered equally across seeds')
            if mask_seeds and any(v!=next(iter(mask_seeds.values())) for v in mask_seeds.values()):raise ValueError('Mask seed sets must match across training seeds')
            seed_sets.append(set(replicates))
        if case['baseline'] not in ids or len(ids)<2 or any(v!=seed_sets[0] for v in seed_sets):
            raise ValueError('Baseline and paired seed sets must match')
    if manifest.get('include_local_cost') and manifest.get('data_scope')!='public_benchmark':
        raise ValueError('Local costs require explicit public benchmark scope')
    return manifest


def evaluate_manifest(manifest, base):
    validate_manifest(manifest);base=Path(base);cases=[];all_runs=[];all_rounds=[];all_layers=[];all_privacy=[];all_feedback=[]
    curve_signatures={};utility=[];curve_status=[]
    for case in manifest['cases']:
        resolved=set()
        loaded={};registration=[];baseline=case['baseline'];methods=[m['id'] for m in case['methods']]
        for method in case['methods']:
            for entry in method['runs']:
                path=(base/entry['path']).resolve()
                if path in resolved:raise ValueError('Same physical run registered twice')
                resolved.add(path)
                reg=dict(case=case['id'],method=method['id'],seed=entry['seed'],replicate=entry.get('replicate','default'),path=str(path))
                if not (path/'summary.json').exists():
                    registration.append(dict(**reg,status='missing_or_incomplete'));continue
                r=audit_run(path,accuracy_target=case.get('accuracy_target'),
                            allow_engineering=manifest.get('allow_engineering',False),
                            include_local_cost=manifest.get('include_local_cost',False))
                if r['seed']!=entry['seed']:raise ValueError('Run seed differs from registration')
                if 'mask_seed' in entry and r['_config'].get('mask_seed')!=entry['mask_seed']:raise ValueError('Run mask seed differs from registration')
                loaded.setdefault((method['id'],entry['seed']),[]).append(r)
                registration.append(dict(**reg,status='audited'))
                all_runs.append({**{k:v for k,v in r.items() if not k.startswith('_')},**reg})
                for key,destination in [('_rounds',all_rounds),('_layers',all_layers),('_privacy_groups',all_privacy),('_feedback',all_feedback)]:
                    destination.extend(dict(**row,case=case['id'],method=method['id'],seed=entry['seed'],replicate=reg['replicate']) for row in r[key])
        complete_seeds=[];seed_rows=[]
        seeds=sorted({e['seed'] for e in case['methods'][0]['runs']})
        for seed in seeds:
            if loaded.get((baseline,seed)):
                for method in methods:
                    for r in loaded.get((method,seed),[]):match_runs(loaded[(baseline,seed)][0],r,case.get('budget_matching','same_rdp'))
            complete=all(len(loaded.get((m['id'],seed),[]))==sum(e['seed']==seed for e in m['runs']) for m in case['methods'])
            if not complete:continue
            ref=loaded[(baseline,seed)][0]
            for method in methods:
                for r in loaded[(method,seed)]:match_runs(ref,r,case.get('budget_matching','same_rdp'))
            complete_seeds.append(seed)
            for method in methods:
                values=loaded[(method,seed)]
                seed_rows.append(dict(case=case['id'],method=method,seed=seed,replicates=len(values),
                                      final_test_acc=statistics.mean(r['final_test_acc'] for r in values)))
        table=[];cost=[]
        for method in methods:
            s=mean_interval([r['final_test_acc'] for r in seed_rows if r['method']==method])
            losses=[statistics.mean(r['test_loss'] for r in loaded[(method,seed)]) for seed in complete_seeds]
            best=[statistics.mean(r['best_test_acc'] for r in loaded[(method,seed)]) for seed in complete_seeds]
            table.append(dict(case=case['id'],method=method,training_seeds=s['n'],mean_final_acc=s['mean'],std=s['std'],ci95=s['ci95'],
                              mean_test_loss=statistics.mean(losses) if losses else None,mean_best_acc=statistics.mean(best) if best else None))
            for metric in ['communication_MB','communication_expected_MB','wall_time_sec','train_FLOPs','rounds_to_target']:
                values=[];missing=False
                for seed in complete_seeds:
                    reps=[r[metric] for r in loaded[(method,seed)]]
                    if any(v is None for v in reps):missing=True;continue
                    values.append(statistics.mean(reps))
                # Unreached thresholds are censored, not averaged over successes.
                statistic=mean_interval(values) if not missing else mean_interval([])
                cost.append(dict(case=case['id'],method=method,metric=metric,**statistic,
                                 status='complete' if not missing and values else 'missing_or_censored'))
        pairs=[]
        for method in methods:
            if method==baseline:continue
            a=[r['final_test_acc'] for r in seed_rows if r['method']==method]
            b=[r['final_test_acc'] for r in seed_rows if r['method']==baseline]
            pairs.append(dict(comparison=f'{method} - {baseline}',**paired_comparison(a,b)))
        for pair,adjusted in zip(pairs,holm_adjust([p['exact_two_sided_sign_flip_p'] for p in pairs])):
            pair['holm_adjusted_p']=adjusted
        complete=all(r['status']=='audited' for r in registration)
        curve_group=case.get('privacy_curve_group')
        if curve_group and complete and complete_seeds:
            reference=loaded[(baseline,complete_seeds[0])][0]
            ignored={'seed','mask_seed','mode','allocation','feedback','freeze_rounds','validation_protocol','epsilons'}
            settings={k:v for k,v in reference['_config'].items() if k not in ignored}
            signature=json.dumps([settings,reference['_model'],{seed:loaded[(baseline,seed)][0]['_split'] for seed in complete_seeds},sorted(methods)],sort_keys=True)
            if curve_group in curve_signatures and signature!=curve_signatures[curve_group]:raise ValueError('Privacy curve cases have unmatched model/data/training/seeds')
            curve_signatures[curve_group]=signature
        if curve_group:
            eps=[r['epsilon_max'] for (m,s),runs in loaded.items() for r in runs]
            homogeneous=all(abs(r['epsilon_max']-r['epsilon_min'])<=1e-9 for runs in loaded.values() for r in runs)
            common_epsilon=bool(eps) and max(eps)-min(eps)<=1e-9
            status='ready' if complete and homogeneous and common_epsilon else 'incomplete_or_heterogeneous_epsilon'
            curve_status.append(dict(case=case['id'],group=curve_group,status=status))
            if status=='ready':
                utility.extend(dict(group=curve_group,case=case['id'],method=r['method'],epsilon=eps[0],
                                    mean_final_acc=r['mean_final_acc'],std=r['std']) for r in table)
        cases.append(dict(id=case['id'],status='complete' if complete else 'incomplete',
                          budget_matching=case.get('budget_matching','same_rdp'),registered_seeds=seeds,
                          complete_seeds=complete_seeds,registration=registration,seed_table=seed_rows,
                          main_table=table,cost_table=cost,paired_comparisons=pairs,
                          conclusion='No superiority claim: descriptive evaluation only; inspect paired effects, uncertainty and scope.'))
    return dict(version=1,status='complete' if all(c['status']=='complete' for c in cases) else 'incomplete',
                scope='offline saved-log evaluation; no training or new privacy queries',cases=cases,
                runs=all_runs,rounds=all_rounds,layers=all_layers,privacy_groups=all_privacy,feedback=all_feedback,
                privacy_utility=utility,privacy_curve_status=curve_status)


def write_evaluation(result, manifest_path, output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    manifest_path=Path(manifest_path)
    result=dict(result,manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    (out/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    (out/'manifest.json').write_bytes(manifest_path.read_bytes())
    tables={'table_runs':result['runs'],'rounds':result['rounds'],'layers':result['layers'],'privacy_groups':result['privacy_groups'],'feedback':result['feedback'],'privacy_utility':result['privacy_utility']}
    for field,name in [('main_table','table_main'),('main_table','table_ablation'),('cost_table','table_cost'),('seed_table','table_seeds')]:
        tables[name]=[r for c in result['cases'] for r in c[field]]
    tables['table_paired']=[dict(case=c['id'],**row) for c in result['cases'] for row in c['paired_comparisons']]
    for name,rows in tables.items():
        if not rows:continue
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (out/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=['# P4 saved-log evaluation','',f'Status: {result["status"]}. No training was performed.','',
           'RAND/mask replicates are averaged within training seed. Incomplete seed blocks are excluded from paired tables and listed explicitly. Missing values are not zero.','']
    for c in result['cases']:
        lines.extend([f'## {c["id"]}','',f'{c["status"]}; budget matching: {c["budget_matching"]}; complete seeds: {c["complete_seeds"]}.','',
                      '| Method | Training seeds | Final accuracy mean (%) | Std |','|---|---:|---:|---:|'])
        for r in c['main_table']:lines.append(f'| {r["method"]} | {r["training_seeds"]} | {r["mean_final_acc"]} | {r["std"]} |')
        lines.extend(['','| Paired comparison | Mean gap (pp) | t 95% CI (pp) | Exact two-sided p | Holm p |',
                      '|---|---:|---|---:|---:|'])
        for p in c['paired_comparisons']:
            lines.append(f'| {p["comparison"]} | {p["mean"]} | {p["ci95"]} | {p["exact_two_sided_sign_flip_p"]} | {p["holm_adjusted_p"]} |')
        lines.extend(['',c['conclusion'],'','Expected communication is not measured traffic. Local runtime requires public benchmark scope; unavailable FLOPs remain null.',''])
    lines.extend(['Budgets are per run, not the combined privacy cost of publishing all runs. Saved-artifact checks do not certify the mechanism or establish paper reproduction. Three seed pairs cannot yield an exact two-sided sign-flip p below 0.25.'])
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return out
