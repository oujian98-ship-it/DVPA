"""Finish FMNIST seeds43/44 for three methods; aggregate peak scores with42."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import contextlib
import copy
import csv
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import torch
import yaml

from src.brdp_source_runner import run as run_baseline,_json
from src.brdp_source_audit import audit as audit_baseline
from src.brdp_source_protocol import validate_config,load_datasets,partition_and_groups
from src.dvpa_config import ROOT,load_config,resolve,public_preflight
from src.dvpa_runner import run as run_dvpa
from src.dvpa_audit import audit as audit_dvpa
from src.dvpa_train_public import verify_validation
from src.metrics import evaluate
from src.st_runner import write_csv
from src.upstream_runner import source_model
from src.data import split_checksum
from scripts.run_dvpa_comparison import verify_identity,normalized_privacy,tensor_digest,dataset_digest
from scripts.run_dvpa_suite import verify_gate,verify_formal
from scripts.run_train_public_baselines import metric_record,TITLES,DVPA

PRIOR=ROOT/'results/trainpublic1000_fmnist_seed42_dvpa_idp_sayg_20261106'
REPORTS=ROOT/'reports/train_public_three_seeds_20261106'
OUTPUT=ROOT/'results/trainpublic1000_fmnist_seeds42_43_44_dvpa_idp_sayg_20261106'
LABELS=('DVPA','IDP_FedAvg','Time_Adaptive')


def jobs():
    reference=load_config(DVPA/'config.yaml');records=[]
    for seed in (43,44):
        cfg=copy.deepcopy(reference);cfg['base']['seed']=seed;cfg=resolve(cfg);verify_formal(cfg)
        records.append(dict(name=f'DVPA_seed{seed}',seed=seed,label='DVPA',config=cfg))
        for label,method in [('IDP_FedAvg','IDP_FEDAVG'),('Time_Adaptive','SPEND_AS_YOU_GO')]:
            base=copy.deepcopy(cfg['base']);base['method']=method
            records.append(dict(name=f'{label}_seed{seed}',seed=seed,label=label,config=validate_config(base)))
    return records


def prepare():
    torch.set_num_threads(1)
    REPORTS.mkdir(parents=True,exist_ok=True);(REPORTS/'configs').mkdir(exist_ok=True)
    reference=load_config(DVPA/'config.yaml');verify_validation(reference)
    train,test=load_datasets(reference['base'])
    if len(train)!=59000 or len(test)!=10000:raise ValueError('Expected shared 59000/10000 datasets')
    manifest=dict(dataset='fmnist',weight=10,requested_new_runs=6,seeds=[43,44],reuse_seed42=str(PRIOR),
                  training_examples=59000,public_examples=1000,evaluation_examples=10000,
                  public_file=reference['base']['public_train_file'],public_sha256=reference['base']['public_train_sha256'],
                  train_sha256=dataset_digest(train),test_sha256=dataset_digest(test),
                  metric='per-run maximum test accuracy over 25 rounds',std_ddof=1,
                  output=str(OUTPUT),privacy_caps=[10,20,30],delta=1e-5)
    for job in jobs():
        (REPORTS/'configs'/(job['name']+'.yaml')).write_text(yaml.safe_dump(job['config'],sort_keys=False),encoding='utf-8')
    _json(REPORTS/'plan.json',manifest)


def queue_manifest():
    common=verify_gate()
    paths=[Path(__file__),ROOT/'tests/test_train_public_three_seeds.py',REPORTS/'plan.json']
    paths+=list((REPORTS/'configs').glob('*.yaml'))
    if len(paths)!=9:raise ValueError('Prepare exactly six new run configurations')
    return dict(mechanism_sha256=common['source_sha256'],queue_sha256={
        str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def test_gate():
    before=queue_manifest()
    proc=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',
         'tests/test_train_public_three_seeds.py','--junitxml',str(REPORTS/'pytest.xml'),
         '--basetemp',str(REPORTS/'pytest_workspace')],cwd=ROOT,stdout=subprocess.PIPE,
         stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    (REPORTS/'pytest.log').write_text(proc.stdout,encoding='utf-8');print(proc.stdout,flush=True)
    cases=list(ET.parse(REPORTS/'pytest.xml').getroot().iter('testcase')) if proc.returncode==0 else []
    if proc.returncode or len(cases)!=3 or any(list(c) for c in cases) or before!=queue_manifest():
        raise RuntimeError('New queue tests failed or execution inputs changed')
    _json(REPORTS/'test_gate.json',dict(status='passed',tests_passed=3,manifest=before,
                                     mechanism_tests_passed=verify_gate()['tests_passed']))


def verify_queue_gate():
    gate=json.loads((REPORTS/'test_gate.json').read_text(encoding='utf-8'))
    if gate['status']!='passed' or gate['tests_passed']!=3 or gate['manifest']!=queue_manifest():
        raise RuntimeError('Exact current queue test gate required')
    return gate


def reused_records():
    previous=json.loads((PRIOR/'suite_state.json').read_text(encoding='utf-8'))
    if previous['status']!='completed' or len(previous['completed'])!=3:
        raise ValueError('All three audited seed42 records required')
    reference=load_config(DVPA/'config.yaml')['base'];rows=[]
    for item in previous['completed']:
        folder=Path(item['folder'])
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        checked=json.loads((folder/'audit.json').read_text(encoding='utf-8'))
        evaluation=json.loads((folder/'evaluation_recheck.json').read_text(encoding='utf-8'))
        if evaluation['examples']!=10000 or evaluation['accuracy']!=summary['final_test_accuracy']:
            raise ValueError('Reused seed42 evaluation mismatch')
        cfg=yaml.safe_load((folder/'config.yaml').read_text(encoding='utf-8'))
        base=cfg['base'] if item['label']=='DVPA' else cfg
        if any(base[k]!=v for k,v in reference.items() if k!='method'):
            raise ValueError('Seed42 does not use the requested shared protocol')
        impl=json.loads((folder/'implementation.json').read_text(encoding='utf-8'))
        manifest=impl['source_sha256']['source_snapshot'] if item['label']=='DVPA' else impl['source_sha256']
        for name,sha in manifest.items():
            if hashlib.sha256((folder/'source_snapshot'/name).read_bytes()).hexdigest()!=sha:
                raise ValueError('Frozen seed42 source snapshot modified')
        row=metric_record(folder,item['label'],summary,checked,True)
        row.update(seed=42,name=f"{item['label']}_seed42",evaluation_recheck=evaluation,
                   group_privacy=copy.deepcopy(item['group_privacy']))
        rows.append(row)
    if {r['label'] for r in rows}!=set(LABELS):raise ValueError('Missing seed42 method')
    return rows


def statistics(rows):
    keys=[(r['seed'],r['label']) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate seed/method in aggregation')
    result=[]
    for label in LABELS:
        group=[r for r in rows if r['label']==label]
        if not group:continue
        values=np.array([r['summary']['best_test_accuracy'] for r in group],dtype=float)
        if not np.isfinite(values).all():raise ValueError('Nonfinite peak accuracy')
        result.append(dict(label=label,n=len(values),peak_accuracy_mean=float(values.mean()),
            peak_accuracy_sample_std=float(values.std(ddof=1)) if len(values)>1 else None,
            complete_seed_set={r['seed'] for r in group}=={42,43,44}))
    return result


def report(out,state):
    rows=[];fees=[];paired=[];stats=statistics(state['completed'])
    lookup={(r['seed'],r['label']):r for r in state['completed']}
    for r in state['completed']:
        s=r['summary']
        rows.append(dict(seed=r['seed'],label=r['label'],peak_accuracy=s['best_test_accuracy'],peak_round=r['best_round'],
                         final_accuracy=s['final_test_accuracy'],reused=r['reused'],audit=r['audit']['status']))
        for g in r['group_privacy']:fees.append(dict(seed=r['seed'],label=r['label'],**g))
    for seed in (42,43,44):
        if all((seed,label) in lookup for label in LABELS):
            a=lookup[seed,'DVPA']['summary']['best_test_accuracy']
            paired.append(dict(seed=seed,dvpa_minus_idp_peak_pp=a-lookup[seed,'IDP_FedAvg']['summary']['best_test_accuracy'],
                               dvpa_minus_sayg_peak_pp=a-lookup[seed,'Time_Adaptive']['summary']['best_test_accuracy']))
    for name,values in [('results_by_seed.csv',rows),('mean_std_peak.csv',stats),('group_privacy.csv',fees),('paired_peak_improvements.csv',paired)]:
        if values:write_csv(out/name,values)
    _json(out/'comparison.json',dict(status=state['status'],new_completed=sum(not r['reused'] for r in state['completed']),
          expected_new_runs=6,completed_runs=len(state['completed']),expected_runs=9,
          metric='per-seed maximum accuracy over 25 rounds',std_ddof=1,
          results=state['completed'],statistics=stats,paired=paired,group_privacy=fees))
    lines=['# FMNIST：DVPA权重10、IDP-FedAvg与spend-as-you-go，seeds42/43/44','',
           f"状态：{state['status']}；新增完成{sum(not r['reused'] for r in state['completed'])}/6；含seed42共{len(state['completed'])}/9。当前：{state.get('current_job')}。",'',
           '固定训练集中的1000张为公开辅助集（64张曲率、468张验证、468张确认）；其余59000张分给客户端。官方10000张测试图像只用于评估。',
           '100客户端、Dirichlet α=0.1、25轮、每轮30本地epochs、batch125、lr0.001，原CNN与持久SGD/Cosine设置。',
           'DVPA信息价值权重固定10，第三项协方差校准关闭。两个baseline保留各自原生采样、裁剪和预算策略。',
           'seed42复用已完成并通过审计、复测的三次结果。seed43和44各方法全新训练，核验同seed相同分区、隐私组、模型初始化。',
           '主指标：各次25轮内最高准确率。先逐seed取峰值，再对三个seed计算均值与样本标准差（ddof=1）；没有改用末轮或跨轮平均。','',
           '| 方法 | n | seed42最高 | seed43最高 | seed44最高 | 峰值均值±样本标准差 |','|---|---:|---:|---:|---:|---:|']
    for s in stats:
        label=s['label'];cells=[f"{lookup[seed,label]['summary']['best_test_accuracy']:.4f}%" if (seed,label) in lookup else '—' for seed in (42,43,44)]
        cell=f"{s['peak_accuracy_mean']:.4f}%（仅1次）" if s['n']==1 else f"{s['peak_accuracy_mean']:.4f}% ± {s['peak_accuracy_sample_std']:.4f}个百分点"
        lines.append(f"| {TITLES[label]} | {s['n']} | {' | '.join(cells)} | {cell} |")
    lines+=['','| Seed | DVPA−IDP峰值，百分点 | DVPA−SAYG峰值，百分点 |','|---|---:|---:|']
    for p in paired:lines.append(f"| {p['seed']} | {p['dvpa_minus_idp_peak_pp']:+.4f} | {p['dvpa_minus_sayg_peak_pp']:+.4f} |")
    lines+=['','## 每次运行的实际会计费用','',
            '| Seed | 方法 | 组 | 认证ε上界 | 路径ε诊断 | 共同诊断α | 累计RDP |','|---|---|---:|---:|---:|---:|---:|']
    for g in fees:
        lines.append(f"| {g['seed']} | {TITLES[g['label']]} | {g['group']+1} | {g['epsilon_guarantee']:.6f} | {g['epsilon_path_diagnostic']:.6f} | {g['diagnostic_alpha']} | {g['rdp_path_at_diagnostic_alpha']:.8f} |")
    lines+=['','三组每次运行的共同ε上限为10/20/30，δ=10⁻⁵；实际费用不强行配平。DVPA路径ε诊断不是更小的全策略保证。',
            '费用按完整25轮核验，不因回看选取早期峰值而退还后续消耗。各次运行分别认证，九次联合发布需另行组合。',
            '这是同一评估集上的事后峰值，不是独立验证集选模后的测试成绩。真实DP采样与噪声使用独立秘密随机数。',
            '部分完成时，表中n不足3的均值和标准差仅为当前进度，不作完整三种子结论。','']
    (out/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')


def execute_job(job,out,expected):
    folder=out/job['name'];cfg=job['config']
    frozen=yaml.safe_load((REPORTS/'configs'/(job['name']+'.yaml')).read_text(encoding='utf-8'))
    frozen=resolve(frozen) if job['label']=='DVPA' else validate_config(frozen)
    if cfg!=frozen:raise ValueError('Frozen new-run config changed')
    if job['label']=='DVPA':summary=run_dvpa(cfg,folder);checked=audit_dvpa(folder);base=cfg['base']
    else:summary=run_baseline(cfg,folder,REPORTS/'operational_progress.json');checked=audit_baseline(folder);base=cfg
    _json(folder/'audit.json',checked)
    identity=verify_identity(folder,expected[job['seed']])
    _,test=load_datasets(base)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=source_model()('fmnist',125).to(device)
    model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True))
    loss,accuracy=evaluate(model,test,device)
    if len(test)!=10000 or not np.isclose(accuracy,summary['final_test_accuracy'],rtol=0,atol=1e-9):
        raise ValueError('Independent 10000-example final evaluation mismatch')
    evaluation=dict(examples=10000,accuracy=accuracy,cross_entropy=loss);_json(folder/'evaluation_recheck.json',evaluation)
    for g,row in enumerate(summary['group_privacy']):
        if row['epsilon_guarantee']>base['epsilons'][g]+1e-8:raise ValueError('Privacy cap exceeded')
        if job['label']=='DVPA' and row['rdp_spent']>row['rdp_cap']:raise ValueError('Adaptive RDP cap exceeded')
    reference=load_config(DVPA/'config.yaml');reference['base']['seed']=job['seed']
    value=metric_record(folder,job['label'],summary,checked,False)
    value.update(seed=job['seed'],name=job['name'],identity=identity,evaluation_recheck=evaluation,
                 group_privacy=normalized_privacy(folder,job['label'],public_preflight(reference).orders))
    return value


def queue(out,records,reused,expected,gate,execute=execute_job):
    out=Path(out).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Queue output must remain inside DVPA')
    out.mkdir(parents=True,exist_ok=False)
    (out/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes());_json(out/'test_gate.json',gate)
    _json(out/'expected_identity.json',{str(k):v for k,v in expected.items()})
    state=dict(status='running',expected_new_runs=6,expected_runs=9,completed=copy.deepcopy(reused),current_job=None,
               started_unix=time.time(),process_id=os.getpid(),dataset='fmnist',weight=10,seeds=[42,43,44])
    _json(out/'suite_state.json',state);report(out,state)
    try:
        for job in records:
            state.update(current_job=job['name'],current_job_started_unix=time.time())
            _json(out/'suite_state.json',state);report(out,state);print('START '+job['name'],flush=True)
            started=time.time()
            with (out/(job['name']+'.stdout.log')).open('w',encoding='utf-8',buffering=1) as stdout, (out/(job['name']+'.stderr.log')).open('w',encoding='utf-8',buffering=1) as stderr:
                with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):value=execute(job,out,expected)
            value['elapsed_seconds']=time.time()-started;state['completed'].append(value)
            _json(out/'suite_state.json',state);report(out,state)
            print(f"DONE {job['name']} best={value['summary']['best_test_accuracy']:.4f}%",flush=True)
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        state.update(status='completed',current_job=None,completed_unix=time.time())
    except BaseException as e:
        state.update(status='failed',failure_type=type(e).__name__,failure_message=str(e),failed_unix=time.time())
        _json(out/'suite_state.json',state);report(out,state);raise
    _json(out/'suite_state.json',state);report(out,state)
    return state


def start():
    torch.set_num_threads(1)
    gate=verify_queue_gate();records=jobs();reused=reused_records()
    plan=json.loads((REPORTS/'plan.json').read_text(encoding='utf-8'))
    cfg=records[0]['config'];verify_validation(cfg)
    train,test=load_datasets(cfg['base'])
    if dataset_digest(train)!=plan['train_sha256'] or dataset_digest(test)!=plan['test_sha256']:
        raise ValueError('Dataset changed after preflight')
    expected={}
    for seed in (43,44):
        base=next(j['config']['base'] for j in records if j['label']=='DVPA' and j['seed']==seed)
        parts,ids=partition_and_groups(base,train.tensors[1])
        torch.manual_seed(seed)
        for _ in range(100):source_model()('fmnist',125)
        initial=source_model()('fmnist',125).state_dict()
        expected[seed]=dict(parts=parts,group_ids=ids.tolist(),split_checksum=split_checksum(parts),initial_tensor_sha256=tensor_digest(initial))
    del train,test,initial
    return queue(OUTPUT,records,reused,expected,gate)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ['prepare','test','run']:parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    if args.test:test_gate()
    if args.run:start()
    if not any(vars(args).values()):parser.error('Use --prepare, --test or --run')
