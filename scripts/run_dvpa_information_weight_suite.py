"""Complete the requested five 10x-VOI runs, retaining the completed FMNIST42."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'

import argparse
import contextlib
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

from src.brdp_source_runner import _json
from src.brdp_source_protocol import load_datasets
from src.dvpa_config import ROOT,load_config,public_preflight
from src.dvpa_dataset_support import baseline_evaluation,verify_public_validation,verify_forecast_validation
from src.dvpa_runner import run
from src.dvpa_audit import audit
from src.metrics import evaluate
from src.st_runner import write_csv
from src.upstream_runner import source_model
from scripts.run_dvpa_comparison import verify_identity,normalized_privacy
from scripts.run_dvpa_datasets import dataset_digest
from scripts.run_dvpa_suite import verify_formal,verify_gate

PRIOR=ROOT/'results/dvpa_fmnist_mnist_seeds42_43_44_20261105'
REUSED=ROOT/'results/dvpa_information_weight10_fmnist_seed42_20261106/DVPA_weight10_seed42'
REPORTS=ROOT/'reports/voi_weight_suite_20261106'
OUTPUT=ROOT/'results/dvpa_information_weight10_fmnist_mnist_seeds42_43_44_20261106'
REQUESTED=(('fmnist',43),('fmnist',44),('mnist',42),('mnist',43),('mnist',44))
LABEL='DVPA_weight10'
METHODS=(LABEL,'DVPA','IDP_FedAvg','Time_Adaptive')
TITLES={LABEL:'DVPA，信息价值权重10','DVPA':'原版DVPA，权重1',
        'IDP_FedAvg':'IDP-FedAvg适配版','Time_Adaptive':'spend-as-you-go适配版'}


def jobs():
    rows=[]
    for dataset,seed in REQUESTED:
        cfg=load_config(PRIOR/f'{dataset}_DVPA_seed{seed}'/'config.yaml')
        cfg['controller']['information_value_weight']=10.0
        verify_formal(cfg)
        rows.append(dict(name=f'{dataset}_DVPA_weight10_seed{seed}',dataset=dataset,
                         seed=seed,label=LABEL,config=cfg))
    return rows


def prepare():
    REPORTS.mkdir(parents=True,exist_ok=True)
    cfgdir=REPORTS/'configs';cfgdir.mkdir(exist_ok=True)
    for job in jobs():
        verify_public_validation(job['config']);verify_forecast_validation(job['config'])
        (cfgdir/(job['name']+'.yaml')).write_text(yaml.safe_dump(job['config'],sort_keys=False),encoding='utf-8')
    _json(REPORTS/'plan.json',dict(weight=10.0,new_runs=5,expected_total_runs=6,
          requested=[dict(dataset=d,seed=s) for d,s in REQUESTED],reuse=str(REUSED),
          reference_suite=str(PRIOR),output=str(OUTPUT),covariance_calibration_enabled=False,
          evaluation_examples=9000,privacy_caps=[10,20,30],delta=1e-5,
          baseline_policy='reuse the 18 existing reference runs, do not retrain baselines'))


def test_manifest():
    common=verify_gate()
    paths=[Path(__file__),ROOT/'tests/test_dvpa_information_weight_suite.py',REPORTS/'plan.json']
    paths+=list((REPORTS/'configs').glob('*.yaml'))
    if len(paths)!=8:raise ValueError('Prepare exactly five configurations first')
    return dict(mechanism_gate=common['source_sha256'],queue_sha256={
        p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def test_gate():
    before=test_manifest()
    proc=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',
                         'tests/test_dvpa_information_weight_suite.py','--junitxml',str(REPORTS/'pytest.xml'),
                         '--basetemp',str(REPORTS/'pytest_workspace')],cwd=ROOT,
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    (REPORTS/'pytest.log').write_text(proc.stdout,encoding='utf-8');print(proc.stdout,flush=True)
    if proc.returncode:raise RuntimeError('Weight10 queue tests failed')
    suites=ET.parse(REPORTS/'pytest.xml').getroot()
    cases=list(suites.iter('testcase'))
    if len(cases)!=3 or any(list(c) for c in cases) or before!=test_manifest():
        raise RuntimeError('Incomplete tests or changed queue sources')
    _json(REPORTS/'test_gate.json',dict(status='passed',tests_passed=3,manifest=before,
          existing_mechanism_tests_passed=verify_gate()['tests_passed']))


def verify_queue_gate():
    value=json.loads((REPORTS/'test_gate.json').read_text(encoding='utf-8'))
    if value['status']!='passed' or value['tests_passed']!=3 or value['manifest']!=test_manifest():
        raise RuntimeError('Run queue tests for exact current sources/configs')
    return value


def reuse_record():
    summary=json.loads((REUSED/'summary.json').read_text(encoding='utf-8'))
    checked=json.loads((REUSED/'audit.json').read_text(encoding='utf-8'))
    evaluation=json.loads((REUSED/'evaluation_recheck.json').read_text(encoding='utf-8'))
    cfg=load_config(REUSED/'config.yaml');verify_formal(cfg)
    source=json.loads((REUSED/'implementation.json').read_text(encoding='utf-8'))
    for name,sha in source['source_sha256']['source_snapshot'].items():
        if hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()!=sha or hashlib.sha256((REUSED/'source_snapshot'/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Reused FMNIST42 does not match current mechanism sources')
    if (summary['status']!='completed' or summary['rounds']!=25 or checked['status']!='passed'
        or cfg['controller']['information_value_weight']!=10.0 or cfg['base']['dataset']!='fmnist'
        or cfg['base']['seed']!=42 or evaluation['examples']!=9000
        or evaluation['accuracy']!=summary['final_test_accuracy']):
        raise ValueError('Cannot reuse incomplete or unverified FMNIST42 result')
    public=public_preflight(cfg)
    return dict(name='fmnist_DVPA_weight10_seed42',dataset='fmnist',seed=42,label=LABEL,
                folder=str(REUSED),reused=True,summary=summary,audit=checked,
                group_privacy=normalized_privacy(REUSED,'DVPA',public.orders),evaluation_recheck=evaluation)


def statistics(rows):
    keys=[(r['dataset'],r['seed'],r['label']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate run in aggregation')
    out=[]
    for dataset in ('fmnist','mnist'):
        for label in METHODS:
            selected=[r for r in rows if r['dataset']==dataset and r['label']==label]
            if not selected:continue
            a=np.array([r['summary']['final_test_accuracy'] for r in selected])
            out.append(dict(dataset=dataset,label=label,n=len(a),mean=float(a.mean()),
                            sample_std=float(a.std(ddof=1)) if len(a)>1 else None,
                            complete_seed_set={r['seed'] for r in selected}=={42,43,44}))
    return out


def report(out,state):
    old=json.loads((PRIOR/'suite_state.json').read_text(encoding='utf-8'))['completed']
    allrows=old+state['completed'];stats=statistics(allrows)
    lookup={(r['dataset'],r['seed'],r['label']):r for r in allrows}
    values=[];fees=[];paired=[]
    for r in allrows:
        values.append(dict(dataset=r['dataset'],seed=r['seed'],label=r['label'],
                           final_accuracy=r['summary']['final_test_accuracy'],best_accuracy=r['summary']['best_test_accuracy']))
        for g in r['group_privacy']:
            fees.append(dict(dataset=r['dataset'],seed=r['seed'],label=r['label'],**g))
    for r in state['completed']:
        d,s=r['dataset'],r['seed'];acc=r['summary']['final_test_accuracy']
        paired.append(dict(dataset=d,seed=s,weight10_minus_weight1_pp=acc-lookup[d,s,'DVPA']['summary']['final_test_accuracy'],
                            weight10_minus_idp_pp=acc-lookup[d,s,'IDP_FedAvg']['summary']['final_test_accuracy'],
                            weight10_minus_sayg_pp=acc-lookup[d,s,'Time_Adaptive']['summary']['final_test_accuracy']))
    for name,content in [('results_by_seed.csv',values),('mean_std.csv',stats),('group_privacy.csv',fees),('paired_improvements.csv',paired)]:
        write_csv(out/name,content)
    _json(out/'comparison.json',dict(status=state['status'],new_completed=sum(not r.get('reused',False) for r in state['completed']),
          expected_new_runs=5,completed_weight10=len(state['completed']),expected_weight10=6,
          results=allrows,statistics=stats,paired=paired,group_privacy=fees))
    lines=['# 信息价值权重10：FMNIST与MNIST三个种子汇总','',
           f"状态：{state['status']}；新增实验完成{sum(not r.get('reused',False) for r in state['completed'])}/5；含已完成FMNIST42共{len(state['completed'])}/6。当前：{state.get('current_job')}。",'',
           '保持10倍信息价值权重，第三项协方差校准关闭。100客户端、25轮、每轮30本地epochs，固定9,000张评估。',
           'FMNIST42权重10复用已完成并通过审计的71.4556%结果；原版DVPA和两个baseline复用20261105批次，未重跑。',
           '同数据集同seed核验相同分区、分组和模型初始化；真实DP噪声独立。MNIST沿用固定迁移参数与公开数值检查。',
           '主指标为第25轮；标准差为样本标准差ddof=1。权重10额外偏重信息，不等于原始预计总失真。','']
    for d in ('fmnist','mnist'):
        lines += [f'## {d.upper()}','', '| 方法 | n | seed42 | seed43 | seed44 | 均值±样本标准差 |','|---|---:|---:|---:|---:|---:|']
        for label in METHODS:
            stat=next((r for r in stats if r['dataset']==d and r['label']==label),None)
            cells=[f"{lookup[d,s,label]['summary']['final_test_accuracy']:.4f}%" if (d,s,label) in lookup else '—' for s in (42,43,44)]
            mean='—' if stat is None else (f"{stat['mean']:.4f}%（仅1次）" if stat['n']==1 else f"{stat['mean']:.4f}% ± {stat['sample_std']:.4f}个百分点")
            lines.append(f"| {TITLES[label]} | {stat['n'] if stat else 0} | {' | '.join(cells)} | {mean} |")
        lines+=['','| Seed | 权重10−原版，百分点 | 权重10−IDP，百分点 | 权重10−SAYG，百分点 |','|---|---:|---:|---:|']
        for r in paired:
            if r['dataset']==d:lines.append(f"| {r['seed']} | {r['weight10_minus_weight1_pp']:+.4f} | {r['weight10_minus_idp_pp']:+.4f} | {r['weight10_minus_sayg_pp']:+.4f} |")
        lines.append('')
    lines+=['## 权重10各次运行的实际费用','',
            '| 数据集 | Seed | 组 | α | 累计RDP | 路径ε诊断 | 全策略ε上限 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in state['completed']:
        for g in r['group_privacy']:
            lines.append(f"| {r['dataset']} | {r['seed']} | {g['group']+1} | {g['diagnostic_alpha']} | {g['rdp_path_at_diagnostic_alpha']:.8f} | {g['epsilon_path_diagnostic']:.6f} | {g['epsilon_guarantee']:.0f} |")
    lines+=['','δ=10⁻⁵，三组每次运行分别核验10/20/30。路径ε仅作诊断，不是更小的全策略保证；六次运行联合发布不沿用单次预算。',
            '实际费用不强行配平。未凑齐三个种子时，不将部分均值解释成稳定提升。','']
    (out/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')


def execute_job(job,out):
    cfg=load_config(REPORTS/'configs'/(job['name']+'.yaml'))
    if cfg!=job['config']:raise ValueError('Frozen job configuration changed')
    folder=out/job['name'];summary=run(cfg,folder);checked=audit(folder)
    _json(folder/'audit.json',checked)
    inputs=json.loads((PRIOR/'formal_inputs.json').read_text(encoding='utf-8'))
    expected=inputs['expected_identity'][f"{job['dataset']}_{job['seed']}"]
    identity=verify_identity(folder,expected)
    _,test=load_datasets(cfg['base']);test,_=baseline_evaluation(cfg['base'],test)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=source_model()(job['dataset'],125).to(device)
    model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True))
    loss,accuracy=evaluate(model,test,device)
    if len(test)!=9000 or not np.isclose(accuracy,summary['final_test_accuracy'],rtol=0,atol=1e-9):
        raise ValueError('Independent final model evaluation mismatch')
    evaluation=dict(examples=len(test),accuracy=accuracy,cross_entropy=loss)
    _json(folder/'evaluation_recheck.json',evaluation)
    fees=normalized_privacy(folder,'DVPA',public_preflight(cfg).orders)
    for g,row in enumerate(summary['group_privacy']):
        if row['rdp_spent']>row['rdp_cap'] or row['epsilon_guarantee']>cfg['base']['epsilons'][g]:
            raise ValueError('Privacy cap exceeded')
    return dict(**{k:job[k] for k in ('name','dataset','seed','label')},folder=str(folder),reused=False,
                summary=summary,audit=checked,identity=identity,group_privacy=fees,evaluation_recheck=evaluation)


def queue(out,records,reused,execute=execute_job,gate=None):
    out=Path(out).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must remain inside DVPA')
    out.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',expected_new_runs=5,expected_runs=6,completed=[reused],
               current_job=None,started_unix=time.time(),process_id=os.getpid(),weight=10.0)
    (out/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes())
    if gate is not None:_json(out/'test_gate.json',gate)
    _json(out/'suite_state.json',state);report(out,state)
    try:
        for job in records:
            state.update(current_job=job['name'],current_job_started_unix=time.time())
            _json(out/'suite_state.json',state);report(out,state)
            print('START '+job['name'],flush=True)
            started=time.time()
            with (out/(job['name']+'.stdout.log')).open('w',encoding='utf-8',buffering=1) as stdout, (out/(job['name']+'.stderr.log')).open('w',encoding='utf-8',buffering=1) as stderr:
                with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                    value=execute(job,out)
            value['elapsed_seconds']=time.time()-started
            state['completed'].append(value)
            _json(out/'suite_state.json',state);report(out,state)
            print(f"DONE {job['name']} accuracy={value['summary']['final_test_accuracy']:.4f}%",flush=True)
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        state.update(status='completed',current_job=None,completed_unix=time.time())
    except BaseException as error:
        state.update(status='failed',failure_type=type(error).__name__,failure_message=str(error),failed_unix=time.time())
        _json(out/'suite_state.json',state);report(out,state)
        raise
    _json(out/'suite_state.json',state);report(out,state)
    return state


def start():
    torch.set_num_threads(1)
    gate=verify_queue_gate();records=jobs();reused=reuse_record()
    old=json.loads((PRIOR/'formal_inputs.json').read_text(encoding='utf-8'))
    for dataset in ('fmnist','mnist'):
        cfg=next(r['config'] for r in records if r['dataset']==dataset)
        verify_public_validation(cfg);verify_forecast_validation(cfg)
        train,test=load_datasets(cfg['base'])
        if len(train)!=60000 or dataset_digest(train)!=old['datasets'][dataset]['train'] or dataset_digest(test)!=old['datasets'][dataset]['test']:
            raise ValueError('Dataset differs from baseline experiment')
        del train,test
    result=queue(OUTPUT,records,reused,gate=gate)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--test',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    if args.test:test_gate()
    if args.run:start()
    if not any(vars(args).values()):parser.error('Use --prepare, --test or --run')
