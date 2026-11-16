"""Two native baselines on the completed DVPA 59000/1000/10000 protocol."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import copy
import csv
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml

from src.brdp_source_protocol import validate_config,load_datasets
from src.brdp_source_runner import run,_json
from src.brdp_source_audit import audit
from src.dvpa_config import ROOT,load_config,public_preflight
from src.metrics import evaluate
from src.st_runner import write_csv
from src.upstream_runner import source_model
from scripts.run_dvpa_suite import verify_gate,test_gate
from scripts.run_dvpa_comparison import verify_identity,normalized_privacy

DVPA_PARENT=ROOT/'results/dvpa_trainpublic1000_fmnist_weight10_seed42_20261106'
DVPA=DVPA_PARENT/'DVPA_weight10_seed42'
REPORTS=ROOT/'reports/train_public_baselines_20261106'
OUTPUT=ROOT/'results/trainpublic1000_fmnist_seed42_dvpa_idp_sayg_20261106'
TITLES={'DVPA':'DVPA，10倍信息价值权重','IDP_FedAvg':'IDP-FedAvg适配版','Time_Adaptive':'spend-as-you-go适配版'}


def jobs():
    reference=load_config(DVPA/'config.yaml')
    rows=[]
    for label,method in [('IDP_FedAvg','IDP_FEDAVG'),('Time_Adaptive','SPEND_AS_YOU_GO')]:
        cfg=copy.deepcopy(reference['base']);cfg.update(method=method,mechanism='gaussian',recycle_prob=0.0)
        cfg=validate_config(cfg)
        if any(cfg[k]!=v for k,v in reference['base'].items() if k!='method'):
            raise ValueError('Baseline must preserve all shared training/data fields')
        rows.append(dict(name=label+'_seed42',label=label,config=cfg))
    return rows


def prepare():
    REPORTS.mkdir(parents=True,exist_ok=True);(REPORTS/'configs').mkdir(exist_ok=True)
    for j in jobs():
        (REPORTS/'configs'/(j['name']+'.yaml')).write_text(yaml.safe_dump(j['config'],sort_keys=False),encoding='utf-8')
    _json(REPORTS/'plan.json',dict(dataset='fmnist',seed=42,new_runs=2,reuse=str(DVPA),
          private_training_examples=59000,public_training_examples=1000,evaluation_examples=10000,
          policies='IDP constant q=.9; SAYG native recurrence and q=.5/.6/.7 -> .9 at round13',
          privacy_caps=[10,20,30],delta=1e-5,baseline_costs_are_not_forced_equal_to_dvpa=True))


def metric_record(folder,label,summary,checked,reused):
    with (folder/'rounds.csv').open(encoding='utf-8') as handle:rows=list(csv.DictReader(handle))
    if len(rows)!=25 or checked['status']!='passed' or summary['evaluation_examples']!=10000:
        raise ValueError('Only completed full-test audited runs may enter comparison')
    peak=max(rows,key=lambda r:float(r['test_accuracy']))
    if abs(float(peak['test_accuracy'])-summary['best_test_accuracy'])>1e-9:
        raise ValueError('Peak does not match saved rounds')
    return dict(label=label,folder=str(folder),summary=summary,audit=checked,reused=reused,
                best_round=int(peak['round']))


def report(out,state):
    metrics=[];fees=[]
    for r in state['completed']:
        s=r['summary']
        metrics.append(dict(method=r['label'],best_accuracy=s['best_test_accuracy'],best_round=r['best_round'],
                            final_accuracy=s['final_test_accuracy'],audit=r['audit']['status']))
        for g in r['group_privacy']:fees.append(dict(method=r['label'],**g))
    write_csv(out/'results_by_method.csv',metrics);write_csv(out/'group_privacy.csv',fees)
    _json(out/'comparison.json',dict(status=state['status'],seed=42,evaluation_examples=10000,
          private_training_examples=59000,public_training_examples=1000,results=state['completed']))
    lines=['# FMNIST seed42：59,000张客户端训练、1,000张公开辅助、10,000张测试','',
           f"状态：{state['status']}；本次baseline完成{sum(not r['reused'] for r in state['completed'])}/2；当前：{state.get('current_job')}。",'',
           '两个baseline全新训练；DVPA复用已完成的新协议结果。三者核验同一客户端分区、隐私组、CNN初始化与完整测试集。',
           '100客户端、25轮、每轮30本地epochs、batch125、lr0.001、持久SGD(momentum0.9)、原Cosine调度。',
           'IDP与spend-as-you-go保留各自原生采样、裁剪、噪声及会计策略；共同上限10/20/30，实际费用不强行配平。',
           'seed42固定分区/初始化，真实DP噪声独立。本表仅一个seed，不计算多种子标准差。','',
           '| 方法 | 25轮最高准确率 | 峰值轮次 | 第25轮准确率 |','|---|---:|---:|---:|']
    for r in metrics:lines.append(f"| {TITLES[r['method']]} | {r['best_accuracy']:.4f}% | {r['best_round']} | {r['final_accuracy']:.4f}% |")
    if len(metrics)==3:
        dvpa=next(r for r in metrics if r['method']=='DVPA')
        lines+=['','| 对照 | DVPA峰值差值，百分点 | DVPA末轮差值，百分点 |','|---|---:|---:|']
        for r in metrics:
            if r['method']!='DVPA':lines.append(f"| {TITLES[r['method']]} | {dvpa['best_accuracy']-r['best_accuracy']:+.4f} | {dvpa['final_accuracy']-r['final_accuracy']:+.4f} |")
    lines+=['','## 实际隐私会计','',
            '| 方法 | 组 | 全策略认证ε | 路径ε诊断 | 共同诊断α | 累计RDP |','|---|---:|---:|---:|---:|---:|']
    for g in fees:
        lines.append(f"| {TITLES[g['method']]} | {g['group']+1} | {g['epsilon_guarantee']:.6f} | {g['epsilon_path_diagnostic']:.6f} | {g['diagnostic_alpha']} | {g['rdp_path_at_diagnostic_alpha']:.8f} |")
    lines+=['','δ=10⁻⁵。DVPA路径ε不是更小的全策略保证。各次运行分别认证，三个实验联合发布需另行组合。',
            '峰值是同一10,000张评估集上回看25轮取最大值，并非独立验证集选模后的测试成绩。费用仍按完整25轮计算。','']
    (out/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')


def train():
    torch.set_num_threads(1)
    gate=verify_gate();records=jobs()
    for j in records:
        frozen=validate_config(yaml.safe_load((REPORTS/'configs'/(j['name']+'.yaml')).read_text(encoding='utf-8')))
        if frozen!=j['config']:raise ValueError('Frozen baseline config changed')
    source=json.loads((DVPA/'implementation.json').read_text(encoding='utf-8'))
    for name,sha in source['source_sha256']['source_snapshot'].items():
        if hashlib.sha256((DVPA/'source_snapshot'/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen DVPA source snapshot changed')
        # Only these baseline data-adapter files were changed after DVPA ran.
        if name not in {'brdp_source_protocol.py','brdp_source_runner.py','brdp_source_audit.py'}:
            if hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()!=sha:
                raise ValueError('DVPA mechanism changed; cannot reuse previous result')
    cfg=load_config(DVPA/'config.yaml');anchors=public_preflight(cfg).orders
    summary=json.loads((DVPA/'summary.json').read_text(encoding='utf-8'))
    checked=json.loads((DVPA/'audit.json').read_text(encoding='utf-8'))
    dvpa=metric_record(DVPA,'DVPA',summary,checked,True)
    dvpa['group_privacy']=normalized_privacy(DVPA,'DVPA',anchors)
    expected=json.loads((DVPA_PARENT/'expected_identity.json').read_text(encoding='utf-8'))
    train_data,test=load_datasets(records[0]['config'])
    if len(train_data)!=59000 or len(test)!=10000:raise ValueError('Wrong shared data protocol')
    del train_data,test
    OUTPUT.mkdir(parents=True,exist_ok=False)
    (OUTPUT/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes());_json(OUTPUT/'test_gate.json',gate)
    state=dict(status='running',new_runs=2,completed=[dvpa],current_job=None,started_unix=time.time(),process_id=os.getpid())
    _json(OUTPUT/'suite_state.json',state);report(OUTPUT,state)
    try:
        for job in records:
            state.update(current_job=job['name'],current_job_started_unix=time.time())
            _json(OUTPUT/'suite_state.json',state);report(OUTPUT,state)
            print('START '+job['name'],flush=True)
            folder=OUTPUT/job['name'];started=time.time()
            # The native runner writes round progress and parameter-cost records.
            summary=run(job['config'],folder,REPORTS/'operational_progress.json')
            checked=audit(folder);_json(folder/'audit.json',checked)
            identity=verify_identity(folder,expected)
            _,test=load_datasets(job['config'])
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model=source_model()('fmnist',125).to(device)
            model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True))
            loss,accuracy=evaluate(model,test,device)
            if len(test)!=10000 or not np.isclose(accuracy,summary['final_test_accuracy'],rtol=0,atol=1e-9):
                raise ValueError('Independent full-test evaluation mismatch')
            evaluation=dict(examples=10000,accuracy=accuracy,cross_entropy=loss)
            _json(folder/'evaluation_recheck.json',evaluation)
            for g,row in enumerate(summary['group_privacy']):
                if row['epsilon_guarantee']>job['config']['epsilons'][g]+1e-8:raise ValueError('Privacy target exceeded')
            record=metric_record(folder,job['label'],summary,checked,False)
            record.update(identity=identity,evaluation_recheck=evaluation,
                          group_privacy=normalized_privacy(folder,job['label'],anchors),elapsed_seconds=time.time()-started)
            state['completed'].append(record)
            _json(OUTPUT/'suite_state.json',state);report(OUTPUT,state)
            print(f"DONE {job['name']} final={accuracy:.4f}% best={summary['best_test_accuracy']:.4f}%",flush=True)
            del model,test;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        state.update(status='completed',current_job=None,completed_unix=time.time())
    except BaseException as e:
        state.update(status='failed',failure_type=type(e).__name__,failure_message=str(e),failed_unix=time.time())
        _json(OUTPUT/'suite_state.json',state);report(OUTPUT,state);raise
    _json(OUTPUT/'suite_state.json',state);report(OUTPUT,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ['prepare','test','run']:parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    if args.test:test_gate()
    if args.run:train()
    if not any(vars(args).values()):parser.error('Use --prepare, --test or --run')
