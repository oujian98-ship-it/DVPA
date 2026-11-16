"""A single preregistered 10x-VOI FMNIST seed42 sensitivity experiment."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'

import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml

from src.brdp_source_runner import _json
from src.brdp_source_protocol import load_datasets
from src.dvpa_config import ROOT,load_config,public_preflight,resolve
from src.dvpa_controller import DecisionValueController
from src.dvpa_feedback import DPHistoryFeedback
from src.dvpa_filter import GroupRDPFilter,PrecisionAction
from src.dvpa_local_search import action_key
from src.dvpa_dataset_support import baseline_evaluation,verify_public_validation,verify_forecast_validation
from src.dvpa_runner import run
from src.dvpa_audit import audit
from src.metrics import evaluate
from src.upstream_runner import source_model
from scripts.run_dvpa_comparison import verify_identity
from scripts.run_dvpa_suite import verify_formal,verify_gate

PRIOR=ROOT/'results/dvpa_fmnist_mnist_seeds42_43_44_20261105'
BASELINE=PRIOR/'fmnist_DVPA_seed42'
CONFIG=ROOT/'configs/dvpa_fmnist_information_weight10_seed42.yaml'
REPORTS=ROOT/'reports/voi_weight_20261106'
OUTPUT=ROOT/'results/dvpa_information_weight10_fmnist_seed42_20261106'
WEIGHT=10.0


def prepare():
    cfg=load_config(BASELINE/'config.yaml')
    cfg['controller']['information_value_weight']=WEIGHT
    cfg=resolve(cfg);verify_formal(cfg)
    CONFIG.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    REPORTS.mkdir(parents=True,exist_ok=True)
    _json(REPORTS/'plan.json',dict(weight=WEIGHT,chosen_before_new_accuracy=True,
          factor_selection='fixed first sensitivity factor 10, not selected by accuracy or offline action count',
          only_mechanism_change='controller information_value_weight 1 -> 10',
          covariance_calibration_enabled=False,config=str(CONFIG),output=str(OUTPUT),
          baseline=str(BASELINE),evaluation_examples=9000,
          privacy_caps=[10,20,30],delta=1e-5,
          interpretation='weighted selection objective is a preference ablation, not unweighted expected distortion'))
    return cfg


def saved_action(d):
    return PrecisionAction(d['round_index'],d['precision'],tuple(tuple(x) for x in d['layer_precision']),
                           d['feedback_basis_hash'],d['history_hash'])


def offline():
    torch.set_num_threads(1)
    cfg=load_config(CONFIG);p=public_preflight(cfg)
    f,ledger,c=DPHistoryFeedback(p),GroupRDPFilter(p),DecisionValueController(p)
    records=[json.loads(v) for v in (BASELINE/'actions.jsonl').read_text(encoding='utf-8').splitlines()]
    result=dict(scope='fixed prior DP histories and balances, not retrained accuracy',weight=WEIGHT,checks=[])
    for t,record in enumerate(records,1):
        f.prepare(t)
        f.set_public_coefficients(t,json.loads((BASELINE/'public_proxy'/f'round_{t:03d}.json').read_text(encoding='utf-8')))
        previous=saved_action(record['certified']['action'])
        if t in [4,8,13,20,24]:
            p.config['controller']['information_value_weight']=1.0
            one,one_log=c.choose(t,f,ledger.remaining())
            if action_key(one)!=action_key(previous):
                raise ValueError('Default weight no longer reproduces baseline action')
            p.config['controller']['information_value_weight']=WEIGHT
            ten,ten_log=c.choose(t,f,ledger.remaining())
            row=dict(round=t,weight1_action=one.as_dict(),weight10_action=ten.as_dict(),
                     changed=action_key(one)!=action_key(ten),weight1=one_log,weight10=ten_log)
            result['checks'].append(row)
            _json(REPORTS/'offline.json',result)
            print(json.dumps(dict(round=t,changed=row['changed'],ranking=ten_log['ranking_status'],
                                  local_information_changed=ten_log['local_information_changed_action'])),flush=True)
        cert=ledger.certify(previous);ledger.reserve(cert)
        f.observe(t,torch.load(BASELINE/'released'/f'round_{t:03d}.pt',map_location='cpu',weights_only=True),cert)
    result.update(status='completed',changed_checks=sum(r['changed'] for r in result['checks']))
    _json(REPORTS/'offline.json',result)


def train():
    torch.set_num_threads(1)
    cfg=load_config(CONFIG);gate=verify_gate();verify_formal(cfg)
    reference=load_config(BASELINE/'config.yaml')
    reference['controller']['information_value_weight']=WEIGHT
    if cfg!=reference:raise ValueError('Experiment must change information weight only')
    verify_public_validation(cfg);verify_forecast_validation(cfg)
    if cfg['base']['dataset']!='fmnist' or cfg['base']['seed']!=42 or cfg['controller']['information_value_weight']!=WEIGHT:
        raise ValueError('Wrong fixed experiment identity')
    OUTPUT.mkdir(parents=True,exist_ok=False)
    (OUTPUT/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes())
    _json(OUTPUT/'test_gate.json',gate)
    state=dict(status='running',weight=WEIGHT,seed=42,dataset='fmnist',started_unix=time.time(),process_id=os.getpid())
    _json(OUTPUT/'experiment_state.json',state)
    try:
        folder=OUTPUT/'DVPA_weight10_seed42'
        result=run(cfg,folder)
        checked=audit(folder);_json(folder/'audit.json',checked)
        expected=json.loads((PRIOR/'formal_inputs.json').read_text(encoding='utf-8'))['expected_identity']['fmnist_42']
        identity=verify_identity(folder,expected)
        _,test=load_datasets(cfg['base']);test,_=baseline_evaluation(cfg['base'],test)
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model=source_model()('fmnist',125).to(device)
        model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True))
        loss,accuracy=evaluate(model,test,device)
        if len(test)!=9000 or not np.isclose(accuracy,result['final_test_accuracy'],rtol=0,atol=1e-9):
            raise ValueError('Independent final evaluation mismatch')
        evaluation=dict(examples=len(test),accuracy=accuracy,cross_entropy=loss)
        _json(folder/'evaluation_recheck.json',evaluation)
        previous=json.loads((BASELINE/'summary.json').read_text(encoding='utf-8'))
        comparison=dict(weight=WEIGHT,baseline_weight=1.0,current=result,previous=previous,
                        final_gain_pp=accuracy-previous['final_test_accuracy'],
                        audit=checked,identity=identity,evaluation_recheck=evaluation)
        _json(OUTPUT/'comparison.json',comparison)
        lines=['# FMNIST seed42：未来信息价值10倍权重敏感性实验','',
               '相同100客户端、25轮、每轮30本地epochs、固定9,000张评估。仅信息价值权重由1变为10。',
               '反馈信号系数、采样、裁剪、公开曲率、预测、细搜索、排名检查及隐私过滤器保持一致；第三项协方差校准关闭。',
               '公开seed控制初始化与分区，真实DP随机数独立；此为单种子新训练比较，不是配对噪声的因果收益证明。','',
               '| 方法 | 最终准确率 | 训练中最高准确率 |','|---|---:|---:|',
               f"| 原版，权重1 | {previous['final_test_accuracy']:.4f}% | {previous['best_test_accuracy']:.4f}% |",
               f"| 信息价值权重10 | {accuracy:.4f}% | {result['best_test_accuracy']:.4f}% |",'',
               f"最终差值：{comparison['final_gain_pp']:+.4f}个百分点。",'',
               '权重10的选择目标包含额外信息偏好，不等于原版预计总失真；原始信息值、加权信息值及未加权目标分别记录。','',
               '| 组 | α | 累计RDP费用 | RDP上限 | 路径ε诊断 | 全策略ε上限 |',
               '|---|---:|---:|---:|---:|---:|']
        for row in result['group_privacy']:
            if row['rdp_spent']>row['rdp_cap']:raise ValueError('RDP cap exceeded')
            lines.append(f"| {row['group']+1} | {row['alpha']} | {row['rdp_spent']:.8f} | {row['rdp_cap']:.8f} | {row['epsilon_path_diagnostic']:.6f} | {row['epsilon_guarantee']:.0f} |")
        lines += ['', 'δ=10⁻⁵。路径ε仅作诊断，不构成更小的全历史策略保证。','',
                  '粗排名诊断：`'+json.dumps(result['numerical_ranking'],ensure_ascii=False)+'`','',
                  '细搜索诊断：`'+json.dumps(result['local_search'],ensure_ascii=False)+'`','']
        (OUTPUT/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
        state.update(status='completed',completed_unix=time.time(),final_accuracy=accuracy,
                     gain_pp=comparison['final_gain_pp'],audit='passed')
        _json(OUTPUT/'experiment_state.json',state)
        print(json.dumps(state,ensure_ascii=False),flush=True)
    except BaseException as error:
        state.update(status='failed',failure_type=type(error).__name__,failure_message=str(error),failed_unix=time.time())
        _json(OUTPUT/'experiment_state.json',state)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    if args.offline:offline()
    if args.run:train()
    if not any(vars(args).values()):parser.error('Use --prepare, --offline or --run')
