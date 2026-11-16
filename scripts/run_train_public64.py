"""Fresh FMNIST seed42 three-method comparison: public64/client59936/test10000."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import spearmanr
import torch
import yaml
from src.brdp_source_runner import _json
from src.brdp_source_protocol import load_datasets,partition_and_groups
from src.dvpa_config import ROOT,load_config,public_preflight,resolve
from src.dvpa_controller import spatial_optimum
from src.dvpa_feedback import DPHistoryFeedback
from src.dvpa_filter import GroupRDPFilter,PrecisionAction
from src.dvpa_public64 import make_package,read_package,verify_validation,partition_record
from src.dvpa_public_proxy import curvature_traces
from src.dvpa_forecast import forecast_coefficients,parameters_from_config
from src.dvpa_runner import run
from src.dvpa_audit import audit
from src.layer_registry import LayerRegistry
from src.metrics import evaluate
from src.upstream_runner import source_model
from src.data import split_checksum
from scripts.check_dvpa_proxy import measure,candidates
from scripts.run_dvpa_suite import verify_formal,verify_gate,test_gate
from scripts.run_dvpa_comparison import tensor_digest,verify_identity

REPORTS=ROOT/'reports/train_public64_seed42_20261106'
PACKAGE=ROOT/'data/public_fmnist_train64.pt'
CONFIG=ROOT/'configs/dvpa_fmnist_trainpublic64_weight10_seed42.yaml'
OUTPUT=ROOT/'results/trainpublic64_fmnist_seed42_dvpa_idp_sayg_20261106'
HISTORY=ROOT/'results/dvpa_trainpublic1000_fmnist_weight10_seed42_20261106/DVPA_weight10_seed42'


def public_validation(cfg):
    """Use new public train input only, plus previously released DP models.

    Record the actual benefit result. It is not a certificate of FL task gain;
    correctness/isolation checks gate this fixed-data-protocol transfer trial.
    No old test-based benefit certificate is re-labelled for the new data.
    """
    torch.set_num_threads(1)
    _,sha,data=read_package(cfg['base'])
    check_cfg=copy.deepcopy(cfg)
    check_cfg['feedback'].update(utility_proxy='dimension_signal',curvature_forecast='hold_current')
    protocol=public_preflight(resolve(check_cfg))
    feedback=DPHistoryFeedback(protocol);ledger=GroupRDPFilter(protocol)
    model=source_model()('fmnist',125)
    model.load_state_dict(torch.load(HISTORY/'initial_model.pt',map_location='cpu',weights_only=True))
    reg=LayerRegistry(model)
    saved=[json.loads(v)['certified']['action'] for v in (HISTORY/'actions.jsonl').read_text(encoding='utf-8').splitlines()]
    checks=[];history=[];manifest={}
    for path in [HISTORY/'initial_model.pt',HISTORY/'actions.jsonl',*[(HISTORY/'released'/f'round_{t:03d}.pt') for t in range(1,22)]]:
        manifest[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    for t in range(1,22):
        feedback.prepare(t)
        if t in [4,13,21]:
            report=curvature_traces(model,data['images'][:64],probes=16)
            history.append(report['coefficients'])
            dim=np.array(list(protocol.dimensions.values()),dtype=float)
            importance=feedback.expected_costs(feedback.pred_mean,feedback.pred_cov,[t],origin=t)[0,0]/(dim*protocol.scales[t-1]**2/4)
            costs={'dimension_signal':dim*importance,'public_curvature_signal':np.array(report['traces'])*importance}
            old,new=[spatial_optimum(v,0.02) for v in costs.values()]
            allocations=candidates(old,new)
            predictions={k:np.sum(v[None,:]/allocations,axis=1) for k,v in costs.items()}
            model=model.to(device)
            for split in ['same_public64']:
                ids=data['indices']['calibration']
                losses=measure(model,data['images'][ids].to(device),data['labels'][ids].to(device),
                               allocations,protocol.scales[t-1]*0.5/np.sqrt(protocol.z0),64,20261106+t)
                for metric,values in zip(['cross_entropy','predictive_kl'],losses):
                    gain=values[:,1]-values[:,2];se=float(gain.std(ddof=1)/len(gain)**0.5)
                    checks.append(dict(round=t,split=split,metric=metric,examples=len(ids),draws=64,
                                       correlation={k:float(spearmanr(v,values.mean(0)).statistic) for k,v in predictions.items()},
                                       gain=float(gain.mean()),paired_se=se,margin=float(gain.mean()-3*se),
                                       allocations=allocations.tolist(),measured_means=values.mean(0).tolist()))
            model=model.cpu()
            print(json.dumps(dict(public_validation_round=t,checks=checks[-2:]),ensure_ascii=False),flush=True)
        item=saved[t-1]
        action=PrecisionAction(t,item['precision'],tuple(tuple(v) for v in item['layer_precision']),item['feedback_basis_hash'],item['history_hash'])
        cert=ledger.certify(action);ledger.reserve(cert)
        released=torch.load(HISTORY/'released'/f'round_{t:03d}.pt',map_location='cpu',weights_only=True)
        feedback.observe(t,released,cert);reg.apply(model,released)
    parameters=parameters_from_config(cfg['feedback'])
    prefix=np.array(history[:2],copy=True)
    before=forecast_coefficients(prefix,np.arange(4),**parameters)
    extended=np.vstack([prefix,np.array(history[-1])*[2,.5,1,1]])
    prefix_causal=bool(np.array_equal(before,forecast_coefficients(extended[:2],np.arange(4),**parameters)))
    predicted=forecast_coefficients(history,np.arange(6),**parameters)
    held=[v for v in checks if v['split']=='same_public64' and v['metric']=='predictive_kl']
    benefit=all(v['correlation']['public_curvature_signal']>=0.5 and v['margin']>0 for v in held)
    proof=dict(status='completed',validation_kind='public64_only_fixed_protocol_check',
               public_data_sha256=sha,parameters=parameters,probes=16,
               auxiliary_examples=64,independent_validation_examples=0,official_test_examples_queried=0,
               source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['src/dvpa_public64.py','src/dvpa_forecast.py','src/dvpa_public_proxy.py','scripts/run_train_public64.py']},
               signal_settings={k:cfg['feedback'][k] for k in ['lambda0','lambda1','signal_scale']},
               required_checks=dict(split_isolation=True,finite_positive_curvature=bool(np.isfinite(predicted).all() and (predicted>0).all()),
                   scale_preserved=bool(np.allclose(predicted.sum(1),582026)),
                   forecast_causal=prefix_causal,
                   no_test_data_for_validation=True),
               public_controlled_check_completed=True,public_proxy_benefit_passed=bool(benefit),
               forecast_empirical_benefit_validated=False,
               scope='Fixed-parameter transfer check on the same 64 public images; NO independent holdout and no claim of FL or forecasting benefit',
               history_manifest=manifest,checks=checks)
    _json(REPORTS/'public_validation.json',proof)
    verify_validation(cfg,proof)
    print(json.dumps(dict(public_validation='completed',benefit_passed=bool(benefit)),ensure_ascii=False),flush=True)
    return proof


def prepare():
    REPORTS.mkdir(parents=True,exist_ok=True)
    cfg=load_config(ROOT/'configs/dvpa_fmnist_information_weight10_seed42.yaml')
    if not PACKAGE.exists():make_package(PACKAGE,cfg['base']['data_root'])
    sha=hashlib.sha256(PACKAGE.read_bytes()).hexdigest()
    cfg['base'].pop('evaluation_public_file',None);cfg['base'].pop('evaluation_public_sha256',None)
    cfg['base'].update(public_train_file=str(PACKAGE.relative_to(ROOT)),public_train_sha256=sha,public_train_count=64)
    cfg['feedback'].update(public_proxy_file=str(PACKAGE.relative_to(ROOT)),public_proxy_sha256=sha,
                           public_proxy_validation=str((REPORTS/'public_validation.json').relative_to(ROOT)),
                           forecast_validation=str((REPORTS/'public_validation.json').relative_to(ROOT)))
    cfg=resolve(cfg);verify_formal(cfg)
    CONFIG.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    _json(REPORTS/'data_protocol.json',partition_record(cfg['base']))
    public_validation(cfg)


def report(state):
    import csv
    from src.st_runner import write_csv
    rows=[];fees=[]
    for record in state['completed']:
        summary=record['summary']
        rows.append(dict(method=record['label'],peak_accuracy=summary['best_test_accuracy'],
                         peak_round=record['best_round'],final_accuracy=summary['final_test_accuracy']))
        fees.extend(dict(method=record['label'],**g) for g in record['group_privacy'])
    write_csv(OUTPUT/'results_by_method.csv',rows);write_csv(OUTPUT/'group_privacy.csv',fees)
    _json(OUTPUT/'comparison.json',state)
    lines=['# FMNIST seed42：64张公开辅助、59,936张客户端训练、10,000张评估','',
           '状态：'+state['status']+'；当前任务：'+str(state.get('current_job'))+'。',
           '三种方法均从相同初始化重新训练25轮，共用客户端分区和隐私组；DVPA信息价值权重10，第三项协方差校准关闭。',
           '仅固定64张公开输入供曲率计算；受控扰动检查也使用这64张，不属于独立验证集。其余936张并入客户端，不再作为本次免费辅助查询。历史公开记录和旧结果保留。',
           '官方测试集仅用于评估，不进入控制器。最高准确率为回看25轮的测试峰值，非独立验证集选模成绩。','',
           '| 方法 | 最高准确率 | 轮次 | 最后准确率 |','|---|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['method']} | {row['peak_accuracy']:.2f}% | {row['peak_round']} | {row['final_accuracy']:.2f}% |")
    if len(rows)==3:
        lines+=['',f"DVPA相对IDP-FedAvg峰值：{rows[0]['peak_accuracy']-rows[1]['peak_accuracy']:+.2f}个百分点；相对spend-as-you-go：{rows[0]['peak_accuracy']-rows[2]['peak_accuracy']:+.2f}个百分点。"]
    lines+=['','| 方法 | 组 | 认证ε | 路径ε诊断 | α | 实际累计RDP |','|---|---:|---:|---:|---:|---:|']
    for g in fees:lines.append(f"| {g['method']} | {g['group']+1} | {g['epsilon_guarantee']:.6f} | {g['epsilon_path_diagnostic']:.6f} | {g['diagnostic_alpha']} | {g['rdp_path_at_diagnostic_alpha']:.8f} |")
    lines+=['','δ=10⁻⁵，共同上限10/20/30；保留各方法原生调度，实际费用分别核验，未强行配平。DVPA路径ε不能作为更小的全策略保证。费用计算全部25轮；各次运行联合发布仍需另行组合。',
            '单个seed不计算样本标准差；旧1,000张公开集结果不覆盖、不混入本表。','']
    (OUTPUT/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')


def train():
    import csv,gc
    from src.brdp_source_protocol import validate_config
    from src.brdp_source_runner import run as baseline_run
    from src.brdp_source_audit import audit as baseline_audit
    from scripts.run_train_public_baselines import metric_record
    from scripts.run_dvpa_comparison import normalized_privacy
    torch.set_num_threads(1)
    cfg=load_config(CONFIG);gate=verify_gate();verify_formal(cfg);verify_validation(cfg)
    if cfg['controller']['information_value_weight']!=10 or cfg['base']['seed']!=42:raise ValueError('Wrong weight/seed')
    OUTPUT.mkdir(parents=True,exist_ok=False)
    _json(OUTPUT/'test_gate.json',gate)
    (OUTPUT/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes())
    train_data,test=load_datasets(cfg['base'])
    if len(train_data)!=59936 or len(test)!=10000:raise ValueError('Wrong protocol')
    parts,ids=partition_and_groups(cfg['base'],train_data.tensors[1])
    torch.manual_seed(42)
    for _ in range(100):source_model()('fmnist',125)
    initial=source_model()('fmnist',125).state_dict()
    expected=dict(parts=parts,group_ids=ids.tolist(),split_checksum=split_checksum(parts),initial_tensor_sha256=tensor_digest(initial))
    _json(OUTPUT/'expected_identity.json',expected)
    del train_data,test,initial
    anchors=public_preflight(cfg).orders
    state=dict(status='running',seed=42,completed=[],current_job=None,started_unix=time.time(),process_id=os.getpid(),
               public_examples=64,client_training_examples=59936,evaluation_examples=10000)
    try:
        for label,method in [('DVPA','JOINT'),('IDP_FedAvg','IDP_FEDAVG'),('Time_Adaptive','SPEND_AS_YOU_GO')]:
            name=label+'_seed42';folder=OUTPUT/name
            state.update(current_job=name,current_job_started_unix=time.time())
            _json(OUTPUT/'suite_state.json',state);report(state);print('START '+name,flush=True)
            started=time.time()
            if method=='JOINT':summary=run(cfg,folder);checked=audit(folder)
            else:
                base=validate_config(dict(cfg['base'],method=method))
                summary=baseline_run(base,folder,REPORTS/'operational_progress.json');checked=baseline_audit(folder)
            _json(folder/'audit.json',checked)
            identity=verify_identity(folder,expected)
            _,test=load_datasets(cfg['base'])
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model=source_model()('fmnist',125).to(device)
            model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True))
            loss,accuracy=evaluate(model,test,device)
            if not np.isclose(accuracy,summary['final_test_accuracy'],rtol=0,atol=1e-9):raise ValueError('Full-test recheck mismatch')
            privacy=normalized_privacy(folder,label,anchors)
            if any(g['epsilon_guarantee']>cfg['base']['epsilons'][g['group']]+1e-8 for g in privacy):raise ValueError('Privacy exceeded')
            evaluation=dict(examples=10000,accuracy=accuracy,cross_entropy=loss)
            _json(folder/'evaluation_recheck.json',evaluation)
            record=metric_record(folder,label,summary,checked,False)
            record.update(identity=identity,group_privacy=privacy,evaluation_recheck=evaluation,elapsed_seconds=time.time()-started)
            state['completed'].append(record)
            _json(OUTPUT/'suite_state.json',state);report(state)
            print(f"DONE {name} peak={summary['best_test_accuracy']:.4f}%",flush=True)
            del model,test;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        state.update(status='completed',current_job=None,completed_unix=time.time())
    except BaseException as exc:
        state.update(status='failed',failure_type=type(exc).__name__,failure_message=str(exc),failed_unix=time.time())
        _json(OUTPUT/'suite_state.json',state);report(state);raise
    _json(OUTPUT/'suite_state.json',state);report(state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ['prepare','test','run']:parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    if args.test:test_gate()
    if args.run:train()
    if not any(vars(args).values()):parser.error('Use --prepare, --test or --run')


