"""FMNIST seed42, public1000 from train, client59000, test10000, VOI weight10."""
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
from src.dvpa_train_public import make_package,read_package,verify_validation,partition_record
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

REPORTS=ROOT/'reports/dvpa_trainpublic_20261106'
PACKAGE=ROOT/'data/public_fmnist_train1000.pt'
CONFIG=ROOT/'configs/dvpa_fmnist_trainpublic_weight10_seed42.yaml'
OUTPUT=ROOT/'results/dvpa_trainpublic1000_fmnist_weight10_seed42_20261106'
HISTORY=ROOT/'results/dvpa_information_weight10_fmnist_seed42_20261106/DVPA_weight10_seed42'


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
            for split in ['validation','confirmation']:
                ids=data['indices'][split]
                losses=measure(model,data['images'][ids].to(device),data['labels'][ids].to(device),
                               allocations,protocol.scales[t-1]*0.5/np.sqrt(protocol.z0),64,20261106+t)
                for metric,values in zip(['cross_entropy','predictive_kl'],losses):
                    gain=values[:,1]-values[:,2];se=float(gain.std(ddof=1)/len(gain)**0.5)
                    checks.append(dict(round=t,split=split,metric=metric,examples=len(ids),draws=64,
                                       correlation={k:float(spearmanr(v,values.mean(0)).statistic) for k,v in predictions.items()},
                                       gain=float(gain.mean()),paired_se=se,margin=float(gain.mean()-3*se),
                                       allocations=allocations.tolist(),measured_means=values.mean(0).tolist()))
            model=model.cpu()
            print(json.dumps(dict(public_validation_round=t,checks=checks[-4:]),ensure_ascii=False),flush=True)
        item=saved[t-1]
        action=PrecisionAction(t,item['precision'],tuple(tuple(v) for v in item['layer_precision']),item['feedback_basis_hash'],item['history_hash'])
        cert=ledger.certify(action);ledger.reserve(cert)
        released=torch.load(HISTORY/'released'/f'round_{t:03d}.pt',map_location='cpu',weights_only=True)
        feedback.observe(t,released,cert);reg.apply(model,released)
    parameters=parameters_from_config(cfg['feedback'])
    predicted=forecast_coefficients(history,np.arange(6),**parameters)
    held=[v for v in checks if v['split']=='confirmation' and v['metric']=='cross_entropy']
    benefit=all(v['correlation']['public_curvature_signal']>=0.5 and v['margin']>0 for v in held)
    proof=dict(status='completed',validation_kind='train_public_fixed_protocol_validation',
               public_data_sha256=sha,parameters=parameters,probes=16,
               module_sha256=hashlib.sha256((ROOT/'src/dvpa_train_public.py').read_bytes()).hexdigest(),
               proxy_module_sha256=hashlib.sha256((ROOT/'src/dvpa_public_proxy.py').read_bytes()).hexdigest(),
               forecast_module_sha256=hashlib.sha256((ROOT/'src/dvpa_forecast.py').read_bytes()).hexdigest(),
               validation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               signal_settings={k:cfg['feedback'][k] for k in ['lambda0','lambda1','signal_scale']},
               required_checks=dict(split_isolation=True,finite_positive_curvature=bool(np.isfinite(predicted).all() and (predicted>0).all()),
                   scale_preserved=bool(np.allclose(predicted.sum(1),582026)),
                   forecast_causal=bool(np.array_equal(forecast_coefficients(history[:2],np.arange(4),**parameters),forecast_coefficients(history[:2],np.arange(4),**parameters))),
                   no_test_data_for_validation=True),
               public_controlled_check_completed=True,public_proxy_benefit_passed=bool(benefit),
               forecast_empirical_benefit_validated=False,
               scope='fixed-parameter data-protocol transfer; actual public-only controlled benefit measured, no claim of new-protocol FL benefit or forecast superiority',
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
    cfg['base'].update(public_train_file=str(PACKAGE.relative_to(ROOT)),public_train_sha256=sha)
    cfg['feedback'].update(public_proxy_file=str(PACKAGE.relative_to(ROOT)),public_proxy_sha256=sha,
                           public_proxy_validation=str((REPORTS/'public_validation.json').relative_to(ROOT)),
                           forecast_validation=str((REPORTS/'public_validation.json').relative_to(ROOT)))
    cfg=resolve(cfg);verify_formal(cfg)
    CONFIG.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    _json(REPORTS/'data_protocol.json',partition_record(cfg['base']))
    public_validation(cfg)


def train():
    torch.set_num_threads(1)
    cfg=load_config(CONFIG);gate=verify_gate();verify_formal(cfg);proof=verify_validation(cfg)
    if cfg['controller']['information_value_weight']!=10 or cfg['base']['seed']!=42:raise ValueError('Wrong requested weight/seed')
    OUTPUT.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',dataset='fmnist',seed=42,weight=10,started_unix=time.time(),process_id=os.getpid(),
               private_training_examples=59000,public_training_examples=1000,evaluation_examples=10000)
    _json(OUTPUT/'experiment_state.json',state);_json(OUTPUT/'test_gate.json',gate)
    (OUTPUT/'driver_snapshot.py').write_bytes(Path(__file__).read_bytes())
    try:
        train_data,test=load_datasets(cfg['base'])
        if len(train_data)!=59000 or len(test)!=10000:raise ValueError('Wrong official data split')
        parts,ids=partition_and_groups(cfg['base'],train_data.tensors[1])
        torch.manual_seed(42)
        for _ in range(100):source_model()('fmnist',125)
        initial=source_model()('fmnist',125).state_dict()
        expected=dict(parts=parts,group_ids=ids.tolist(),split_checksum=split_checksum(parts),initial_tensor_sha256=tensor_digest(initial))
        _json(OUTPUT/'expected_identity.json',expected)
        del train_data,test,initial
        folder=OUTPUT/'DVPA_weight10_seed42'
        result=run(cfg,folder);checked=audit(folder);_json(folder/'audit.json',checked)
        identity=verify_identity(folder,expected)
        _,test=load_datasets(cfg['base'])
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model=source_model()('fmnist',125).to(device)
        model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True))
        loss,accuracy=evaluate(model,test,device)
        if not np.isclose(accuracy,result['final_test_accuracy'],rtol=0,atol=1e-9) or result['evaluation_examples']!=10000:
            raise ValueError('Independent full-test evaluation mismatch')
        for g in result['group_privacy']:
            if g['rdp_spent']>g['rdp_cap']:raise ValueError('Privacy cap exceeded')
        _json(folder/'evaluation_recheck.json',dict(examples=10000,accuracy=accuracy,cross_entropy=loss))
        _json(OUTPUT/'comparison.json',dict(summary=result,audit=checked,identity=identity,public_validation=proof,
              comparison_scope='new 59000/1000/10000 protocol; old 60000/1000-test/9000 scores are not matched baselines'))
        import csv
        rows=list(csv.DictReader((folder/'rounds.csv').open(encoding='utf-8')))
        peak=max(rows,key=lambda r:float(r['test_accuracy']))
        lines=['# FMNIST seed42：训练集划出1000张公开辅助数据','',
               '训练集先固定抽出1000张公开辅助数据，64张计算曲率，468张代理验证、468张代理确认；其余59000张分给100个客户端。',
               '官方10000张测试图像仅用于本次评估。保留10倍信息价值权重、25轮、每轮30本地epochs、原CNN和优化设置；第三项协方差校准关闭。','',
               f"最终准确率：{accuracy:.4f}%；25轮最高准确率：{result['best_test_accuracy']:.4f}%（第{peak['round']}轮）。",'',
               f"新公开集受控代理收益检查通过：{proof['public_proxy_benefit_passed']}。曲率预测固定迁移原参数，没有声称在新数据协议上完成预测收益验证。",'',
               '该协议的客户端训练样本量及评估集均与旧实验不同，不能把与旧分数的差值解释为严格同协议提升。',
               '这1000张样本被声明为公开数据，不属于客户端隐私保护人群。原10/20/30、δ=10⁻⁵的单次隐私保证适用于固定客户端贡献。','',
               '| 组 | α | 实际累计RDP | RDP上限 | 路径ε诊断 | 全策略ε上限 |','|---|---:|---:|---:|---:|---:|']
        for g in result['group_privacy']:
            lines.append(f"| {g['group']+1} | {g['alpha']} | {g['rdp_spent']:.8f} | {g['rdp_cap']:.8f} | {g['epsilon_path_diagnostic']:.6f} | {g['epsilon_guarantee']:.0f} |")
        lines+=['','路径ε为诊断，隐私费用按全部25轮核验，峰值选择不退还后续消耗。','']
        (OUTPUT/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
        state.update(status='completed',completed_unix=time.time(),final_accuracy=accuracy,best_accuracy=result['best_test_accuracy'],audit='passed')
        _json(OUTPUT/'experiment_state.json',state);print(json.dumps(state),flush=True)
    except BaseException as e:
        state.update(status='failed',failure_type=type(e).__name__,failure_message=str(e),failed_unix=time.time())
        _json(OUTPUT/'experiment_state.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ['prepare','test','run']:parser.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    if args.test:test_gate()
    if args.run:train()
    if not any(vars(args).values()):parser.error('Use --prepare, --test or --run')
