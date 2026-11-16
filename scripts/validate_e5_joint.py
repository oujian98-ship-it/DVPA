"""E5 full 2x2: S2/D3 crossed with equal clipping/4A; fresh complete seed blocks."""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import secrets
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from scripts.validate_bottom_up import preflight
from src.upstream_runner import SOURCE
from src.p1_runner import run
from src.p4_evaluation import evaluate_manifest,write_evaluation
from src.noise_savings import summarize_noise_savings
from src.e5_analysis import METHODS,factorial_summary


def configurations():
    base=yaml.safe_load(Path('configs/p3_ema.yaml').read_text())
    prior=yaml.safe_load(Path('configs/p2_4a_bu.yaml').read_text())['allocation']
    excluded={'seed','mask_seed','mode','freeze_rounds','feedback','allocation'}
    fixed={k:v for k,v in base.items() if k not in excluded}
    feedback={**base['feedback'],'patience':5}
    static=dict(mode='BU',freeze_rounds=[15,17],static_order=['conv1','conv2','fc1','fc'])
    dynamic=dict(mode='ADAPTIVE',freeze_rounds=[],feedback=feedback,diagnostic_freezing=dict(
        kind='confidence',min_freeze_round=15,min_active_fraction=.5,freeze_order='bottom_up',protected_groups=['fc1'],
        alpha_total=.05,drift_window=3,max_ratio_range=.25,drift_margin_ratio=.25))
    methods={}
    for name,policy,allocation in [('STATIC_EQUAL',static,{'layer_clip':'equal_active'}),('STATIC_4A',static,prior),
                                   ('DYNAMIC_EQUAL',dynamic,{'layer_clip':'equal_active'}),('DYNAMIC_4A',dynamic,prior)]:
        methods[name]=dict(copy.deepcopy(policy),allocation=copy.deepcopy(allocation))
    return fixed,methods


def write_report(result,root):
    lines=['# E5：P2 + 保守 P3 的 2×2 联合消融','',f"FMNIST，25 轮，登记种子 {result['training_seeds']}。全部四组均为新训练，每组独立私有 DP 随机源；相同种子下完整 RDP、数据划分、模型和训练预算审计匹配。",
           '静态为 S2 conv1@15→conv2@17；动态为预设 D3 置信冻结。4A 使用原先验，不按结果调参。',
           '', '| 方法 | 历程最高均值 ± 样本标准差 | 最终均值 ± 样本标准差 |','|---|---:|---:|']
    spread=lambda v:'NA' if v is None else f'{v:.2f}'
    for r in result['main_table']:lines.append(f"| {r['method']} | {r['best']['mean']:.2f} ± {spread(r['best']['std'])}% | {r['final']['mean']:.2f} ± {spread(r['final']['std'])}% |")
    for metric,rows in result['contrasts'].items():
        lines.extend(['',f'## 因子对比：{metric}','','差值为百分点；interaction=(动态4A−动态等份)−(静态4A−静态等份)。',
                      '', '| 对比 | 平均差 ± 标准差 | 各 seed 差值 | 精确双侧 p | Holm p |','|---|---:|---|---:|---:|'])
        for r in rows:lines.append(f"| {r['contrast']} | {r['mean']:+.2f} ± {spread(r['std'])} | {r['gaps']} | {r['exact_two_sided_sign_flip_p']:.3f} | {r['holm_adjusted_p_within_metric']:.3f} |")
    lines.extend(['','## 逐次结果','','| seed | 方法 | 最高 | 最高轮 | 最终 | 实际冻结轮次 | 累计期望噪声节省 |','|---:|---|---:|---:|---:|---|---:|'])
    for r in result['per_run']:lines.append(f"| {r['seed']} | {r['method']} | {r['best_test_acc']:.2f}% | {r['best_round']} | {r['final_test_acc']:.2f}% | {json.dumps(r['freeze_first_round'])} | {100*r['noise_savings']['cumulative_expected_noise_saved_fraction']:.4f}% |")
    lines.extend(['','## 静态与动态掩码对比','',json.dumps(result['same_freeze_schedule_by_seed'],ensure_ascii=False,indent=2),
                  '', '若某个配置下全程冻结日程相同，则该次在线反馈未在掩码上产生独立变化；不同 DP 随机源下的准确率差不能单独归因于反馈。',
                  '', '## 解释范围','','只运行本次登记的全部四组；不混入先前较高结果，不选择种子或重跑。峰值来自 25 次公开测试集评估，不等于独立验证集选模。',
                  '三种子双侧精确符号翻转检验最小 p 为 0.25，因此本次不能据此宣称统计显著。两个指标各自五项对比做 Holm 校正；交互项只描述当前范围内的非加性效应。',
                  '动态 4A 下冻结时机允许按原 D3 规则变化，不强行对齐静态掩码；因此联合收益包含分配改变后对决策轨迹的影响。不得把它直接归因于单独的 UCB 或 4A。',
                  '噪声节省为 Σ d_t v_t 相对同方差全活跃参照，非实测噪声范数、NF 精度、最终模型误差或 epsilon 节省。',
                  'D3 的 UCB 针对历史已剪裁、已抽样聚合能量；漂移余量和稳定窗口是启发式，不构成未来永久冻结效用保证。',
                  'P0 gate 未改变。此前未完成的多种子纯冻结对照并未被本次四组消融自动补全。不同实验结果联合发布的总隐私成本没有在这里认证。',
                  '每次运行的私有随机状态独立保存在 .private_diagnostics/ 下，并由协议哈希校验；不要随公开结果发布。完整环境快照见 environment.json。'])
    (root/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results-dir',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);a=p.parse_args()
    if not a.seeds or len(set(a.seeds))!=len(a.seeds) or any(s not in [42,43,44] for s in a.seeds):raise ValueError('Only registered seeds42/43/44 allowed')
    root=Path(a.results_dir).resolve();root.mkdir(parents=True,exist_ok=False);completed=[];total=4*len(a.seeds)
    def status(state,**extra):
        (root/'status.json').write_text(json.dumps(dict(state=state,completed=completed,total=total,**extra),indent=2))
    fixed,methods=configurations();status('privacy_preflight');preflight(fixed,root/'privacy_preflight.npz')
    private=Path('.private_diagnostics')/('e5_'+secrets.token_hex(12));private.mkdir(parents=True,exist_ok=False)
    registered={};rng_hashes={}
    for seed in a.seeds:
        registered[str(seed)]={}
        for name,method in methods.items():
            state=(private/f'{name}_seed{seed}.json').resolve()
            state.write_text(json.dumps(dict(sampling=secrets.randbits(128),noise=secrets.randbits(63))))
            rng_hashes[str(state)]=hashlib.sha256(state.read_bytes()).hexdigest()
            registered[str(seed)][name]={**copy.deepcopy(method),'diagnostic_rng_path':str(state)}
    sources=[f for f in SOURCE.rglob('*') if f.is_file()]+list(Path('src').glob('*.py'))
    sources.extend([Path(__file__),Path('scripts/validate_bottom_up.py'),Path('configs/p3_ema.yaml'),Path('configs/p2_4a_bu.yaml')])
    protocol=dict(kind='controlled_joint_allocation_freezing_v1',data_scope='public_benchmark',fixed_config=fixed,
        methods_by_seed=registered,training_seeds=a.seeds,private_rng_sha256=rng_hashes,
        authorization='User requested complete E5 joint experiment: four groups x seeds42/43/44; P0 gate unchanged',
        privacy_preflight='privacy_preflight.npz',privacy_preflight_sha256=hashlib.sha256((root/'privacy_preflight.npz').read_bytes()).hexdigest(),
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        primary_metric='best_test_acc',secondary_metric='final_test_acc',
        interaction='(DYNAMIC_4A-DYNAMIC_EQUAL)-(STATIC_4A-STATIC_EQUAL)',
        selection_policy='Fresh complete four-method seed blocks; no prior run substitution, tuning or silent reruns; independent secret entropy per run',
        scope='Exploratory FMNIST factorial study, not the original plan CIFAR expansion or a future-utility theorem; per-run accounting only')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    environment=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
                     cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','opacus','PyYAML','pytest']})
    (root/'environment.json').write_text(json.dumps(environment,indent=2))
    manifest=dict(version=1,data_scope='public_benchmark',allow_engineering=False,include_local_cost=False,
        cases=[dict(id='E5_joint_2x2',baseline='STATIC_EQUAL',budget_matching='same_rdp',accuracy_target=70.,
            methods=[dict(id=name,runs=[dict(seed=s,replicate='default',path=f'{name}_seed{s}') for s in a.seeds]) for name in METHODS])])
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
        for seed in a.seeds:
            for name in METHODS:
                directory=root/f'{name}_seed{seed}';status('running',current=directory.name)
                cfg=copy.deepcopy(fixed);cfg.update(seed=seed,mask_seed=seed,**registered[str(seed)][name],validation_protocol=str(root/'protocol.json'))
                run(cfg,directory);completed.append(directory.name)
        status('training_completed_auditing')
        evaluation=evaluate_manifest(manifest,root);write_evaluation(evaluation,root/'manifest.json',root/'analysis')
        records=[];noise_trace=[]
        for seed in a.seeds:
            for name in METHODS:
                folder=root/f'{name}_seed{seed}';s=json.loads((folder/'summary.json').read_text())
                with (folder/'rounds.csv').open() as f:rows=list(csv.DictReader(f))
                savings,trace=summarize_noise_savings(rows,s['parameters'])
                noise_trace.extend(dict(seed=seed,method=name,**r) for r in trace)
                records.append(dict(seed=seed,method=name,best_test_acc=s['best_test_acc'],final_test_acc=s['final_test_acc'],
                    best_round=next(int(r['round']) for r in rows if float(r['test_acc'])==s['best_test_acc']),
                    freeze_first_round=s['freeze_first_round'],noise_savings=savings))
        result=factorial_summary(records,a.seeds)
        indexed={(r['seed'],r['method']):r for r in records}
        result['same_freeze_schedule_by_seed']={str(s):{
            'equal':indexed[(s,'STATIC_EQUAL')]['freeze_first_round']==indexed[(s,'DYNAMIC_EQUAL')]['freeze_first_round'],
            '4A':indexed[(s,'STATIC_4A')]['freeze_first_round']==indexed[(s,'DYNAMIC_4A')]['freeze_first_round']} for s in a.seeds}
        (root/'factorial_analysis.json').write_text(json.dumps(result,indent=2));write_report(result,root)
        with (root/'noise_savings_rounds.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(noise_trace[0]));writer.writeheader();writer.writerows(noise_trace)
        status('completed');print(json.dumps(dict(main_table=result['main_table'],contrasts=result['contrasts']),indent=2),flush=True)
    except BaseException as exc:status('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc));raise


if __name__=='__main__':main()
