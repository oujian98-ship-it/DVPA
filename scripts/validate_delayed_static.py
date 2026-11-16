"""S2/S3 delayed static masks, seed42 only; no adaptive scores in training."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from scripts.validate_bottom_up import preflight
from scripts.analyze_noise_savings import analyze
from src.upstream_runner import SOURCE
from src.p1_runner import run
from src.p4_evaluation import evaluate_manifest,write_evaluation


def static_methods():
    return dict(S2_STATIC_BU_15_17=dict(mode='BU',freeze_rounds=[15,17],static_order=['conv1','conv2','fc1','fc']),
                S3_STATIC_REVERSE_15_17=dict(mode='BU',freeze_rounds=[15,17],static_order=['conv2','conv1','fc1','fc']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results-dir',required=True)
    p.add_argument('--diagnostic-root',default='results/d123_freezing_seed42')
    a=p.parse_args();root=Path(a.results_dir).resolve();old=Path(a.diagnostic_root).resolve()
    root.mkdir(parents=True,exist_ok=False);completed=[]
    def status(state,**extra):
        (root/'status.json').write_text(json.dumps(dict(state=state,completed=completed,total=2,**extra),indent=2))
    base=yaml.safe_load(Path('configs/p2_equal_bu.yaml').read_text())
    fixed={k:v for k,v in base.items() if k not in {'mode','freeze_rounds','seed','mask_seed'}}
    methods=static_methods();status('privacy_preflight');preflight(base,root/'privacy_preflight.npz')
    sources=[p for p in SOURCE.rglob('*') if p.is_file()]+list(Path('src').glob('*.py'))
    sources.extend([Path(__file__),Path('scripts/validate_bottom_up.py'),Path('scripts/analyze_noise_savings.py')])
    protocol=dict(kind='controlled_freezing_diagnostics_v1',data_scope='public_benchmark',fixed_config=fixed,
                  methods=methods,training_seeds=[42],authorization='User requested only S2/S3 seed42 now; multi-seed/E5 deferred; P0 unchanged',
                  privacy_preflight='privacy_preflight.npz',privacy_preflight_sha256=hashlib.sha256((root/'privacy_preflight.npz').read_bytes()).hexdigest(),
                  source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  primary_metric='best_test_acc',secondary_metric='final_test_acc',
                  selection_policy='Two prespecified full static runs, no feedback, no tuning or silent reruns',
                  limitations='Matched seed42/data/config/RDP/masks, but independent private DP randomness. No causal/significance claim from one pair. Schedules chosen to match prior observed D2/D3, so this is a post-hoc diagnostic, not a new unbiased benchmark.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    pairs=[('S2_STATIC_BU_15_17','D3_CONFIDENCE_BU'),('S3_STATIC_REVERSE_15_17','D2_DELAYED_EMA')]
    manifest=dict(version=1,data_scope='public_benchmark',allow_engineering=False,include_local_cost=False,
                  cases=[dict(id=f'{s}_vs_{d}',baseline=s,budget_matching='same_rdp',accuracy_target=70.,methods=[
                      dict(id=s,runs=[dict(seed=42,replicate='default',path=s+'_seed42')]),
                      dict(id=d,runs=[dict(seed=42,replicate='default',path=str(old/(d+'_seed42')))])]) for s,d in pairs])
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
        for name,method in methods.items():
            status('running',current=name+'_seed42');cfg=copy.deepcopy(fixed)
            cfg.update(seed=42,mask_seed=42,**method,validation_protocol=str(root/'protocol.json'))
            run(cfg,root/(name+'_seed42'));completed.append(name+'_seed42')
        status('training_completed_auditing')
        result=evaluate_manifest(manifest,root);write_evaluation(result,root/'manifest.json',root/'analysis')
        comparison=[]
        for static,dynamic in pairs:
            folders=[root/(static+'_seed42'),old/(dynamic+'_seed42')]
            def load(folder,name):
                with (folder/name).open() as f:return list(csv.DictReader(f))
            left=load(folders[0],'layers.csv');right=load(folders[1],'layers.csv')
            fields=['round','layer','mask','active_params','weight','noise_tax']
            if [{k:r[k] for k in fields} for r in left]!=[{k:r[k] for k in fields} for r in right]:raise ValueError('Static mask/noise allocation differs from matched dynamic run')
            summaries=[json.loads((f/'summary.json').read_text()) for f in folders]
            comparison.append(dict(static=static,dynamic=dynamic,static_best=summaries[0]['best_test_acc'],dynamic_best=summaries[1]['best_test_acc'],
                                   static_final=summaries[0]['final_test_acc'],dynamic_final=summaries[1]['final_test_acc'],
                                   dynamic_minus_static_best_pp=summaries[1]['best_test_acc']-summaries[0]['best_test_acc'],
                                   dynamic_minus_static_final_pp=summaries[1]['final_test_acc']-summaries[0]['final_test_acc'],
                                   same_masks_weights_noise_tax_all_rounds=True))
        (root/'comparison.json').write_text(json.dumps(comparison,indent=2))
        paths=[root/(name+'_seed42') for name in methods]+[old/(name+'_seed42') for name in ['D1_STATIC_BU','D2_DELAYED_EMA','D3_CONFIDENCE_BU']]
        analyze(paths,root/'noise_savings')
        lines=['# S2 / S3 延迟静态对照（seed42）','','完整 RDP、数据划分、模型和训练配置匹配；静态与对应动态组全部 25 轮掩码、裁剪权重、噪声税一致。',
               '', '| 静态对照 | 静态峰值 | 动态方法 | 动态峰值 | 动态−静态峰值差 | 静态最终 | 动态最终 | 最终差 |','|---|---:|---|---:|---:|---:|---:|---:|']
        for r in comparison:
            lines.append(f"| {r['static']} | {r['static_best']:.2f}% | {r['dynamic']} | {r['dynamic_best']:.2f}% | {r['dynamic_minus_static_best_pp']:+.2f} pp | {r['static_final']:.2f}% | {r['dynamic_final']:.2f}% | {r['dynamic_minus_static_final_pp']:+.2f} pp |")
        lines.extend(['','S2 固定 conv1@15→conv2@17，S3 固定 conv2@15→conv1@17，不读取任何 DP 反馈。对应动态方法在旧运行产生了相同掩码，因此此处不存在已观察到的掩码/噪声节省独立贡献。',
                      '各次运行的私有采样/噪声随机源独立；仅 seed42 的精度差不能用于因果归因或显著性判断。此对照在观察 D2/D3 后定义，属于事后诊断。',
                      '若动态与静态实际掩码在同一随机历史下完全相同，反馈评分作为后处理本身不会改变训练更新；需要跨种子检验它是否产生有益的不同决策。',
                      '累计噪声节省应按 d_t v_t 加权，详细结果见 [噪声能量分析](noise_savings/report_zh.md)。其参照为固定方差的全活跃维度，不是真实 NF 训练，也不是实测噪声向量。',
                      '未运行 seed43/44 或 E5；多种子应同时包含这些匹配静态对照，再决定是否推进保守动态冻结与 4A 的联合消融。'])
        (root/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
        status('completed');print(json.dumps(comparison,indent=2),flush=True)
    except BaseException as exc:status('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc));raise


if __name__=='__main__':main()
