"""E4: locked static BU vs Raw/Debiased/EMA DP feedback; seed 42 only; peak and final metrics."""
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
from src.upstream_runner import SOURCE
from src.p1_runner import run
from src.p4_evaluation import evaluate_manifest, write_evaluation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',required=True)
    args=parser.parse_args();root=Path(args.results_dir).resolve()
    root.mkdir(parents=True,exist_ok=False)
    equal=yaml.safe_load(Path('configs/p2_equal_bu.yaml').read_text())
    variants={'STATIC_BU':equal}
    for name,file in [('DP_RAW','p3_raw.yaml'),('DP_DEBIASED','p3_debiased.yaml'),('DP_EMA','p3_ema.yaml')]:
        variants[name]=yaml.safe_load((Path('configs')/file).read_text())
    excluded={'seed','mask_seed','mode','freeze_rounds','feedback'}
    fixed={k:v for k,v in equal.items() if k not in excluded}
    if any(fixed!={k:v for k,v in cfg.items() if k not in excluded} for cfg in variants.values()):raise ValueError('Unmatched feedback experimental configurations')
    methods={name:{k:cfg[k] for k in ('mode','freeze_rounds','feedback') if k in cfg} for name,cfg in variants.items()}
    (root/'status.json').write_text(json.dumps(dict(state='privacy_preflight',completed=[],total=len(methods)),indent=2))
    preflight(equal,root/'privacy_preflight.npz')
    sources=[p for p in SOURCE.rglob('*') if p.is_file()]+list(Path('src').glob('*.py'))
    sources.extend([Path(__file__),Path('scripts/validate_bottom_up.py')])
    protocol=dict(kind='controlled_feedback_comparison_v1',
        authorization='User explicitly requested E4 4B-only, seed42 only; P0 gate unchanged',
        fixed_config=fixed,methods=methods,training_seeds=[42],
        privacy_preflight='privacy_preflight.npz',
        privacy_preflight_sha256=hashlib.sha256((root/'privacy_preflight.npz').read_bytes()).hexdigest(),
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        primary_metric='best_test_acc',secondary_metric='final_test_acc',delta=1e-5,
        hypothesis='DP EMA minus static BU; equal_active clipping and complete per-client RDP',
        selection_policy='seed42 only per user; both full 25-round curves, no prior tuning or reruns',
        scope='single public-benchmark pair; peak metric requested by user; no robust/significance claim')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    manifest=dict(version=1,data_scope='public_benchmark',allow_engineering=False,include_local_cost=False,
        cases=[dict(id='E4_4B_only_seed42',baseline='STATIC_BU',budget_matching='same_rdp',accuracy_target=70.,
            methods=[dict(id=name,runs=[dict(seed=42,replicate='default',path=name+'_seed42')]) for name in methods])])
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));completed=[]
    for name,method in methods.items():
        directory=root/(name+'_seed42')
        (root/'status.json').write_text(json.dumps(dict(state='running',current=directory.name,completed=completed,total=len(methods)),indent=2))
        cfg=copy.deepcopy(fixed);cfg.update(seed=42,mask_seed=42,**method,validation_protocol=str(root/'protocol.json'))
        try:run(cfg,directory)
        except Exception as exc:
            (root/'status.json').write_text(json.dumps(dict(state='failed',current=directory.name,error=str(exc),completed=completed,total=len(methods)),indent=2));raise
        completed.append(directory.name)
    (root/'status.json').write_text(json.dumps(dict(state='training_completed_auditing',completed=completed,total=len(methods)),indent=2))
    result=evaluate_manifest(manifest,root);write_evaluation(result,root/'manifest.json',root/'analysis')
    table=[]
    for name in methods:
        folder=root/(name+'_seed42');s=json.loads((folder/'summary.json').read_text())
        with (folder/'rounds.csv').open() as f:rows=list(csv.DictReader(f))
        peak_round=next(int(r['round']) for r in rows if float(r['test_acc'])==s['best_test_acc'])
        table.append(dict(method=name,best_test_acc=s['best_test_acc'],best_round=peak_round,
                          final_test_acc=s['final_test_acc'],freeze_first_round=s['freeze_first_round']))
    gaps={r['method']:dict(peak_gap_pp=r['best_test_acc']-table[0]['best_test_acc'],
                          final_gap_pp=r['final_test_acc']-table[0]['final_test_acc']) for r in table[1:]}
    comparison=dict(training_seeds=1,table=table,primary_metric='best_test_acc',gaps_vs_static=gaps,
                    privacy='P4 audit passed, complete per-client RDP matched',
                    limitation='One seed, test-curve peak is descriptive; no robustness or significance claim')
    (root/'comparison.json').write_text(json.dumps(comparison,indent=2))
    lines=['# E4：4B-only（seed 42）','',
           '静态 BU 在第 9/17 轮冻结；Raw / Debiased / EMA 使用预设历史聚合反馈决定冻结。两组均为等份裁剪、关闭 4A。展示历程最高值与最终轮。',
           '', '| 方法 | 历程最高准确率 | 首次达到最高值轮次 | 最终准确率 |','|---|---:|---:|---:|']
    for r in table:lines.append(f'| {r["method"]} | {r["best_test_acc"]:.2f}% | {r["best_round"]} | {r["final_test_acc"]:.2f}% |')
    lines.extend(['', '实际冻结轮次：'+json.dumps({r['method']:r['freeze_first_round'] for r in table},ensure_ascii=False), '','相对静态 BU 的差值（百分点）：'+json.dumps(gaps,ensure_ascii=False),
                  '', 'P4 独立会计审计通过，同 seed 的完整 RDP、数据划分、模型和训练设置匹配。P0 gate 未改变。',
                  '仅一个种子，不能估计跨种子标准差/置信区间或判断稳健显著收益。峰值来自公开测试曲线的 25 次评估，不等于最终保存模型的准确率。未运行 seed43/44，也未按结果修改反馈阈值或冻结设置。'])
    (root/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    (root/'status.json').write_text(json.dumps(dict(state='completed',completed=completed,total=len(methods)),indent=2))
    print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':main()
