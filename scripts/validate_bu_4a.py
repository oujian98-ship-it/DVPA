"""E3: locked BU equal vs BU 4A; seed 42 only; peak and final metrics."""
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
    prior=yaml.safe_load(Path('configs/p2_4a_bu.yaml').read_text())
    excluded={'seed','mask_seed','allocation'}
    fixed={k:v for k,v in equal.items() if k not in excluded}
    if fixed!={k:v for k,v in prior.items() if k not in excluded}:raise ValueError('Unmatched BU experimental configurations')
    allocations={'BU_EQUAL':equal['allocation'],'BU_4A':prior['allocation']}
    (root/'status.json').write_text(json.dumps(dict(state='privacy_preflight',completed=[],total=2),indent=2))
    preflight(fixed,root/'privacy_preflight.npz')
    sources=[p for p in SOURCE.rglob('*') if p.is_file()]+list(Path('src').glob('*.py'))
    sources.extend([Path(__file__),Path('scripts/validate_bottom_up.py')])
    protocol=dict(kind='controlled_bu_allocation_comparison_v1',
        authorization='User explicitly requested E3 BU+4A, seed42 only; P0 gate unchanged',
        fixed_config=fixed,allocations=allocations,training_seeds=[42],
        privacy_preflight='privacy_preflight.npz',
        privacy_preflight_sha256=hashlib.sha256((root/'privacy_preflight.npz').read_bytes()).hexdigest(),
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        primary_metric='best_test_acc',secondary_metric='final_test_acc',delta=1e-5,
        hypothesis='BU 4A minus BU equal_active; same static masks and complete per-client RDP',
        selection_policy='seed42 only per user; both full 25-round curves, no prior tuning or reruns',
        scope='single public-benchmark pair; peak metric requested by user; no robust/significance claim')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    manifest=dict(version=1,data_scope='public_benchmark',allow_engineering=False,include_local_cost=False,
        cases=[dict(id='E3_BU_4A_seed42',baseline='BU_EQUAL',budget_matching='same_rdp',accuracy_target=70.,
            methods=[dict(id=name,runs=[dict(seed=42,replicate='default',path=name+'_seed42')]) for name in allocations])])
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));completed=[]
    for name,allocation in allocations.items():
        directory=root/(name+'_seed42')
        (root/'status.json').write_text(json.dumps(dict(state='running',current=directory.name,completed=completed,total=2),indent=2))
        cfg=copy.deepcopy(fixed);cfg.update(seed=42,mask_seed=42,allocation=allocation,validation_protocol=str(root/'protocol.json'))
        try:run(cfg,directory)
        except Exception as exc:
            (root/'status.json').write_text(json.dumps(dict(state='failed',current=directory.name,error=str(exc),completed=completed,total=2),indent=2));raise
        completed.append(directory.name)
    (root/'status.json').write_text(json.dumps(dict(state='training_completed_auditing',completed=completed,total=2),indent=2))
    result=evaluate_manifest(manifest,root);write_evaluation(result,root/'manifest.json',root/'analysis')
    table=[]
    for name in allocations:
        folder=root/(name+'_seed42');s=json.loads((folder/'summary.json').read_text())
        with (folder/'rounds.csv').open() as f:rows=list(csv.DictReader(f))
        peak_round=next(int(r['round']) for r in rows if float(r['test_acc'])==s['best_test_acc'])
        table.append(dict(method=name,best_test_acc=s['best_test_acc'],best_round=peak_round,
                          final_test_acc=s['final_test_acc'],freeze_first_round=s['freeze_first_round']))
    peak_gap=table[1]['best_test_acc']-table[0]['best_test_acc']
    final_gap=table[1]['final_test_acc']-table[0]['final_test_acc']
    comparison=dict(training_seeds=1,table=table,primary_metric='best_test_acc',peak_gap_pp=peak_gap,
                    final_gap_pp=final_gap,privacy='P4 audit passed, complete per-client RDP matched',
                    limitation='One seed, test-curve peak is descriptive; no robustness or significance claim')
    (root/'comparison.json').write_text(json.dumps(comparison,indent=2))
    lines=['# E3：BU+4A（seed 42）','',
           '两组均使用第 9/17 轮 BU 冻结；仅层裁剪分配不同。按用户关注点展示历程最高值，同时保留最终轮。',
           '', '| 方法 | 历程最高准确率 | 首次达到最高值轮次 | 最终准确率 |','|---|---:|---:|---:|']
    for r in table:lines.append(f'| {r["method"]} | {r["best_test_acc"]:.2f}% | {r["best_round"]} | {r["final_test_acc"]:.2f}% |')
    lines.extend(['',f'4A−等份：最高值差 {peak_gap:+.2f} 个百分点，最终轮差 {final_gap:+.2f} 个百分点。',
                  '', 'P4 独立会计审计通过，同 seed 的完整 RDP、数据划分、模型和训练设置匹配。P0 gate 未改变。',
                  '仅一个种子，不能估计跨种子标准差/置信区间或判断稳健显著收益。峰值来自公开测试曲线的 25 次评估，不等于最终保存模型的准确率。未运行 seed43/44，也未按结果修改先验或冻结时刻。'])
    (root/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
    (root/'status.json').write_text(json.dumps(dict(state='completed',completed=completed,total=2),indent=2))
    print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':main()
