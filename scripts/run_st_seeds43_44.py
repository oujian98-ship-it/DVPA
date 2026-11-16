"""Run eight registered trainings serially; keep the completed seed42 intact."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.st_audit import audit, close  # noqa: E402
from src.st_fourway import METHODS, LABELS, configurations, dump, execute, prepare, source_hashes  # noqa: E402
from src.st_runner import write_csv  # noqa: E402


def check_against_seed42(base):
    historical = ROOT/'results/st_fourway_seed42'
    lock = json.loads((historical/'protocol_lock.json').read_text(encoding='utf-8'))
    current = source_hashes()
    if set(current) != set(lock['source_sha256']):
        raise RuntimeError('Training source file set differs from seed42')
    differences = [name for name, digest in current.items() if digest != lock['source_sha256'][name]]
    if set(differences) - {'st_fourway.py'}:
        raise RuntimeError(f'Training/privacy implementation changed since seed42: {differences}')
    for seed in (43, 44):
        cfg = copy.deepcopy(base)
        cfg['experiment']['seed'] = seed
        for method, candidate in configurations(cfg).items():
            old = yaml.safe_load((historical/'configs'/f'{method}.yaml').read_text(encoding='utf-8'))
            old['experiment']['seed'] = seed
            if candidate != old:
                raise RuntimeError(f'{method} differs from seed42 beyond the training seed')
    return dict(training_and_privacy_sources_identical_to_seed42=True,
                changed_source_files=differences,
                reason='st_fourway.py only generalized seed42 literals to the requested seed')


def combined_report(suite):
    rows = []
    for seed in (42, 43, 44):
        folder = ROOT/'results'/f'st_fourway_seed{seed}'
        comparison = json.loads((folder/'comparison.json').read_text(encoding='utf-8'))
        if {r['method'] for r in comparison} != set(METHODS):
            raise RuntimeError('Missing factorial method')
        for row in comparison:
            if row['seed'] != seed:
                raise RuntimeError('Unexpected seed in comparison')
            audit(folder/row['method'])
            for key in ('epsilon_target', 'delta', 'rho_cap', 'rho_spent', 'alpha'):
                close(row[key], comparison[0][key], f'{seed}.{row["method"]}.{key}')
            rows.append(row)
    for row in rows:
        for key in ('epsilon_target', 'delta', 'rho_cap', 'rho_spent', 'alpha'):
            close(row[key], rows[0][key], f'cross-seed.{key}')
    write_csv(suite/'all_seeds_comparison.csv', rows)
    dump(suite/'all_seeds_comparison.json', rows)
    statistics_rows = []
    for method in METHODS:
        cells = [r for r in rows if r['method'] == method]
        statistics_rows.append(dict(method=method, seeds='42,43,44',
            best_accuracy_mean=statistics.mean(r['best_test_accuracy'] for r in cells),
            best_accuracy_sample_std=statistics.stdev(r['best_test_accuracy'] for r in cells),
            final_accuracy_mean=statistics.mean(r['final_test_accuracy'] for r in cells),
            final_accuracy_sample_std=statistics.stdev(r['final_test_accuracy'] for r in cells)))
    write_csv(suite/'three_seed_summary.csv', statistics_rows)
    lines = ['# trynew 四组实验：seed42/43/44', '',
             '保留历史seed42，追加seed43与44，每个种子四组；所有方法逐项核验同一隐私约束及实际记账成本。', '',
             '| 方法 | seed42 最佳/最终 % | seed43 最佳/最终 % | seed44 最佳/最终 % | 最佳均值±样本标准差 % | 最终均值±样本标准差 % |',
             '|---|---:|---:|---:|---:|---:|']
    for stat in statistics_rows:
        cells = [next(r for r in rows if r['seed']==seed and r['method']==stat['method']) for seed in (42,43,44)]
        values = [f'{r["best_test_accuracy"]:.2f}/{r["final_test_accuracy"]:.2f}' for r in cells]
        lines.append(f'| {LABELS[stat["method"]]} | '+ ' | '.join(values) +
            f' | {stat["best_accuracy_mean"]:.2f} ± {stat["best_accuracy_sample_std"]:.2f}' +
            f' | {stat["final_accuracy_mean"]:.2f} ± {stat["final_accuracy_sample_std"]:.2f} |')
    lines += ['', '全部逐运行结果及rho/RDP/剩余额度详见all_seeds_comparison.csv。',
              'rho/RDP消耗是由实际噪声反推的保守会计成本，不是观测到的真实隐私损失。',
              '每次训练独立满足epsilon10/delta1e-5；不把12次联合公开声称为epsilon10。',
              '三种子结果作描述统计，不宣称统计显著。没有重新运行或替换seed42。']
    (suite/'report_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    base = yaml.safe_load((ROOT/'configs/st_budget_fmnist.yaml').read_text(encoding='utf-8'))
    preflight = check_against_seed42(base)
    suite = ROOT/'results/st_seeds43_44'
    suite.mkdir(parents=True, exist_ok=False)
    dump(suite/'preflight.json', preflight)
    state = dict(status='preparing', seeds=[43,44], completed=[], current_seed=None,
                 started_utc=datetime.now(timezone.utc).isoformat())
    dump(suite/'state.json', state)
    try:
        for seed in (43,44):
            cfg = copy.deepcopy(base)
            cfg['experiment']['seed'] = seed
            prepare(cfg, ROOT/'results'/f'st_fourway_seed{seed}')
        state['status'] = 'running'
        for seed in (43,44):
            state['current_seed'] = seed
            dump(suite/'state.json', state)
            print(f'BEGIN SEED {seed}: four methods, full FMNIST', flush=True)
            execute(ROOT/'results'/f'st_fourway_seed{seed}')
            state['completed'].append(seed)
            dump(suite/'state.json', state)
        combined_report(suite)
        state.update(status='completed', current_seed=None,
                     finished_utc=datetime.now(timezone.utc).isoformat())
        dump(suite/'state.json', state)
        print('Seeds43/44 complete; seed42/43/44 aggregate report saved.', flush=True)
    except Exception:
        state.update(status='failed', error=traceback.format_exc())
        dump(suite/'state.json', state)
        raise


if __name__ == '__main__':
    main()
