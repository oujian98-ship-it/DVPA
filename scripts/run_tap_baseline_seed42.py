"""Only seed42: adapted TAP prior baseline and identical-freeze control."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback
import warnings

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.st_audit import audit as audit_st, close  # noqa: E402
from src.st_runner import write_csv  # noqa: E402
from src.tap_baseline import AUTHOR_ROOT, AUTHOR_CLIENT, audit_baseline, run_baseline, source_schedule  # noqa: E402


VARIANTS = ('TAP_NF', 'TAP_SAME_FREEZE')
SHARED_MODULES = ('clients.py', 'data.py', 'layer_registry.py', 'metrics.py', 'models.py',
                  'st_runner.py', 'st_accountant.py', 'st_budget.py', 'st_mechanism.py')


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def live_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}


def author_hashes():
    return {str(p.relative_to(AUTHOR_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in AUTHOR_ROOT.rglob('*') if p.is_file()}


def verify(root):
    lock = json.loads((root/'protocol_lock.json').read_text(encoding='utf-8'))
    if live_hashes() != lock['source_sha256'] or author_hashes() != lock['author_sha256']:
        raise RuntimeError('Locked implementation or original author source changed')
    for filename, digest in lock['artifact_sha256'].items():
        if hashlib.sha256((root/filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Protocol artifact changed: {filename}')
    return lock


def prepare(root):
    # Reuse the already-registered seed42 JOINT config, with no retuning.
    historical = ROOT/'results/st_fourway_seed42'
    cfg = yaml.safe_load((historical/'configs/JOINT.yaml').read_text(encoding='utf-8'))
    if cfg['experiment']['seed'] != 42:
        raise ValueError('This request authorizes seed42 only')
    old_manifest = json.loads((historical/'JOINT/implementation.json').read_text(encoding='utf-8'))['source_sha256']
    for filename in SHARED_MODULES:
        if live_hashes()[filename] != old_manifest[filename]:
            raise RuntimeError(f'Shared training or accounting differs from historical ST: {filename}')
    root.mkdir(parents=True, exist_ok=False)
    (root/'source_snapshot').mkdir()
    for p in (ROOT/'src').glob('*.py'):
        shutil.copy2(p, root/'source_snapshot'/p.name)
    (root/'author_snapshot').mkdir()
    shutil.copy2(AUTHOR_ROOT/AUTHOR_CLIENT, root/'author_snapshot'/Path(AUTHOR_CLIENT).name)
    (root/'common_config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        schedule = source_schedule(cfg, q_save=.5, transition_round=13)
    dump(root/'source_schedule.json', schedule)
    dump(root/'preflight.json', dict(seed=42, training_and_accounting_identical_to_historical_ST=True,
        shared_modules=list(SHARED_MODULES), author_privacy_methods_executed=True,
        warnings=[str(item.message) for item in captured], sigma_scale=schedule['sigma_scale'],
        source_epsilon_reported_final=schedule['rows'][-1]['source_epsilon_reported'],
        matched_rho_total=sum(row['rho_planned'] for row in schedule['rows'])))
    files = ('common_config.yaml', 'source_schedule.json')
    dump(root/'protocol_lock.json', dict(seed=42, variants=list(VARIANTS), source_sha256=live_hashes(),
        author_sha256=author_hashes(), artifact_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}))
    (root/'protocol_zh.md').write_text('''# ICLR2025 TAP 在当前设置下的适配对照：仅seed42

用户最新指令只运行seed42，不启动seed43/44。两次完整训练：TAP_NF与TAP_SAME_FREEZE。
作者原目录保持不变，加载Client类，直接调用其set_local_sigma_and_sample_rate与rdp_accountant。
这是作者spend-as-you-go调度的适配baseline，不是论文Table2的原样复现。

共同设置：FMNIST官方60k训练/10k公开测试，固定索引分区100客户端；FedAvgCNN；
25轮、30本地epoch、batch125、lr.001、momentum.9、每轮重新初始化本地optimizer；
像素[0,1]；训练种子42。初始化与本地batch种子公式与现有ST完全一致。
共同隐私：固定客户端槽位add/remove邻接、受信任服务器聚合、epsilon10、delta1e-5；
固定阶Gaussian RDP过滤器，不使用采样放大折扣；实际rho总消耗与现有ST对齐。

来自论文第22页Table4：saving q=.5（统一epsilon10组），spending q=.9，transition13。
同一客户端预算下，source c_n=c，因此不用恢复个性化裁剪差异。
作者epochs=remaining_rounds的噪声校准和epsilon余量递推保留，以生成公开sigma时间轮廓。
按事先确定的公式进行一次全局缩放：
scale=sqrt(sum_t[1/(2*sigma_source[t]^2)] / (rho_cap*(1-1e-10)))。
用sigma_matched=scale*sigma_source获得相同共同会计消耗；不根据效用调参。
原始source epsilon仅为轮廓生成诊断，不能充当本实验实际隐私消耗。

TAP_NF：无冻结；使用有效的weight/bias联合分块裁剪和所有坐标同尺度噪声。
TAP_SAME_FREEZE：相同TAP调度，额外使用已有conv1@15、conv2@17冻结，作为掩码控制。
后者是额外控制组，不冒充原论文算法。两组均校正源码K模块/M张量的敏感度不一致。
原文Algorithm2呈现整模型裁剪，第9页实验又说明per-layer clipping；本次采用与ST一致的
有效分块裁剪，保证拼接半径.1。旧源码c=250包含lr/rate单位变换，不能直接当作模型差分半径。

TAP的q随轮次变化是方法的一部分，不强制与ST恒定q=.9相同；报告预期客户端参与次数。
TAP预期1770客户端轮次，ST为2250；本次不作相同计算量/通信量的效率优越性结论。
原源码实际CNN、非IID划分、像素[-1,1]、持久optimizer、cosine与weight_decay均未用于本次
共同harness；这些差异明确披露，不能将结果声称为作者实现未经改动的原样复现。

与现有seed42四组逐一比较，报告best和final，保留负结果。
只有seed42，不计算跨种子标准差，不挑选最优种子，不根据准确率提前停止或替换失败训练。
每次独立满足同一隐私约束，不宣称两次训练联合公开仍为epsilon10。
''', encoding='utf-8')
    return cfg, schedule


def summarize(root):
    verify(root)
    rows = []
    for method in ('UNIFORM', 'TIME_ONLY', 'LAYER_ONLY', 'JOINT'):
        folder = ROOT/'results/st_fourway_seed42'/method
        checked = audit_st(folder)
        result = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        rows.append(dict(method=f'ST_{method}', seed=42, best_test_accuracy=result['best_test_accuracy'],
            final_test_accuracy=result['final_test_accuracy'], rho_spent=result['rho_spent'],
            epsilon_bound=result['epsilon_bound'], delta=result['delta'], expected_client_rounds=2250.,
            audit=checked['status']))
    for variant in VARIANTS:
        folder = root/variant
        checked = audit_baseline(folder)
        dump(folder/'audit.json', checked)
        result = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        rows.append(dict(method=variant, seed=42, best_test_accuracy=result['best_test_accuracy'],
            final_test_accuracy=result['final_test_accuracy'], rho_spent=result['rho_spent'],
            epsilon_bound=result['epsilon_bound'], delta=result['delta'],
            expected_client_rounds=result['expected_client_rounds'], audit=checked['status']))
    for row in rows:
        for key in ('rho_spent', 'epsilon_bound', 'delta'):
            close(row[key], rows[0][key], f'common-budget.{key}')
    joint = next(row for row in rows if row['method']=='ST_JOINT')
    for row in rows:
        row['joint_minus_best_pp'] = joint['best_test_accuracy']-row['best_test_accuracy']
        row['joint_minus_final_pp'] = joint['final_test_accuracy']-row['final_test_accuracy']
    write_csv(root/'comparison.csv', rows)
    dump(root/'comparison.json', rows)
    lines = ['# seed42：ICLR2025 TAP 适配baseline与当前ST比较', '',
        '原论文方法的source-schedule适配baseline；共同harness与隐私成本匹配，非原表复现。', '',
        '| 方法 | 最佳准确率% | 最终准确率% | 实际rho | 预期客户端轮次 | JOINT最佳差值pp | JOINT最终差值pp |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["method"]} | {row["best_test_accuracy"]:.2f} | {row["final_test_accuracy"]:.2f} | '
            f'{row["rho_spent"]:.12f} | {row["expected_client_rounds"]:.0f} | '
            f'{row["joint_minus_best_pp"]:+.2f} | {row["joint_minus_final_pp"]:+.2f} |')
    lines += ['', '正差值表示ST_JOINT较高；负差值表示baseline较高。所有行审计通过。',
        'TAP_NF为不冻结主baseline；TAP_SAME_FREEZE为加入相同冻结的额外控制组。',
        '所有行采用epsilon10/delta1e-5及相同保守会计；rho消耗由实际敏感度/噪声重算。',
        '一个种子只作描述，不能声称稳定提升或显著性。具体适配差异见protocol_zh.md。']
    (root/'report_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return rows


def main():
    root = ROOT/'results/tap_matched_seed42'
    cfg, schedule = prepare(root)
    state = dict(status='running', seed=42, variants=list(VARIANTS), completed=[], current=None,
        started_utc=datetime.now(timezone.utc).isoformat())
    dump(root/'state.json', state)
    try:
        for variant in VARIANTS:
            verify(root)
            state['current'] = variant
            dump(root/'state.json', state)
            print(f'BEGIN {variant}: seed42 only', flush=True)
            run_baseline(cfg, schedule, root/variant, variant)
            dump(root/variant/'audit.json', audit_baseline(root/variant))
            state['completed'].append(variant)
            dump(root/'state.json', state)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summarize(root)
        state.update(status='completed', current=None, finished_utc=datetime.now(timezone.utc).isoformat())
        dump(root/'state.json', state)
    except Exception:
        state.update(status='failed', error=traceback.format_exc())
        dump(root/'state.json', state)
        raise


if __name__ == '__main__':
    main()
