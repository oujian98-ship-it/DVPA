"""Locked single-seed factorial experiment: uniform/time/layer/joint allocation."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch
import yaml

from .st_audit import audit, close
from .st_runner import preview, run, validate_config, write_csv


METHODS = ('UNIFORM', 'TIME_ONLY', 'LAYER_ONLY', 'JOINT')
LABELS = dict(UNIFORM='均匀分配', TIME_ONLY='仅时间调度', LAYER_ONLY='仅层级调度', JOINT='时空联合调度')


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


def configurations(base):
    cfg = validate_config(base)
    if cfg['data']['dataset'] != 'fmnist':
        raise ValueError('This protocol requires FMNIST')
    if cfg['budget'].get('feedback', False):
        raise ValueError('Four-way prior ablation disables feedback in all four cells')
    result = {}
    for method in METHODS:
        cell = copy.deepcopy(cfg)
        if method not in {'TIME_ONLY', 'JOINT'}:
            cell['budget']['time_weights'] = [1.] * cfg['train']['rounds']
        if method not in {'LAYER_ONLY', 'JOINT'}:
            cell['budget']['layer_weights'] = dict.fromkeys(cfg['clipping']['weights'], 1.)
            cell['budget']['layer_decay'] = dict.fromkeys(cfg['clipping']['weights'], 0.)
        cell['budget']['feedback'] = False
        result[method] = cell
    return result


def prepare(base, root):
    configs = configurations(base)
    seed = base['experiment']['seed']
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root/'configs').mkdir()
    (root/'source_snapshot').mkdir()
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copy2(p, root/'source_snapshot'/p.name)
    (root/'base_config.yaml').write_text(yaml.safe_dump(base), encoding='utf-8')
    hashes = {}
    plans = {}
    for method, cfg in configs.items():
        path = root/'configs'/f'{method}.yaml'
        path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
        hashes[method] = hashlib.sha256(path.read_bytes()).hexdigest()
        plans[method] = preview(cfg)
    # Same constraints, geometry and marginal temporal costs within each pair.
    reference = plans['UNIFORM']
    for method, plan in plans.items():
        close(plan['rho_cap'], reference['rho_cap'], f'{method}.rho_cap')
        close(plan['alpha'], reference['alpha'], f'{method}.alpha')
        if not plan['epsilon_accounting_value'] <= base['privacy']['epsilon']:
            raise ValueError('Preview exceeds privacy constraint')
        for a, b in zip(reference['layers'], plan['layers']):
            for key in ('active', 'dimensions', 'clip_radius', 'sensitivity', 'denominator'):
                close(a[key], b[key], f'{method}.{key}')
    for a, b in [('UNIFORM', 'LAYER_ONLY'), ('TIME_ONLY', 'JOINT')]:
        for left, right in zip(plans[a]['privacy'], plans[b]['privacy']):
            close(left['rho_increment'], right['rho_increment'], f'{a}/{b}.time_marginal')
    lock = dict(seed=seed, methods=list(METHODS), source_sha256=source_hashes(), config_sha256=hashes,
                privacy=base['privacy'], data=base['data'], train=base['train'],
                freezing=base['freezing'], clipping=base['clipping'],
                randomness=f'public model/local-order seed{seed}; independent secret sampling/noise entropy per run',
                scope='four separate per-run DP constraints; not a joint-publication privacy certificate')
    dump(root/'protocol_lock.json', lock)
    dump(root/'preflight.json', plans)
    dump(root/'state.json', dict(status='prepared', methods=list(METHODS), completed=[]))
    (root/'protocol_zh.md').write_text(f'''# trynew 四组实验：seed{seed}

授权：2026-10-24，最初完成seed42后，用户追加要求运行seed43、44。
本目录只运行训练种子{seed}，各方法其余参数保持初始seed42协议不变。

|方法|轮次总额度|活跃层预算比例|
|---|---|---|
|UNIFORM 均匀分配|各轮等额|各活跃层等额|
|TIME_ONLY 仅时间调度|既有公开时间先验|各活跃层等额|
|LAYER_ONLY 仅层级调度|各轮等额|既有公开层衰减先验|
|JOINT 时空联合调度|既有公开时间先验|既有公开层衰减先验|

四组反馈均关闭；“仅层级”仍允许层比例随轮次变化，其每轮总额度保持均匀。
冻结保持原配置conv1@15、conv2@17，四组相同；它不是本次对照因素。
FMNIST完整60k/10k、FedAvgCNN、100客户端、25轮、30本地epoch、batch125、lr.001、
momentum.9、固定索引客户端划分、统一模型差分总裁剪半径.1、q=.9、epsilon10、delta1e-5。
客户端槽位add/remove邻接，受信任服务器聚合；不使用采样放大。
四组统一alpha和rho_cap，发布前检查，并从实际敏感度及噪声独立重算账本。
公开训练种子只用{seed}；DP抽样及噪声独立秘密随机，不使用公开训练种子生成DP噪声。
串行执行顺序：UNIFORM、TIME_ONLY、LAYER_ONLY、JOINT。
不调参、不按准确率提前停止、不悄悄重跑或替换失败结果。

主指标：25轮最佳公开测试准确率；次指标：最终准确率。
同时报告rho_spent、rho_remaining、rho利用率、固定阶rdp_spent、epsilon诊断转换值与全局上限。
“实际消耗”指实际噪声机制对应的保守隐私记账成本，不是可观测的真实隐私损失。
这四组是公开先验分配，支出确定；不将epsilon目标列代替成本核验。
仅一个种子，不计算跨种子置信区间或显著性。
每次运行单独满足约束，不宣称把四次私有训练一起公开仍只消耗epsilon10。
''', encoding='utf-8')
    return lock


def verify_lock(root):
    root = Path(root)
    lock = json.loads((root/'protocol_lock.json').read_text(encoding='utf-8'))
    if source_hashes() != lock['source_sha256']:
        raise RuntimeError('Source changed after protocol lock')
    for method in METHODS:
        if hashlib.sha256((root/'configs'/f'{method}.yaml').read_bytes()).hexdigest() != lock['config_sha256'][method]:
            raise RuntimeError(f'{method} config changed after protocol lock')
    return lock


def summarize(root):
    root = Path(root)
    lock = verify_lock(root)
    rows = []
    for method in METHODS:
        result_dir = root/method
        audited = audit(result_dir)
        dump(result_dir/'audit.json', audited)
        summary = json.loads((result_dir/'summary.json').read_text(encoding='utf-8'))
        implementation = json.loads((result_dir/'implementation.json').read_text(encoding='utf-8'))
        if implementation['source_sha256'] != lock['source_sha256']:
            raise ValueError('Training source differs from protocol lock')
        cfg = yaml.safe_load((result_dir/'config.yaml').read_text(encoding='utf-8'))
        expected = yaml.safe_load((root/'configs'/f'{method}.yaml').read_text(encoding='utf-8'))
        if cfg != expected:
            raise ValueError('Training config differs from protocol lock')
        with np.load(result_dir/'accountant.npz') as accounting:
            rdp_spent = float(accounting['rdp_spent'])
        for key in ('best_test_accuracy', 'final_test_accuracy', 'rho_spent', 'epsilon_accounting_value'):
            if not math.isfinite(summary[key]):
                raise ValueError(f'Non-finite {method}.{key}')
        rows.append(dict(method=method, seed=lock['seed'], best_test_accuracy=summary['best_test_accuracy'],
                         final_test_accuracy=summary['final_test_accuracy'],
                         epsilon_target=summary['epsilon_target'], epsilon_bound=summary['epsilon_bound'],
                         epsilon_accounting_value=summary['epsilon_accounting_value'], delta=summary['delta'],
                         rho_cap=summary['rho_cap'], rho_spent=summary['rho_spent'],
                         rho_remaining=summary['rho_cap']-summary['rho_spent'],
                         utilization=summary['rho_spent']/summary['rho_cap'],
                         alpha=summary['alpha'], rdp_spent=rdp_spent, audit=audited['status']))
    for row in rows:
        for key in ('rho_cap', 'rho_spent', 'alpha', 'epsilon_target', 'delta'):
            close(row[key], rows[0][key], f'{row["method"]}.same_privacy.{key}')
    write_csv(root/'comparison.csv', rows)
    dump(root/'comparison.json', rows)
    lines = [f'# 四组实验结果（训练种子{lock["seed"]}）', '',
             '每组独立核验相同隐私约束与实际机制记账消耗。四组均为同一配置的完整25轮训练。', '',
             '| 方法 | 最佳准确率 % | 最终准确率 % | rho实际记账 | rho剩余 | 利用率 % | epsilon诊断值 | 审计 |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f'| {LABELS[r["method"]]} | {r["best_test_accuracy"]:.2f} | {r["final_test_accuracy"]:.2f} | '
                     f'{r["rho_spent"]:.12f} | {r["rho_remaining"]:.3e} | {100*r["utilization"]:.8f} | '
                     f'{r["epsilon_accounting_value"]:.12f} | {r["audit"]} |')
    lines += ['', f'共同约束：epsilon={rows[0]["epsilon_target"]}，delta={rows[0]["delta"]}，'
              f'alpha={rows[0]["alpha"]:.12f}，rho_cap={rows[0]["rho_cap"]:.12f}。',
              '“实际记账”由各层敏感度和真实噪声尺度重算，是保守会计上界，不是观测到的真实隐私损失。',
              '报告仅描述单种子结果，不作显著性或稳定增益结论。公开四次训练的联合隐私成本未认证。']
    (root/'report_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return rows


def execute(root):
    root = Path(root)
    lock = verify_lock(root)
    completed = []
    state = dict(status='running', completed=completed, current=None,
                 started_utc=datetime.now(timezone.utc).isoformat())
    dump(root/'state.json', state)
    try:
        for method in METHODS:
            verify_lock(root)
            state['current'] = method
            dump(root/'state.json', state)
            print(f'BEGIN {method} seed{lock["seed"]}', flush=True)
            cfg = yaml.safe_load((root/'configs'/f'{method}.yaml').read_text(encoding='utf-8'))
            run(cfg, root/method)
            dump(root/method/'audit.json', audit(root/method))
            completed.append(method)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', default='configs/st_budget_fmnist.yaml')
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--seed', type=int, help='Public training seed; leaves DP entropy private')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if not any((args.prepare, args.execute, args.summarize)):
        parser.error('Select --prepare, --execute or --summarize')
    if args.prepare:
        base = yaml.safe_load(Path(args.base_config).read_text(encoding='utf-8'))
        if args.seed is not None:
            base['experiment']['seed'] = args.seed
        prepare(base, args.results_dir)
        print('Four configurations and source snapshot locked.', flush=True)
    if args.execute:
        if args.seed is not None and verify_lock(args.results_dir)['seed'] != args.seed:
            parser.error('--seed differs from prepared protocol')
        execute(args.results_dir)
    if args.summarize:
        summarize(args.results_dir)


if __name__ == '__main__':
    main()
