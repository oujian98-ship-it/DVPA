"""Audit exact per-client RDP equality and report preregistered paired results."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import numpy as np
from scipy.stats import t as student_t

def audit_run(folder, summary, preflight, rounds_count, clients):
    """Independently verify saved ledgers, including every client/round row."""
    with np.load(folder/'accountant.npz') as ledger:
        record={k:ledger[k].copy() for k in ['orders','rdp','targets']}
    np.testing.assert_array_equal(record['orders'],preflight['orders'])
    if record['targets'].shape!=(clients,) or record['rdp'].shape!=(clients,len(record['orders'])):
        raise ValueError(f'Invalid accountant shape: {folder}')
    for i,target in enumerate(record['targets']):
        group=int(np.flatnonzero(preflight['targets']==target)[0])
        np.testing.assert_allclose(record['rdp'][i],preflight['rdp'][-1,group],rtol=0,atol=1e-12)
    seen=set()
    with (folder/'privacy.csv').open() as f:
        for row in csv.DictReader(f):
            t=int(row['round']);i=int(row['client_slot'])
            if not 1<=t<=rounds_count or not 0<=i<clients or (t,i) in seen:
                raise ValueError(f'Invalid or duplicate privacy row: {folder}')
            seen.add((t,i));target=float(row['target'])
            if target!=record['targets'][i]:raise ValueError('Client budget assignment changed')
            group=int(np.flatnonzero(preflight['targets']==target)[0])
            np.testing.assert_allclose([float(row[k]) for k in ['q','sigma','clip']],preflight['params'][t-1,group],rtol=0,atol=1e-12)
            epsilon=float(row['epsilon'])
            if not np.isfinite(epsilon) or epsilon>target+1e-6:
                raise ValueError('Invalid or exceeded privacy guarantee')
            np.testing.assert_allclose(epsilon,preflight['epsilon'][t-1,group],rtol=0,atol=1e-12)
            if t==rounds_count:
                np.testing.assert_allclose(epsilon,summary['epsilon_n_final'][i],rtol=0,atol=1e-12)
    if len(seen)!=rounds_count*clients:raise ValueError('Incomplete per-round privacy ledger')
    return record

def report(root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text())
    archive=root/protocol['privacy_preflight']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=protocol['privacy_preflight_sha256']:
        raise ValueError('Preflight checksum mismatch')
    preflight=np.load(archive)
    runs=[];by_seed={};audit=[]
    for seed in protocol['training_seeds']:
        baseline=None
        block={}
        for mode in ['NF','BU','TD','RAND']:
            items=[]
            for ms in protocol['random_mask_seeds'] if mode=='RAND' else [seed]:
                folder=root/f'{mode}_seed{seed}_mask{ms}'
                if not (folder/'summary.json').exists():continue
                s=json.loads((folder/'summary.json').read_text())
                if not s.get('exact_privacy_preflight_matched') or s.get('budget_violation_rate')!=0:
                    raise RuntimeError(f'Unvalidated privacy result: {folder}')
                record=audit_run(folder,s,preflight,protocol['fixed_config']['rounds'],protocol['fixed_config']['clients'])
                if mode=='NF':baseline=(record,s['split_checksum'])
                elif baseline is not None:
                    for k in record:np.testing.assert_allclose(record[k],baseline[0][k],rtol=0,atol=1e-12)
                    if s['split_checksum']!=baseline[1]:raise RuntimeError('Unmatched client partition')
                with (folder/'rounds.csv').open() as f:rounds=list(csv.DictReader(f))
                if len(rounds)!=protocol['fixed_config']['rounds']:raise RuntimeError('Incomplete training curve')
                items.append(s['final_test_acc'])
                runs.append(dict(seed=seed,mode=mode,mask_seed=ms,final_test_acc=s['final_test_acc'],active_params_last=int(rounds[-1]['active_params']),noise_energy_last=float(rounds[-1]['noise_energy_expected']),epsilon_max=float(rounds[-1]['epsilon_max']),privacy_matched=True,path=str(folder)))
            required=len(protocol['random_mask_seeds']) if mode=='RAND' else 1
            if len(items)==required:block[mode]=statistics.mean(items)
        if len(block)==4:by_seed[seed]=block
        audit.append(dict(seed=seed,complete_seed_block=len(block)==4,available_modes=list(block)))
    tables=[]
    for mode in ['NF','BU','TD','RAND']:
        values=[b[mode] for b in by_seed.values()]
        tables.append(dict(mode=mode,complete_training_seeds=len(values),mean_final_acc=statistics.mean(values) if values else None,std_final_acc=statistics.stdev(values) if len(values)>1 else None))
    comparisons=[]
    for comparator in ['NF','TD','RAND']:
        gaps=[b['BU']-b[comparator] for b in by_seed.values()]
        n=len(gaps);mean=statistics.mean(gaps) if n else None
        ci=None;p=None
        if n>1:
            half=float(student_t.ppf(.975,n-1))*statistics.stdev(gaps)/np.sqrt(n)
            ci=[mean-half,mean+half]
            permutations=[abs(np.mean(np.asarray(gaps)*signs)) for signs in itertools.product([-1,1],repeat=n)]
            p=sum(v>=abs(mean)-1e-12 for v in permutations)/len(permutations)
        comparisons.append(dict(comparison=f'BU - {comparator}',seeds=n,paired_gaps_pp=gaps,mean_gap_pp=mean,t_95_ci_pp=ci,exact_two_sided_sign_flip_p=p))
    complete=len(by_seed)==len(protocol['training_seeds'])
    if not complete:conclusion='INCOMPLETE: no final superiority conclusion until all registered runs complete.'
    elif all(c['mean_gap_pp']>0 for c in comparisons):
        conclusion='BU has the highest seed-averaged final accuracy in this fixed configuration; 3 seeds cannot establish general superiority or a two-sided sign-flip p<0.05.'
    else:
        not_better=[c['comparison'] for c in comparisons if c['mean_gap_pp']<=0]
        conclusion='The claim that BU beats all comparators is not supported in this configuration: '+', '.join(not_better)
    expected_runs=len(protocol['training_seeds'])*(3+len(protocol['random_mask_seeds']))
    result=dict(status='complete' if complete else 'running',completed_runs=len(runs),expected_runs=expected_runs,complete_seed_blocks=len(by_seed),privacy_audit='all completed runs matched preflight and corresponding NF ledgers; every saved client/round q, sigma, clip and epsilon checked',audit=audit,method_table=tables,paired_comparisons=comparisons,conclusion=conclusion,scope='Public FMNIST, fixed freeze rounds 9/17, same per-run RDP guarantees; not a multi-dataset or deployment claim')
    (root/'comparison.json').write_text(json.dumps(result,indent=2))
    seed_rows=[dict(seed=seed,**values) for seed,values in by_seed.items()]
    for name,rows in [('table_runs.csv',runs),('table_comparison.csv',tables),('table_paired.csv',comparisons),('table_seed_means.csv',seed_rows)]:
        if rows:
            with (root/name).open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Equal-budget Bottom-Up validation',f'\nStatus: {result["status"]}; completed {len(runs)}/{expected_runs} runs.\n', '| Method | Completed training seeds | Final accuracy mean (%) | Across-seed std |','|---|---:|---:|---:|']
    for r in tables:
        fmt=lambda v:'TBD' if v is None else f'{v:.3f}'
        lines.append(f'| {r["mode"]} | {r["complete_training_seeds"]} | {fmt(r["mean_final_acc"])} | {fmt(r["std_final_acc"])} |')
    lines.extend(['',conclusion,'','RAND masks are averaged within training seed before paired comparisons. See comparison.json for per-seed gaps, confidence intervals and exact sign-flip tests. No uncompleted run is treated as a result.'])
    lines.extend(['','## Complete training-seed blocks','', '| Training seed | NF (%) | BU (%) | TD (%) | RAND, mean of masks (%) |','|---|---:|---:|---:|---:|'])
    for r in seed_rows:
        lines.append(f'| {r["seed"]} | {r["NF"]:.3f} | {r["BU"]:.3f} | {r["TD"]:.3f} | {r["RAND"]:.3f} |')
    lines.extend(['','## Paired differences','', '| Comparison | Training seeds | Mean gap (pp) | Paired t 95% CI (pp) | Exact two-sided sign-flip p |','|---|---:|---:|---:|---:|'])
    for r in comparisons:
        ci=r['t_95_ci_pp'];ci_text='TBD' if ci is None else f'[{ci[0]:.3f}, {ci[1]:.3f}]'
        lines.append(f'| {r["comparison"]} | {r["seeds"]} | {fmt(r["mean_gap_pp"])} | {ci_text} | {fmt(r["exact_two_sided_sign_flip_p"])} |')
    lines.extend(['','Three training seeds cannot yield an exact two-sided sign-flip p below 0.25. Intervals with only three seeds are uncertain; this is a fixed-configuration effect estimate.','','## Completed individual runs','', '| Seed | Method | Mask seed | Final accuracy (%) | Active parameters, final round |','|---|---|---:|---:|---:|'])
    for r in runs:
        lines.append(f'| {r["seed"]} | {r["mode"]} | {r["mask_seed"]} | {r["final_test_acc"]:.2f} | {r["active_params_last"]} |')
    lines.extend(['','## Accounting and scope','',result['privacy_audit']+'.', '', 'All methods share the same per-client RDP trajectories (601 orders), delta=1e-5, partitions and fixed hyperparameters within training seed. DP noise and sampling remain independently randomized. Target epsilon groups are 10/20/30; final guarantees are recorded in each privacy.csv. These are per-run guarantees, not a combined budget for publishing every run.', '', 'Freeze rounds 9/17 and final-round test accuracy were fixed before training. Active dimension and remaining model capacity differ by order, so this comparison does not isolate direction from dimension. Historical source-compatible results use different clipping and cannot substitute for NF here. The P0 gate remains not_passed.'])
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    names={'NF':'不冻结 NF','BU':'自底向上 BU','TD':'反向冻结 TD','RAND':'随机冻结 RAND'}
    if not complete:zh_conclusion='实验尚未完成，不能给出最终优越性结论。'
    elif all(c['mean_gap_pp']>0 for c in comparisons):
        zh_conclusion='BU 在该固定配置下的平均最终准确率最高；三个训练种子不足以确认一般性、稳定或统计显著的优越性。'
    else:
        failed=[c['comparison'].split(' - ')[1] for c in comparisons if c['mean_gap_pp']<=0]
        zh_conclusion='本配置下不能支持“BU 同时优于所有对照”：BU 的平均最终准确率未超过 '+ '、'.join(failed)+'。'
    zh=['# Bottom-Up 同 DP 预算验证报告','',f'完成 {len(runs)}/{expected_runs} 次训练；完整训练种子组 {len(by_seed)}/{len(protocol["training_seeds"])}。','',zh_conclusion,'','| 方法 | 最终准确率均值（%） | 跨训练种子标准差 |','|---|---:|---:|']
    for r in tables:zh.append(f'| {names[r["mode"]]} | {fmt(r["mean_final_acc"])} | {fmt(r["std_final_acc"])} |')
    zh.extend(['','RAND 先在每个训练种子内平均三个掩码，再跨训练种子汇总；9 次 RAND 运行不视为 9 个独立训练种子。主指标事先固定为第 25 轮测试准确率。','','| 训练种子 | NF | BU | TD | RAND（三掩码均值） |','|---|---:|---:|---:|---:|'])
    for r in seed_rows:zh.append(f'| {r["seed"]} | {r["NF"]:.3f} | {r["BU"]:.3f} | {r["TD"]:.3f} | {r["RAND"]:.3f} |')
    zh.extend(['','| 配对比较 | 平均差值（百分点） | 配对 t 95% 区间 | 双侧精确 sign-flip p |','|---|---:|---|---:|'])
    for r in comparisons:
        ci=r['t_95_ci_pp'];ci_text='待完成' if ci is None else f'[{ci[0]:.3f}, {ci[1]:.3f}]'
        zh.append(f'| {r["comparison"]} | {fmt(r["mean_gap_pp"])} | {ci_text} | {fmt(r["exact_two_sided_sign_flip_p"])} |')
    eps_text='、'.join(f'{v:.6f}' for v in preflight['epsilon'][-1])
    zh.extend(['','所有已完成运行均通过独立审计：每个客户端每轮的 q、σ、裁剪半径、ε 对照预算预检；最终全部 RDP 阶数对照预检及同种子 NF；客户端划分 checksum 匹配；未发现超预算。',f'目标 ε 分组为 10、20、30；实际最终会计 ε 为 {eps_text}，δ=1e-5。比较匹配实际会计开销，不仅匹配预算上限。这里是每次运行的预算；联合发布所有运行的总隐私不能视作单次预算。','','实验范围为公开 FMNIST 官方 60000/10000 划分、100 个客户端、作者实际 582026 参数 CNN、25 全局轮、本地 30 epochs、固定第 9/17 轮冻结。三个训练种子为 42/43/44，随机掩码种子为 11/23/37；训练中未调参、挑种子或挑最高测试轮。','','三个训练种子时，双侧精确检验最小 p=0.25，不能据此声称 p<0.05 的显著优越。小样本 t 区间依赖分布假设，应谨慎解释。各冻结方向的活跃维度、模型容量与组间裁剪分配不同，因此此处不能单独证明冻结方向的因果优势；更小的噪声能量也不自动意味着更高准确率。P0 完整验收仍未通过，本结果不用于宣称已复现论文全部基线。','','协议见 protocol.json；逐次结果见 table_runs.csv；逐种子汇总见 table_seed_means.csv；配对统计见 comparison.json。'])
    (root/'report_zh.md').write_text('\n'.join(zh),encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results-dir',required=True);a=p.parse_args()
    r=report(a.results_dir);print(json.dumps(r,indent=2))
