"""Seed42 D1 exact mask replay, then D2 delayed EMA and D3 confidence guards."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import secrets
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
from scripts.validate_bottom_up import preflight
from src.upstream_runner import SOURCE
from src.p1_runner import run
from src.p4_evaluation import evaluate_manifest,write_evaluation


def compare_replay(left,right):
    """Fail closed: exact full curves, masks, releases, model and final accountant."""
    checks={}
    for name in ['rounds.csv','layers.csv','privacy.csv','layer_registry.json','source_manifest.json']:
        checks[name]=left.joinpath(name).read_bytes()==right.joinpath(name).read_bytes()
    a=torch.load(left/'model.pt',map_location='cpu',weights_only=True)
    b=torch.load(right/'model.pt',map_location='cpu',weights_only=True)
    checks['model_tensors']=set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
    with np.load(left/'accountant.npz') as a,np.load(right/'accountant.npz') as b:
        checks['accountant']=set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
    return dict(passed=all(checks.values()),checks=checks,
                scope='Shared secret noise/sampling state for identical D1 pair only. Exact equality, not approximate final accuracy.')


def configurations(base,rng_path):
    common={k:v for k,v in base.items() if k not in {'seed','mask_seed','mode','freeze_rounds','feedback'}}
    feedback=copy.deepcopy(base['feedback'])
    delayed=copy.deepcopy(feedback);delayed['patience']=5
    guard=dict(min_freeze_round=15,min_active_fraction=.5,freeze_order='any',protected_groups=['fc1'])
    methods={
        'D1_STATIC_BU':dict(mode='BU',freeze_rounds=[9,17],diagnostic_rng_path=str(rng_path)),
        'D1_REPLAY_BU':dict(mode='ADAPTIVE',freeze_rounds=[],feedback=feedback,
                            diagnostic_freezing=dict(kind='replay',replay_rounds=[9,17]),diagnostic_rng_path=str(rng_path)),
        'D2_DELAYED_EMA':dict(mode='ADAPTIVE',freeze_rounds=[],feedback=delayed,
                              diagnostic_freezing=dict(kind='delayed',**guard)),
        'D3_CONFIDENCE_BU':dict(mode='ADAPTIVE',freeze_rounds=[],feedback=delayed,
                                diagnostic_freezing={**guard,'kind':'confidence','freeze_order':'bottom_up',
                                    'alpha_total':.05,'drift_window':3,'max_ratio_range':.25,'drift_margin_ratio':.25})}
    return common,methods


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results-dir',required=True)
    args=parser.parse_args();root=Path(args.results_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    def status(state,**fields):
        (root/'status.json').write_text(json.dumps(dict(state=state,completed=completed,total=4,**fields),indent=2))
    completed=[];status('privacy_preflight')
    base=yaml.safe_load(Path('configs/p3_ema.yaml').read_text())
    preflight(base,root/'privacy_preflight.npz')
    private=Path('.private_diagnostics').resolve();private.mkdir(exist_ok=True)
    rng_path=private/('d1_'+secrets.token_hex(12)+'.json')
    rng_path.write_text(json.dumps(dict(sampling=secrets.randbits(128),noise=secrets.randbits(63))))
    fixed,methods=configurations(base,rng_path)
    sources=[p for p in SOURCE.rglob('*') if p.is_file()]+list(Path('src').glob('*.py'))
    sources.extend([Path(__file__),Path('scripts/validate_bottom_up.py')])
    protocol=dict(kind='controlled_freezing_diagnostics_v1',data_scope='public_benchmark',
        authorization='User requested D1 then both D2 and D3 after D1 passes, seed42 only; P0 gate unchanged',
        fixed_config=fixed,methods=methods,training_seeds=[42],
        privacy_preflight='privacy_preflight.npz',privacy_preflight_sha256=hashlib.sha256((root/'privacy_preflight.npz').read_bytes()).hexdigest(),
        d1_rng_sha256=hashlib.sha256(rng_path.read_bytes()).hexdigest(),
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        gate='D1 exact round/layer/privacy/model/accountant equality; otherwise STOP before D2/D3',
        d2_recovery_definition='Final accuracy no more than 2 percentage points below fresh D1 static BU; descriptive one-seed threshold, not significance',
        primary_metric='best_test_acc',secondary_metric='final_test_acc',
        selection_policy='All preregistered seed42 methods, 25 rounds each; no tuning/reruns. D2/D3 independent private RNG.',
        confidence_scope='95% simultaneous historical clipped/sampled aggregate-energy upper bounds under conditional Gaussian noise, Bonferroni over 25*4 slots. Drift guards are heuristics; no future-utility certificate.',
        privacy_scope='Per-run RDP comparison only; not joint publication certification for all experiments. D1 duplicate transcripts share entropy locally; secret file must not be exported.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    manifest=dict(version=1,data_scope='public_benchmark',allow_engineering=False,include_local_cost=False,
                  cases=[dict(id='D1_D2_D3_seed42',baseline='D1_STATIC_BU',budget_matching='same_rdp',accuracy_target=70.,
                              methods=[dict(id=name,runs=[dict(seed=42,replicate='default',path=name+'_seed42')]) for name in methods])])
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
        for name,method in methods.items():
            status('running',current=name+'_seed42')
            cfg=copy.deepcopy(fixed);cfg.update(seed=42,mask_seed=42,**copy.deepcopy(method),validation_protocol=str(root/'protocol.json'))
            run(cfg,root/(name+'_seed42'));completed.append(name+'_seed42')
            if name=='D1_REPLAY_BU':
                gate=compare_replay(root/'D1_STATIC_BU_seed42',root/'D1_REPLAY_BU_seed42')
                (root/'d1_gate.json').write_text(json.dumps(gate,indent=2))
                if not gate['passed']:raise RuntimeError('D1 path equality failed; D2/D3 not authorized past this diagnostic gate')
                print('D1 exact replay gate PASSED; starting D2 and then D3',flush=True)
        status('training_completed_auditing')
        result=evaluate_manifest(manifest,root);write_evaluation(result,root/'manifest.json',root/'analysis')
        table=[]
        for name in methods:
            folder=root/(name+'_seed42');s=json.loads((folder/'summary.json').read_text())
            with (folder/'rounds.csv').open() as f:rows=list(csv.DictReader(f))
            table.append(dict(method=name,best_test_acc=s['best_test_acc'],best_round=next(int(r['round']) for r in rows if float(r['test_acc'])==s['best_test_acc']),
                              final_test_acc=s['final_test_acc'],freeze_first_round=s['freeze_first_round'],final_active_params=int(rows[-1]['active_params'])))
        baseline=table[0]
        for row in table:
            row['peak_gap_pp']=row['best_test_acc']-baseline['best_test_acc']
            row['final_gap_pp']=row['final_test_acc']-baseline['final_test_acc']
        comparison=dict(table=table,d1_passed=True,d2_recovered_within_2pp=table[2]['final_gap_pp']>=-2.,
                        caveat='One seed; D2/D3 are composite safeguards, no isolated causal attribution or future-utility guarantee. All comparisons complete-RDP matched.')
        (root/'comparison.json').write_text(json.dumps(comparison,indent=2))
        lines=['# D1 / D2 / D3（seed42）','','D1 通过精确轨迹回放检查。四组全部完成 25 轮；P4 会计审计和策略记录回放通过。4A 始终关闭。',
               '', '| 方法 | 历程最高 | 峰值轮 | 最终 | 峰值差 / pp | 最终差 / pp | 最终训练参数 |','|---|---:|---:|---:|---:|---:|---:|']
        for r in table:lines.append(f"| {r['method']} | {r['best_test_acc']:.2f}% | {r['best_round']} | {r['final_test_acc']:.2f}% | {r['peak_gap_pp']:+.2f} | {r['final_gap_pp']:+.2f} | {r['final_active_params']:,} |")
        lines.extend(['','实际冻结轮次：'+json.dumps({r['method']:r['freeze_first_round'] for r in table},ensure_ascii=False),
                      '', 'D2 最终值距新静态基线不超过 2 个百分点：'+str(comparison['d2_recovered_within_2pp']),
                      'D2 同时加入延迟、patience 和容量保护，不能据此把恢复唯一归因于其中一项。D3 同时改变评分和顺序，也不能单独归因于 UCB。',
                      'D3 的置信度针对历史已裁剪、已抽样聚合更新；历史稳定窗口和漂移余量仅为保守启发式，不构成对未来训练损失的保证。',
                      '仅 seed42，不计算跨种子稳健性结论；本次未调参和重跑。P0 gate 未改变。D1 的私有随机状态单独留存在可信本地目录，禁止随公开结果打包。'])
        (root/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
        status('completed');print(json.dumps(comparison,indent=2),flush=True)
    except BaseException as exc:
        status('failed' if not isinstance(exc,KeyboardInterrupt) else 'interrupted',error=str(exc));raise


if __name__=='__main__':main()
