"""Run the four paper-training-setting methods, only seed42, serially."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.paper_setting_fourway import (AUTHOR_ROOT, METHODS, audit, eps_from_rdp,  # noqa: E402
    increment, preflight, run, validate)
from src.st_audit import close  # noqa: E402
from src.st_runner import write_csv  # noqa: E402


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def implementation_hashes():
    paths = list((ROOT/'src').glob('*.py'))
    paths.append(Path(__file__))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def author_hashes():
    return {str(p.relative_to(AUTHOR_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in AUTHOR_ROOT.rglob('*') if p.is_file()}


def verify(root):
    lock = json.loads((root/'protocol_lock.json').read_text(encoding='utf-8'))
    if implementation_hashes()!=lock['implementation_sha256'] or author_hashes()!=lock['author_sha256']:
        raise RuntimeError('Locked implementation or author source changed')
    for name,digest in lock['artifacts_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Locked config/preflight changed: {name}')


def prepare(root):
    cfg = yaml.safe_load((ROOT/'configs/paper_setting_fourway_seed42.yaml').read_text(encoding='utf-8'))
    validate(cfg)
    root.mkdir(parents=True,exist_ok=False)
    dump(root/'state.json',dict(status='preparing',seed=42,completed=[],current=None))
    cached = ROOT/'reports/paper_setting_fourway_preflight.json'
    plan = json.loads(cached.read_text(encoding='utf-8')) if cached.exists() else preflight(cfg)
    # Independently check the public cached profile before trusting it.
    orders = tuple(plan['orders'])
    for axis in ('time','uniform'):
        if len(plan[f'{axis}_q'])!=cfg['rounds'] or len(plan[f'{axis}_sigma'])!=cfg['rounds']:
            raise ValueError('Wrong cached preflight horizon')
        for g in range(3):
            total = np.zeros(len(orders))
            for t in range(cfg['rounds']):
                expected_q = (cfg['q_save'][g] if t+1<cfg['transition_round'] else cfg['q_spend']) if axis=='time' else cfg['q_spend']
                close(plan[f'{axis}_q'][t][g],expected_q,'preflight q')
                total += increment(expected_q,plan[f'{axis}_sigma'][t][g],orders)
            actual = eps_from_rdp(total,orders,cfg['delta'])
            np.testing.assert_allclose(actual,plan['matched_actual_epsilon'][g],rtol=1e-7,atol=1e-6)
            if actual>cfg['epsilons'][g]+1e-6:
                raise RuntimeError('Public preflight exceeds privacy cap')
    (root/'config.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
    dump(root/'privacy_preflight.json',plan)
    (root/'source_snapshot').mkdir()
    for path in (ROOT/'src').glob('*.py'):
        shutil.copy2(path,root/'source_snapshot'/path.name)
    shutil.copy2(Path(__file__),root/'driver_snapshot.py')
    dump(root/'protocol_lock.json',dict(seed=42,methods=list(METHODS),
        implementation_sha256=implementation_hashes(),author_sha256=author_hashes(),
        artifacts_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                          for name in ('config.yaml','privacy_preflight.json')}))
    (root/'protocol_zh.md').write_text('''# 论文训练设置下的四组对照：仅seed42

用户指定截图中的ICLR2025设置，并明确只跑seed42，设置好后直接启动。
新结果目录隔离保存，不覆盖此前trynew、TAP或E5结果，作者原目录不修改。

## 共同底座

实际作者CNN（582026参数）；官方FMNIST 60k训练、10k独立公开测试；
作者Dirichlet alpha=.1划分（min-size/平衡循环），只使用训练集；像素[-1,1]。
100客户端、25轮、30本地epoch、batch125、lr.001、momentum.9、weight_decay5e-4；
原作者持久optimizer与CosineAnnealingLR(T_max200)，每个本地epoch调度。
保留作者的delta/lr*rate和回传lr*rate变换，rate=.9，裁剪参数c250。
每轮全部100客户端先本地训练，未入选客户端也更新其optimizer/scheduler。
三组比例34/43/23，epsilon上限10/20/30、delta1e-5；实际源扩展orders的采样高斯RDP。
隐私邻接：固定客户端槽位add/remove贡献；隐藏Poisson名单，私有独立噪声熵。
本次四组全部不冻结，不引入额外的静态/动态冻结因素，不使用在线DP反馈。

## 四组定义

|方法|时间|空间|
|---|---|---|
|UNIFORM 均匀分配|q恒为.9、sigma逐组恒定，逐轮等额RDP|层预算均匀|
|TIME_ONLY 仅时间调度|作者spend-as-you-go：前12轮.5/.6/.7，第13轮起.9|层预算均匀|
|LAYER_ONLY 仅层级调度|与UNIFORM相同|非均匀层预算、分层噪声|
|JOINT 时空联合调度|与TIME_ONLY相同|与LAYER_ONLY相同|

时间关闭时必须消融saving采样，否则不能称为逐轮均匀隐私消耗。
时间开启保留作者epsilon余量递推的完整sigma轮廓。
作者原时间调度没有完全耗尽10/20/30上限；均匀组按其实际最终epsilon校准，
使四组实际隐私保证匹配，而不是仅上限相同；实际值保存在privacy_preflight.json。
每个客户端保留全部orders的累计RDP，逐轮检查上限，发布前拒绝超预算。

## 真正的层预算与联合噪声

clip权重omega在四个Block上均为1/4，weight与bias联合裁剪；
修正原源码按模块数归一化却逐Tensor裁剪的敏感度问题，四组同时修正。
空间权重p_k(t)正比exp(-gamma_k*(t-1))，gamma为(.15,.10,.05,.01)，
对应conv1/conv2/fc1/fc；该先验沿用旧ST，不依据本次结果调参。
local_sigma_n和c_n满足c_n*sigma_n=c*global_sigma。
每层最终噪声标准差为：

    s_k = c*global_sigma*lr*rate/sum(q_n) * sqrt(omega_k/p_k)

因此层预算p确实控制噪声，但不改变clip半径。对客户端n，联合白化敏感度为：

    sum_k (c_n*sqrt(omega_k)*lr*rate/sum(q_n)/s_k)^2 = 1/sigma_n^2

先计算这一联合敏感度，再以(q_n,sigma_n)调用采样高斯RDP。
不把层当不相交数据集，不逐层重复计算采样放大。补偿未入选客户端噪声，
保证最终协方差不依赖实现入选人数；不公开入选名单或抽样/噪声种子。

## 报告与边界

四组重新训练，不复用旧精度；报告best/final、分组实际epsilon和完整RDP；
同时报告预期参与客户端轮次，全部方法本地训练次数均为2500客户端轮次。
一个种子只做描述，不计算跨种子标准差，不声称统计显著或稳定提升。
无结果驱动调参、无根据准确率提前停止、无替换失败训练。
采用干净官方split及修正的联合裁剪，属于论文训练设置下的受控扩展，非原代码逐位复现。
Dirichlet标签划分仅用于公开基准构造，隐私会计以固定客户端分区为条件；
不是对私有数据重新划分过程的DP认证，也不是四次训练联合公开的epsilon认证。
''',encoding='utf-8')
    return cfg,plan


def summarize(root,plan):
    verify(root)
    rows=[]
    reference=None
    for method in METHODS:
        folder=root/method
        checked=audit(folder,plan)
        dump(folder/'audit.json',checked)
        result=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        meta=json.loads((folder/'implementation.json').read_text(encoding='utf-8'))
        common={k:meta[k] for k in ('initial_model_sha256','split_checksum','group_assignment','parameters','dimensions')}
        if reference is None:
            reference=common
        elif common!=reference:
            raise ValueError('Initial model, client partition or privacy assignment differs between methods')
        used=result['group_epsilon_final']
        rows.append(dict(method=method,seed=42,best_test_accuracy=result['best_test_accuracy'],
            final_test_accuracy=result['final_test_accuracy'],epsilon_10_used=used[0],
            epsilon_20_used=used[1],epsilon_30_used=used[2],
            expected_participating_client_rounds=result['expected_client_rounds'],
            trained_client_rounds=2500,audit=checked['status']))
    joint=next(row for row in rows if row['method']=='JOINT')
    for row in rows:
        row['joint_minus_best_pp']=joint['best_test_accuracy']-row['best_test_accuracy']
        row['joint_minus_final_pp']=joint['final_test_accuracy']-row['final_test_accuracy']
    write_csv(root/'comparison.csv',rows)
    dump(root/'comparison.json',rows)
    lines=['# 论文训练设置下四组对照（seed42）','','全部方法完成25轮，采用实际作者训练底座和采样RDP会计。','',
        '|方法|最佳准确率%|最终准确率%|epsilon10组实际|epsilon20组实际|epsilon30组实际|审计|',
        '|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append(f'|{row["method"]}|{row["best_test_accuracy"]:.2f}|{row["final_test_accuracy"]:.2f}|'
            f'{row["epsilon_10_used"]:.8f}|{row["epsilon_20_used"]:.8f}|{row["epsilon_30_used"]:.8f}|{row["audit"]}|')
    lines+=['','实际消耗按每客户端完整采样RDP与实际层噪声联合敏感度核验，非只比较epsilon上限。',
        '均匀时间组对齐作者时间组的实际消耗；不冻结四组，层预算确实控制噪声。',
        '只有一个seed42，不声称稳定提升或统计显著。旧trynew结果不混入本表。']
    (root/'report_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    root=ROOT/'results/paper_setting_fourway_seed42'
    cfg,plan=prepare(root)
    state=dict(status='running',seed=42,methods=list(METHODS),completed=[],current=None,
        started_utc=datetime.now(timezone.utc).isoformat())
    dump(root/'state.json',state)
    try:
        for method in METHODS:
            verify(root)
            state['current']=method
            dump(root/'state.json',state)
            print(f'BEGIN PAPER-SETTING {method}: seed42',flush=True)
            run(cfg,plan,method,root/method)
            dump(root/method/'audit.json',audit(root/method,plan))
            state['completed'].append(method)
            dump(root/'state.json',state)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summarize(root,plan)
        state.update(status='completed',current=None,finished_utc=datetime.now(timezone.utc).isoformat())
        dump(root/'state.json',state)
    except Exception:
        state.update(status='failed',error=traceback.format_exc())
        dump(root/'state.json',state)
        raise


if __name__=='__main__':
    main()
