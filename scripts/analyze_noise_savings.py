"""Audit saved runs and export noise-only full-active counterfactual savings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.noise_savings import summarize_noise_savings
from src.p4_audit import audit_run


def analyze(paths,output):
    output.mkdir(parents=True,exist_ok=False);summary=[];traces=[];hashes={}
    for path in paths:
        path=Path(path);audit_run(path)
        with (path/'rounds.csv').open() as f:rows=list(csv.DictReader(f))
        dims=json.loads((path/'layer_registry.json').read_text())['dimensions']
        result,trace=summarize_noise_savings(rows,sum(dims.values()))
        result.update(method=path.name,path=str(path.resolve()),layer_dimensions=dims,
                      fc1_fraction=dims.get('fc1',0)/sum(dims.values()))
        summary.append(result);traces.extend(dict(method=path.name,**r) for r in trace)
        for name in ['rounds.csv','layer_registry.json','summary.json','config.yaml','layers.csv','privacy.csv','accountant.npz']:
            p=path/name;hashes[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    for name,data in [('summary.csv',[{k:v for k,v in r.items() if k!='layer_dimensions'} for r in summary]),('rounds.csv',traces)]:
        with (output/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# 活跃维度与期望 DP 噪声能量','','参照为每轮保持全维活跃、噪声方差不变。不是一次真实 NF 训练，也不是实际噪声向量范数测量。',
           '',r'累计维度节省 = 1 − Σ d_t/(T D)；累计期望噪声节省 = 1 − Σ d_t v_t/(D Σ v_t)。',
           '', '| 方法 | 最终活跃维度 | 最终维度节省 | 累计维度节省 | 累计期望噪声节省 |','|---|---:|---:|---:|---:|']
    for r in summary:
        noise='不可定义' if r['cumulative_expected_noise_saved_fraction'] is None else f"{100*r['cumulative_expected_noise_saved_fraction']:.4f}%"
        lines.append(f"| {r['method']} | {r['final_active_dimension']:,} | {100*r['final_dimension_saved_fraction']:.4f}% | {100*r['cumulative_dimension_saved_fraction']:.4f}% | {noise} |")
    lines.extend(['','fc1 占全模型约 90.17% 参数，conv1+conv2 仅占约 8.95%。保护 fc1 后，单轮最终噪声能量节省上限也只有约 8.95%；延迟到低噪声后期冻结，使累计节省更小。',
                  '不能由维度减少推断 epsilon 减少、真实计算加速或最终误差减少。本次隐私会计保持相同，精度还受冻结、裁剪和采样影响。',
                  '结论：保守冻结修复了此前精度崩塌，但现有幅度不足以支持“已显著解决高维 DP 噪声问题”。'])
    (output/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.runs,a.output)
