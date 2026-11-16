"""Extract observed E4 scores; frozen rounds are missing observations, never zeros."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze(root, output):
    output.mkdir(parents=True,exist_ok=False)
    contract=('Evidence: E4 seed42 published DP feedback only. Three method panels share '
              'linear score/noise-tax axes; threshold=1; x marks last observation before '
              'freezing, annotated by effective freeze round. No interpolation after freeze. '
              'CSV stores every observed ratio including zero. No causal attribution to '
              'clipping/cancellation is possible from aggregate energy alone.')
    (output/'visual-contract.md').write_text(contract,encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.3),sharex=True,sharey=True,layout='constrained')
    colors=['#0072B2','#D55E00','#009E73','#666666'];styles=['-','--','-.',':']
    table=[];curves=[];hashes={}
    for ax,method in zip(axes,['DP_RAW','DP_DEBIASED','DP_EMA']):
        folder=root/(method+'_seed42')
        def read(name):
            p=folder/name;hashes[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
            with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))
        rows=read('feedback.csv');rounds={int(r['round']):r for r in read('rounds.csv')}
        for g,color,style in zip(['conv1','conv2','fc1','fc'],colors,styles):
            observed=[r for r in rows if r['layer']==g and r['observed']=='True']
            low=[int(r['round']) for r in observed if float(r['score'])<float(r['threshold'])]
            counted=[int(r['round']) for r in observed if int(r['low_count'])>0]
            selected=next((r for r in observed if r['freeze_next_round']=='True'),None)
            effective=int(selected['round'])+1 if selected else None
            table.append(dict(method=method,layer=g,first_below_threshold_round=min(low) if low else None,
                              first_counted_low_round=min(counted) if counted else None,
                              decision_round=int(selected['round']) if selected else None,freeze_round=effective,
                              score_ratio_at_decision=float(selected['score'])/float(selected['noise_tax']) if selected else None,
                              accuracy_at_decision=float(rounds[int(selected['round'])]['test_acc']) if selected else None,
                              accuracy_at_freeze=float(rounds[effective]['test_acc']) if effective in rounds else None))
            xs=[int(r['round']) for r in observed];ys=[float(r['score'])/float(r['noise_tax']) for r in observed]
            curves.extend(dict(method=method,layer=g,round=x,score_ratio=y) for x,y in zip(xs,ys))
            ax.plot(xs,ys,color=color,linestyle=style,label=g,linewidth=1.7)
            if selected:
                x=int(selected['round']);y=float(selected['score'])/float(selected['noise_tax'])
                ax.plot(x,y,'x',color=color,markersize=8,markeredgewidth=2)
                ax.annotate(f'{g}: F{effective}',(x,y),xytext=(4,10+12*(['conv1','conv2','fc1'].index(g))),textcoords='offset points',fontsize=8,color=color)
        ax.axhline(1.,color='#222222',linewidth=1,linestyle=(0,(4,4)))
        ax.set(title=method.replace('DP_',''),xlabel='Completed observation round',xlim=(1,25))
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Score / expected noise energy')
    axes[-1].legend(frameon=False,loc='upper right',ncol=2)
    fig.suptitle('E4: low-score observations precede permanent feature-layer freezing',fontsize=12)
    for suffix in ['png','pdf','svg']:fig.savefig(output/f'score_ratios.{suffix}',dpi=180)
    plt.close(fig)
    for name,data in [('layer_diagnostics.csv',table),('score_ratios.csv',curves)]:
        with (output/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (output/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    lines=['# E4 layer diagnostics','', 'First low includes warmup; first counted low excludes warmup. Score ratio is from the decision round, one round before effective freezing. Accuracy at freeze is evaluated after training that effective round.','', '| Method | Layer | First low | First counted low | Decision | Freeze | S/B at decision | Accuracy at freeze |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        fmt=lambda key:'—' if r[key] is None else str(r[key])
        ratio='—' if r['score_ratio_at_decision'] is None else f"{r['score_ratio_at_decision']:.6f}"
        acc='—' if r['accuracy_at_freeze'] is None else f"{r['accuracy_at_freeze']:.2f}%"
        lines.append(f"| {r['method']} | {r['layer']} | {fmt('first_below_threshold_round')} | {fmt('first_counted_low_round')} | {fmt('decision_round')} | {fmt('freeze_round')} | {ratio} | {acc} |")
    lines.extend(['','Aggregate DP energy alone cannot distinguish clipping bias, client cancellation, or true convergence. These require additional diagnostics.','', '![Scores](score_ratios.png)'])
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.root,a.output)
