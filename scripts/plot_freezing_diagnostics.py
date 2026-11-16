"""Plot completed D1/D2/D3 saved public-benchmark curves and historical E4 EMA."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root,old,output):
    output.mkdir(parents=True,exist_ok=False)
    (output/'visual-contract.md').write_text(
        'Seed42 diagnostic evidence. Panels: (a) accuracy, (b) active fraction, '
        '(c) D3 guarded UCB/noise score. Static and D1 replay overlap exactly and '
        'are drawn once after checking D1 gate. E4 EMA is historical context with '
        'independent DP randomness. No uncertainty bands for one seed. Published '
        'DP scores stop at last active observation, not zero after freezing. '
        'Left shaded area in (c) forbids effective freezing before round15. '
        'No final-utility confidence guarantee is implied.',encoding='utf-8')
    if not json.loads((root/'d1_gate.json').read_text())['passed']:raise ValueError('D1 gate not passed')
    hashes={};source=[]
    def read(p):
        hashes[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
        with p.open() as f:return list(csv.DictReader(f))
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    for label,folder,color,style in [
        ('Static BU = D1 replay',root/'D1_STATIC_BU_seed42','#222222','-'),
        ('D2 delayed EMA',root/'D2_DELAYED_EMA_seed42','#D55E00','--'),
        ('D3 confidence BU',root/'D3_CONFIDENCE_BU_seed42','#0072B2','-.'),
        ('E4 EMA (historical)',old/'DP_EMA_seed42','#999999',':')]:
        rows=read(folder/'rounds.csv');x=[int(r['round']) for r in rows]
        if x!=list(range(1,26)):raise ValueError('Incomplete curve')
        axes[0].plot(x,[float(r['test_acc']) for r in rows],label=label,color=color,linestyle=style,linewidth=1.8)
        axes[1].step(x,[float(r['active_param_fraction']) for r in rows],where='post',color=color,linestyle=style,linewidth=1.8)
        source.extend(dict(method=label,round=int(r['round']),accuracy=float(r['test_acc']),active_fraction=float(r['active_param_fraction'])) for r in rows)
    feedback=read(root/'D3_CONFIDENCE_BU_seed42'/'feedback.csv')
    for g,color,style in zip(['conv1','conv2','fc1','fc'],['#0072B2','#D55E00','#009E73','#666666'],['-','--','-.',':']):
        rows=[r for r in feedback if r['layer']==g and r['observed']=='True']
        x=[int(r['round']) for r in rows];y=[float(r['score'])/float(r['noise_tax']) for r in rows]
        axes[2].plot(x,y,label=g,color=color,linestyle=style,linewidth=1.6)
        for r in rows:
            if r['freeze_next_round']=='True':
                axes[2].plot(int(r['round']),float(r['score'])/float(r['noise_tax']),'x',color=color,markersize=8,markeredgewidth=2)
    axes[2].axvspan(.5,14.5,color='#DDDDDD',alpha=.35,zorder=0)
    axes[2].axhline(1.,color='#222222',linestyle=(0,(4,4)),linewidth=1)
    axes[0].set(title='(a) Test accuracy',ylabel='Accuracy (%)',ylim=(0,80))
    axes[1].set(title='(b) Trainable capacity',ylabel='Active / total parameters',ylim=(-.02,1.04))
    axes[2].set(title='(c) D3: UCB + drift margin',ylabel='Guarded score / noise energy',ylim=(0,None))
    axes[0].legend(frameon=False,fontsize=8,loc='lower right');axes[2].legend(frameon=False,fontsize=8,ncol=2)
    for ax in axes:
        ax.set(xlabel='Round',xlim=(1,25));ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('D1 / D2 / D3 diagnostics: seed42, equal clipping, matched per-run RDP',fontsize=12)
    for ext in ['png','svg','pdf']:fig.savefig(output/f'diagnostics.{ext}',dpi=180)
    plt.close(fig)
    with (output/'curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source)
    (output/'source_hashes.json').write_text(json.dumps(hashes,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','old','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();plot(a.root,a.old,a.output)
