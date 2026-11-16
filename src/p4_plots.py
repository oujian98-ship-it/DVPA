"""Saved-evaluation plots. This module never trains or reads datasets/models."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def _slug(value):
    text=str(value)
    return re.sub(r'[^A-Za-z0-9_.-]','_',text)[:60]+'_'+hashlib.sha256(text.encode()).hexdigest()[:8]


def plot_evaluation(result, output):
    if result.get('status')!='complete':raise ValueError('Plots require a complete audited evaluation')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    produced=[];skipped={}
    def save(fig,path):
        fig.tight_layout();fig.savefig(path);plt.close(fig);produced.append(str(path.relative_to(out)))
    for case in result['cases']:
        folder=out/_slug(case['id']);folder.mkdir()
        curves=[r for r in result['rounds'] if r['case']==case['id']]
        layers=[r for r in result['layers'] if r['case']==case['id']]
        for fields,filename in [(['test_acc'],'fig_accuracy_round.pdf'),
                                (['active_param_fraction','noise_energy_expected'],'fig_active_dimensions.pdf')]:
            fig,axes=plt.subplots(len(fields),1,squeeze=False,figsize=(6,3*len(fields)))
            for field,ax in zip(fields,axes[:,0]):
                for method in sorted({r['method'] for r in curves}):
                    # Replicate-average within seed, then across training seeds.
                    values=defaultdict(list)
                    for r in curves:
                        if r['method']==method:values[(int(r['round']),r['seed'])].append(float(r[field]))
                    times=sorted({t for t,_ in values});seedmeans={t:[np.mean(v) for (when,_),v in values.items() if when==t] for t in times}
                    means=np.array([np.mean(seedmeans[t]) for t in times])
                    ax.plot(times,means,label=method)
                    if all(len(seedmeans[t])>1 for t in times):
                        std=np.array([np.std(seedmeans[t],ddof=1) for t in times]);ax.fill_between(times,means-std,means+std,alpha=.15)
                events=sorted({t for r in result['runs'] if r['case']==case['id'] for t in r.get('freeze_event_rounds',[])})
                for i,t in enumerate(events):ax.axvline(t,color='grey',linestyle='--',alpha=.3,label='freeze event (any run)' if i==0 else None)
                ax.set(xlabel='Round',ylabel=field);ax.legend(fontsize='small')
            save(fig,folder/filename)
        groups=[r for r in result['privacy_groups'] if r['case']==case['id']]
        fig,ax=plt.subplots(figsize=(6,4))
        for method,target in sorted({(r['method'],r['target']) for r in groups}):
            by_time=defaultdict(list)
            for r in groups:
                if r['method']==method and r['target']==target:by_time[r['round']].append(r['epsilon_mean'])
            times=sorted(by_time);ax.plot(times,[np.mean(by_time[t]) for t in times],label=f'{method}, cap={target:g}')
        ax.set(xlabel='Round',ylabel='Group mean accounted epsilon');ax.legend(fontsize='x-small')
        save(fig,folder/'fig_privacy_spend.pdf')
        # Show individual registered runs, not a selected best-looking mask.
        for method,seed,rep in sorted({(r['method'],r['seed'],r['replicate']) for r in layers}):
            rows=[r for r in layers if (r['method'],r['seed'],r['replicate'])==(method,seed,rep)]
            names=list(dict.fromkeys(r['layer'] for r in rows));times=sorted({int(r['round']) for r in rows})
            lookup={(r['layer'],int(r['round'])):r for r in rows}
            for field in ['mask','weight','noise_tax']:
                data=np.array([[float(lookup[(g,t)][field]) for t in times] for g in names])
                fig,ax=plt.subplots(figsize=(6,3));im=ax.imshow(data,aspect='auto',interpolation='nearest')
                ax.set_yticks(range(len(names)),names);ax.set_xticks(range(len(times)),times)
                ax.set(xlabel='Round',title=f'{method}, seed {seed}, replicate {rep}: {field}')
                fig.colorbar(im,ax=ax)
                save(fig,folder/f'heatmap_{_slug(method)}_s{seed}_{_slug(rep)}_{field}.pdf')
        skipped[case['id']]=dict(freeze_decisions='requires isolated oracle report',
                                  estimator_error='requires isolated noiseless aggregate truth',
                                  privacy_utility='requires separately registered privacy-strength cases with comparable model/data/training; heterogeneous epsilon is not reduced to one scalar')
    for group in sorted({r['group'] for r in result.get('privacy_utility',[])}):
        points=[r for r in result['privacy_utility'] if r['group']==group]
        if len({r['epsilon'] for r in points})<2:
            skipped[group]='Privacy utility needs at least two registered strengths';continue
        fig,ax=plt.subplots(figsize=(6,4))
        for method in sorted({r['method'] for r in points}):
            rows=sorted([r for r in points if r['method']==method],key=lambda r:r['epsilon'])
            ax.plot([r['epsilon'] for r in rows],[r['mean_final_acc'] for r in rows],marker='o',label=method)
        ax.set(xlabel='Common final accounted epsilon (homogeneous clients)',ylabel='Final accuracy (%)');ax.legend()
        save(fig,out/('fig_privacy_utility_'+_slug(group)+'.pdf'))
    status=dict(produced=produced,skipped=skipped,scope='saved-log descriptive plots; bands are across-seed standard deviations')
    (out/'plot_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    return status


def plot_oracle_report(report, output, *, offline_only=False):
    if not offline_only:raise ValueError('Oracle plots require private offline scope')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=report['oracle'];out=Path(output);out.mkdir(parents=True,exist_ok=False);produced=[]
    joined=data['joined'];names=sorted({r['layer'] for r in joined});times=sorted({r['decision_round'] for r in joined})
    if joined:
        lookup={(r['layer'],r['decision_round']):r for r in joined}
        fig,axes=plt.subplots(2,1,figsize=(6,5),squeeze=False)
        for field,ax in zip(['truth','predicted'],axes[:,0]):
            matrix=np.array([[np.nan if lookup.get((g,t),{}).get(field) is None else float(lookup[(g,t)][field]) for t in times] for g in names])
            ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',vmin=0,vmax=1,interpolation='nearest')
            ax.set_yticks(range(len(names)),names);ax.set_xticks(range(len(times)),times)
            ax.set(title=f'{field} (blank = unavailable)',xlabel='Effective decision round')
        fig.tight_layout();fig.savefig(out/'fig_freeze_decisions.pdf');plt.close(fig);produced.append('fig_freeze_decisions.pdf')
    estimator=data['estimator']
    if estimator['n']:
        fig,ax=plt.subplots(figsize=(4,3));ax.bar(['Bias','RMSE'],[estimator['bias'],estimator['rmse']])
        ax.set(ylabel='Energy estimation error',title='Isolated offline aggregate-energy diagnostic')
        fig.tight_layout();fig.savefig(out/'fig_estimator_error.pdf');plt.close(fig);produced.append('fig_estimator_error.pdf')
        for axis in ['epsilon','dimension']:
            strata=[r for r in data.get('estimator_strata',[]) if r.get(axis) is not None]
            if strata:
                fig,ax=plt.subplots(figsize=(5,3));ax.scatter([r[axis] for r in strata],[r['rmse'] for r in strata])
                ax.set(xlabel=axis,ylabel='Estimator RMSE',title='Declared offline strata')
                fig.tight_layout();name=f'fig_estimator_error_vs_{axis}.pdf';fig.savefig(out/name);plt.close(fig);produced.append(name)
    status=dict(scope='private offline only; not a DP publication artifact',produced=produced,
                note='Estimator summary is not an epsilon/dimension trend without stratified observations')
    (out/'plot_status.json').write_text(json.dumps(status,indent=2));return status
