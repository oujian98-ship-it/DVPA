import copy
import csv
import itertools
import json
import numpy as np
import pytest
import yaml
from src.p4_audit import audit_run, match_runs, spent, increment
from src.p4_evaluation import evaluate_manifest, validate_manifest, write_evaluation
from src.p4_metrics import mean_interval, paired_comparison, decision_metrics, estimator_metrics, client_utility
from src.p4_oracle import conditional_decomposition, analyze_oracle


def save_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def make_run(path,seed=1,mode='NF',acc=70.,freeze=False,sigmas=(2.,2.)):
    """Small hand-built artifact fixture, not a training/evaluation experiment."""
    path.mkdir()
    cfg=dict(seed=seed,mode=mode,rounds=2,clients=2,batch_size=1,local_epochs=1,lr=.1,rate=1.,clip=2.,
             epsilons=[10.,20.,30.],q_save=[1.,1.,1.],q_spend=1.,transition_round=2,
             data_root='synthetic_fixture_only',profile='joint_corrected',privacy_enabled=True,
             engineering_only=False,freeze_rounds=[2] if freeze else [])
    (path/'config.yaml').write_text(yaml.safe_dump(cfg))
    (path/'layer_registry.json').write_text(json.dumps(dict(dimensions=dict(a=2,b=1))))
    (path/'source_manifest.json').write_text(json.dumps({'src/models/model.py':'test_fixture_model_hash'}))
    orders=np.array([2.,3.]);rho=np.array([increment(tuple(orders),1.,s) for s in sigmas]);targets=np.array([10.,20.])
    product=2./np.mean(1/np.asarray(sigmas));variance=(product*.1/2)**2
    privacy=[];rounds=[];layers=[]
    for t in [1,2]:
        epsilons=[spent(orders,r*t,1e-5) for r in rho];counts=dict(a=0 if freeze and t==2 else 2,b=1)
        d=sum(counts.values());K=sum(v>0 for v in counts.values())
        for i in [0,1]:privacy.append(dict(round=t,client_slot=i,q=1.,sigma=sigmas[i],clip=product/sigmas[i],target=targets[i],epsilon=epsilons[i]))
        rounds.append(dict(round=t,test_acc=acc-2+t,test_loss=1.,privacy_enabled=True,q_bar=1.,
                           epsilon_min=min(epsilons),epsilon_max=max(epsilons),active_groups=K,active_params=d,
                           active_param_fraction=d/3,noise_var_per_coordinate=variance,noise_energy_expected=d*variance,
                           communication_bytes_expected=8*d,train_loss_raw=1234.))
        for g,count in counts.items():
            w=1/K if count else 0.
            layers.append(dict(round=t,layer=g,mask=int(count>0),active_params=count,weight=w,
                               noise_tax=count*variance,clip_radius_mean=2*np.sqrt(w)))
    save_csv(path/'rounds.csv',rounds);save_csv(path/'layers.csv',layers);save_csv(path/'privacy.csv',privacy)
    np.savez(path/'accountant.npz',orders=orders,rdp=rho*2,targets=targets)
    (path/'summary.json').write_text(json.dumps(dict(seed=seed,mode=mode,status='p2_experiment',
        privacy_enabled=True,final_test_acc=acc,best_test_acc=acc,epsilon_n_final=epsilons,
        budget_violation_rate=0.,split_checksum=f'fixture_partition_seed{seed}',train_flops=None)))
    (path/'runtime_local.json').write_text(json.dumps(dict(wall_time_sec=3.2)))
    return path


def manifest_for(paths):
    return dict(version=1,data_scope='public_benchmark',cases=[dict(id='toy_contract',baseline='NF',
        budget_matching='same_rdp',methods=[dict(id=m,runs=[dict(seed=s,path=str(p),replicate=r) for s,r,p in entries]) for m,entries in paths.items()])])


@pytest.mark.parametrize('kind',['replay','delayed','confidence'])
def test_diagnostic_saved_transcript_audit_and_corruption(tmp_path,kind):
    from src.diagnostic_freezing import DiagnosticFreezer
    from src.dp_feedback_freezer import DEFAULTS
    path=make_run(tmp_path/'run',mode='ADAPTIVE',freeze=kind!='confidence')
    cfg=yaml.safe_load((path/'config.yaml').read_text());cfg['freeze_rounds']=[]
    feedback={**DEFAULTS,'warmup_rounds':1,'patience':1}
    policy=dict(kind=kind,min_freeze_round=2,min_active_fraction=.25,freeze_order='any',protected_groups=[])
    if kind=='replay':policy=dict(kind='replay',replay_rounds=[2])
    if kind=='confidence':policy.update(alpha_total=.05,drift_window=2,max_ratio_range=.25,drift_margin_ratio=.25)
    cfg.update(feedback=feedback,diagnostic_freezing=policy)
    (path/'config.yaml').write_text(yaml.safe_dump(cfg))
    rounds=list(csv.DictReader((path/'rounds.csv').open()));layers=list(csv.DictReader((path/'layers.csv').open()))
    for r in layers:r['released_dp_update_energy']=0.
    save_csv(path/'layers.csv',layers)
    controller=DiagnosticFreezer(['a','b'],dict(a=2,b=1),feedback,policy,2);rows=[]
    for t in [1,2]:
        active=controller.decide(t)
        stats={g:(controller.dimensions[g],0.) for g in active if active[g]}
        rows.extend(controller.observe_statistics(t,stats,float(rounds[t-1]['noise_var_per_coordinate'])))
    save_csv(path/'feedback.csv',rows);(path/'feedback_state.json').write_text(json.dumps(controller.state_dict()))
    assert len(audit_run(path)['_feedback'])==4
    rows[0]['score']+=1.;save_csv(path/'feedback.csv',rows)
    with pytest.raises(ValueError,match='diagnostic score'):audit_run(path)


def test_saved_run_audit_recomputes_budget_costs_and_censors_unreached_target(tmp_path):
    path=make_run(tmp_path/'run',freeze=True)
    r=audit_run(path,accuracy_target=80.)
    assert r['rounds_to_target'] is None and r['communication_MB'] is None and r['train_FLOPs'] is None
    assert r['communication_expected_MB']==pytest.approx(32/1e6) and r['wall_time_sec'] is None
    assert r['active_fraction_last']==pytest.approx(1/3)
    assert all('train_loss_raw' not in row for row in r['_rounds'])
    assert audit_run(path,accuracy_target=69.)['rounds_to_target']==1
    assert audit_run(path,include_local_cost=True)['wall_time_sec']==3.2


@pytest.mark.parametrize('corruption',['duplicate_privacy','missing_round','bad_rdp','bad_epsilon','wrong_final_acc',
    'bad_noise','reactivation','weight_sum','frozen_weight','wrong_mask','engineering','non_dp','wrong_clip'])
def test_saved_artifact_corruption_fails_closed(tmp_path,corruption):
    p=make_run(tmp_path/'run',freeze=True)
    if corruption=='bad_rdp':
        with np.load(p/'accountant.npz') as a:arrays={k:a[k] for k in a.files}
        arrays['rdp'][0,0]+=.1;np.savez(p/'accountant.npz',**arrays)
    elif corruption in {'engineering','non_dp','wrong_final_acc'}:
        s=json.loads((p/'summary.json').read_text())
        if corruption=='engineering':s['status']='engineering_only'
        elif corruption=='non_dp':s['privacy_enabled']=False
        else:s['final_test_acc']+=1
        (p/'summary.json').write_text(json.dumps(s))
    else:
        file='privacy.csv' if corruption in {'duplicate_privacy','bad_epsilon','wrong_clip'} else 'rounds.csv' if corruption in {'missing_round','bad_noise'} else 'layers.csv'
        rows=list(csv.DictReader((p/file).open()))
        if corruption=='duplicate_privacy':rows.append(rows[0])
        elif corruption=='bad_epsilon':rows[0]['epsilon']='0'
        elif corruption=='wrong_clip':rows[0]['clip']='9'
        elif corruption=='missing_round':rows.pop()
        elif corruption=='bad_noise':rows[0]['noise_energy_expected']='10'
        elif corruption=='reactivation':rows[0]['active_params']='0'
        elif corruption=='weight_sum':rows[0]['weight']='.9'
        elif corruption=='frozen_weight':rows[2]['weight']='.5'
        else:rows[0]['mask']='0'
        save_csv(p/file,rows)
    with pytest.raises(ValueError):audit_run(p)


def test_pair_matching_rejects_partition_budget_and_training_mismatch(tmp_path):
    a=audit_run(make_run(tmp_path/'a'));b=copy.deepcopy(a)
    b['_split']='other'
    with pytest.raises(ValueError,match='partition'):match_runs(a,b)
    b=copy.deepcopy(a);b['_rdp'][0,0]+=.1
    with pytest.raises(ValueError,match='RDP'):match_runs(a,b)
    match_runs(a,b,'same_cap')  # Explicit cap-only scope does not assert equal spend.
    b=copy.deepcopy(a);b['_config']['local_epochs']=9
    with pytest.raises(ValueError,match='training'):match_runs(a,b)


def test_random_replicates_are_averaged_before_seed_statistics(tmp_path):
    paths={'NF':[],'RAND':[]}
    for seed in [1,2,3]:
        paths['NF'].append((seed,'default',make_run(tmp_path/f'nf{seed}',seed=seed,acc=70.)))
        for rep,acc in [('x',60.+seed),('y',80.+seed),('z',70.+seed)]:
            paths['RAND'].append((seed,rep,make_run(tmp_path/f'rand{seed}{rep}',seed=seed,mode='RAND',acc=acc,freeze=True)))
    m=manifest_for(paths);result=evaluate_manifest(m,tmp_path)
    table=result['cases'][0]['main_table'][1]
    assert result['status']=='complete' and table['training_seeds']==3
    assert table['mean_final_acc']==72. and table['std']==1.
    pair=result['cases'][0]['paired_comparisons'][0]
    assert pair['gaps']==[1.,2.,3.] and pair['exact_two_sided_sign_flip_p']==.25
    file=tmp_path/'manifest.json';file.write_text(json.dumps(m))
    out=write_evaluation(result,file,tmp_path/'analysis')
    assert (out/'table_cost.csv').exists() and (out/'table_ablation.csv').exists()
    with pytest.raises(FileExistsError):write_evaluation(result,file,out)
    (paths['RAND'][-1][2]/'summary.json').unlink()
    incomplete=evaluate_manifest(m,tmp_path)
    assert incomplete['status']=='incomplete' and incomplete['cases'][0]['complete_seeds']==[1,2]


def test_registration_rejects_pseudoreplication_and_unequal_seed_sets():
    m=manifest_for({'A':[(1,'x','a')],'NF':[(1,'x','b')]})
    validate_manifest(m)
    bad=copy.deepcopy(m);bad['cases'][0]['methods'][0]['runs'].append(dict(seed=1,path='c',replicate='x'))
    with pytest.raises(ValueError):validate_manifest(bad)
    bad=copy.deepcopy(m);bad['cases'][0]['methods'][0]['runs'][0]['seed']=2
    with pytest.raises(ValueError):validate_manifest(bad)
    bad=copy.deepcopy(m);bad.update(include_local_cost=True,data_scope='private')
    with pytest.raises(ValueError):validate_manifest(bad)


def test_unknown_oracle_truth_is_not_a_negative_and_zero_denominators_are_null():
    r=decision_metrics([True,True,False,False,True],[True,False,True,False,None])
    assert r['tp']==r['fp']==r['fn']==r['tn']==1
    assert r['false_freeze_rate']==r['missed_freeze_rate']==r['f1']==.5 and r['excluded']==1
    r=decision_metrics([False],[False]);assert r['precision'] is None and r['f1'] is None
    with pytest.raises(ValueError):decision_metrics([True],[True,False])
    with pytest.raises(ValueError):decision_metrics([1],[True])


def test_pairing_and_degenerate_statistics():
    assert mean_interval([])['mean'] is None and mean_interval([1.])['ci95'] is None
    assert paired_comparison([1.,1.],[1.,1.])['exact_two_sided_sign_flip_p']==1.
    with pytest.raises(ValueError):paired_comparison([1.],[1.,2.])
    assert estimator_metrics([-1.,3.],[1.,1.],target='sampled_clipped_aggregate_energy')['rmse']==2.
    with pytest.raises(ValueError):estimator_metrics([1.],[1.],target='unclipped_shadow_energy')


def test_conditional_mse_matches_exact_bernoulli_enumeration_and_keeps_cross_terms():
    u=np.array([[2.,1.],[4.,3.]]);h=np.array([[1.,0.],[3.,0.]])
    v=np.array([[.5,0.],[1.,0.]]);q=np.array([.25,.75]);variance=.2
    r=conditional_decomposition(u,h,v,q,[True,False],variance,offline_only=True)
    exact=0.
    for choice in itertools.product([0,1],repeat=2):
        b=np.array(choice);prob=np.prod(np.where(b,q,1-q))
        error=u.mean(0)-(b[:,None]*v).sum(0)/q.sum()
        exact+=prob*(error@error+variance)
    assert r['conditional_mse']==pytest.approx(exact)
    assert r['combined_bias_sq']==pytest.approx(sum(r[k] for k in ['freeze_and_local_dynamics_bias_sq','clipping_bias_sq','sampling_mean_bias_sq','bias_cross_terms']))
    assert r['bias_cross_terms']!=0.
    with pytest.raises(ValueError):conditional_decomposition(u,h,v,q,[True,False],variance)


def test_oracle_join_has_one_round_lag_missing_truth_and_censored_delay():
    feedback=[dict(round=1,layer='a',observed=True,freeze_next_round=False,score=2.),
              dict(round=2,layer='a',observed=True,freeze_next_round=True,score=1.),
              dict(round=3,layer='a',observed=False,freeze_next_round=False,score=None)]
    oracle=[dict(decision_round=t,layer='a',mse_active=2.,mse_freeze=1.) for t in [1,2,3]]
    r=analyze_oracle(feedback,oracle,3,offline_only=True)
    assert r['joined'][0]['predicted'] is None and r['joined'][1]['predicted'] is False
    assert r['joined'][2]['predicted'] is True and r['decisions']['evaluable']==2
    assert r['delays'][0]['delay']==2 and r['mean_abs_consecutive_score_change']==1.
    feedback[1]['freeze_next_round']=False;feedback[2]['observed']=True
    r=analyze_oracle(feedback,oracle,3,offline_only=True)
    assert r['delays'][0]['censored'] and r['delays'][0]['delay'] is None
    oracle[1].update(mse_active=None,mse_freeze=None)
    assert analyze_oracle(feedback,oracle,3,offline_only=True)['decisions']['evaluable']==1
    with pytest.raises(ValueError):analyze_oracle(feedback,oracle,3)


def test_private_client_metrics_are_isolated_and_quantiles_are_actual():
    rows=[dict(client_slot=i,target=10.,accuracy=float(i*10)) for i in range(3)]
    with pytest.raises(ValueError):client_utility(rows)
    r=client_utility(rows,offline_only=True)[0]
    assert r['worst_accuracy']==0 and r['p50']==10.


def add_feedback_fixture(path):
    from src.dp_feedback_freezer import FeedbackFreezer
    import torch
    cfg=yaml.safe_load((path/'config.yaml').read_text());cfg.update(mode='ADAPTIVE',freeze_rounds=[])
    cfg['feedback']=dict(patience=1,warmup_rounds=1,cooldown_rounds=0)
    (path/'config.yaml').write_text(yaml.safe_dump(cfg))
    summary=json.loads((path/'summary.json').read_text());summary.update(mode='ADAPTIVE',status='p3_experiment',freeze_first_round={'a':2})
    (path/'summary.json').write_text(json.dumps(summary))
    freezer=FeedbackFreezer(['a','b'],cfg['feedback']);feedback=[]
    for t in [1,2]:
        active=freezer.decide(t)
        feedback.extend(freezer.observe(t,{g:torch.zeros(2 if g=='a' else 1) for g,v in active.items() if v},.04))
    save_csv(path/'feedback.csv',feedback)
    (path/'feedback_state.json').write_text(json.dumps(freezer.state_dict()))
    rows=list(csv.DictReader((path/'layers.csv').open()))
    for row in rows:row['released_dp_update_energy']=0.
    save_csv(path/'layers.csv',rows)


@pytest.mark.parametrize('corrupt',[None,'pending','frozen_score','state'])
def test_p3_feedback_saved_causality_is_independently_audited(tmp_path,corrupt):
    p=make_run(tmp_path/'p3',freeze=True);add_feedback_fixture(p)
    if corrupt=='state':
        s=json.loads((p/'feedback_state.json').read_text());s['last_observed']=1
        (p/'feedback_state.json').write_text(json.dumps(s))
    elif corrupt:
        rows=list(csv.DictReader((p/'feedback.csv').open()))
        if corrupt=='pending':rows[0]['freeze_next_round']='False'
        else:rows[2]['score']='0'
        save_csv(p/'feedback.csv',rows)
    if corrupt:
        with pytest.raises(ValueError):audit_run(p)
    else:
        r=audit_run(p);assert len(r['_feedback'])==4 and r['_feedback'][2]['score'] is None


def test_holm_and_numpy_boolean_inputs():
    from src.p4_metrics import holm_adjust
    assert holm_adjust([.01,.04,.03])==pytest.approx([.03,.06,.06])
    assert holm_adjust([.25,None])==[None,None]
    assert decision_metrics(np.array([True]),np.array([True]))['tp']==1


def test_oracle_strata_rank_and_resource_matched_gap():
    from src.p4_oracle import oracle_gap
    oracle=[dict(decision_round=2,layer=g,mse_active=1.,mse_freeze=2.,estimated_energy=e,
                 noiseless_energy=y,estimator_target='sampled_clipped_aggregate_energy',epsilon=2.,dimension=10)
            for g,e,y in [('a',1.,2.),('b',3.,4.)]]
    feedback=[dict(round=1,layer=g,observed=True,freeze_next_round=False) for g in ['a','b']]
    report=analyze_oracle(feedback,oracle,2,offline_only=True)
    assert report['estimator_strata'][0]['rmse']==1.
    assert report['layer_rank_by_round'][0]['spearman']==pytest.approx(1.)
    resources=dict(dataset='toy',model='toy',split_checksum='toy',seed=1,rounds=2,clients=2,local_epochs=1,sampling_policy='identical')
    a=dict(resources=resources,final_test_acc=80.,test_loss=.5);b=dict(resources=resources,final_test_acc=75.,test_loss=.8)
    assert oracle_gap(a,b,offline_only=True)['accuracy_gap_pp']==5.
    with pytest.raises(ValueError):oracle_gap(a,{**b,'resources':{**resources,'rounds':3}},offline_only=True)


def test_schema_only_cli_never_opens_run_files(tmp_path,monkeypatch,capsys):
    import scripts.evaluate_p4 as script
    manifest=manifest_for({'A':[(1,'x','absent_a')],'NF':[(1,'x','absent_b')]})
    p=tmp_path/'manifest.json';p.write_text(json.dumps(manifest))
    monkeypatch.setattr(script,'evaluate_manifest',lambda *_:pytest.fail('Schema check must not evaluate runs'))
    monkeypatch.setattr('sys.argv',['evaluate_p4','--manifest',str(p),'--check-manifest'])
    script.main();assert 'no evaluation' in capsys.readouterr().out


def test_plots_use_only_complete_synthetic_evaluation_artifact(tmp_path):
    from src.p4_plots import plot_evaluation, plot_oracle_report
    a=make_run(tmp_path/'a',acc=70.);b=make_run(tmp_path/'b',mode='BU',acc=71.,freeze=True)
    result=evaluate_manifest(manifest_for({'NF':[(1,'default',a)],'BU':[(1,'default',b)]}),tmp_path)
    status=plot_evaluation(result,tmp_path/'figures')
    assert len(status['produced'])==9 and all((tmp_path/'figures'/p).read_bytes().startswith(b'%PDF') for p in status['produced'])
    with pytest.raises(ValueError):plot_evaluation(dict(result,status='incomplete'),tmp_path/'no_figures')
    with pytest.raises(ValueError):plot_oracle_report({},tmp_path/'private_plots')


def test_privacy_curve_requires_matching_strength_cases_and_does_not_collapse_heterogeneity(tmp_path):
    a=make_run(tmp_path/'a');b=make_run(tmp_path/'b',mode='BU',freeze=True)
    manifest=manifest_for({'NF':[(1,'default',a)],'BU':[(1,'default',b)]})
    manifest['cases'][0]['privacy_curve_group']='toy_privacy'
    result=evaluate_manifest(manifest,tmp_path)
    assert len(result['privacy_utility'])==2
    ha=make_run(tmp_path/'ha',sigmas=(1.5,2.5));hb=make_run(tmp_path/'hb',mode='BU',freeze=True,sigmas=(1.5,2.5))
    heterogeneous=manifest_for({'NF':[(1,'default',ha)],'BU':[(1,'default',hb)]})
    heterogeneous['cases'][0]['privacy_curve_group']='heterogeneous'
    r=evaluate_manifest(heterogeneous,tmp_path)
    assert not r['privacy_utility'] and r['privacy_curve_status'][0]['status']=='incomplete_or_heterogeneous_epsilon'
    # Different compute cannot be connected into the same privacy/utility curve.
    c=make_run(tmp_path/'c');d=make_run(tmp_path/'d',mode='BU',freeze=True)
    for path in [c,d]:
        cfg=yaml.safe_load((path/'config.yaml').read_text());cfg['local_epochs']=99
        (path/'config.yaml').write_text(yaml.safe_dump(cfg))
    other=manifest_for({'NF':[(1,'default',c)],'BU':[(1,'default',d)]})['cases'][0]
    other.update(id='other_strength',privacy_curve_group='toy_privacy');manifest['cases'].append(other)
    with pytest.raises(ValueError,match='Privacy curve'):evaluate_manifest(manifest,tmp_path)


def test_privacy_curve_plot_and_private_oracle_plots_from_synthetic_inputs(tmp_path):
    from src.p4_plots import plot_evaluation, plot_oracle_report
    cases=[]
    for index,sigma in enumerate([2.,3.]):
        a=make_run(tmp_path/f'a{index}',sigmas=(sigma,sigma))
        b=make_run(tmp_path/f'b{index}',mode='BU',freeze=True,sigmas=(sigma,sigma))
        case=manifest_for({'NF':[(1,'default',a)],'BU':[(1,'default',b)]})['cases'][0]
        case.update(id=f'epsilon_{index}',privacy_curve_group='strengths');cases.append(case)
    result=evaluate_manifest(dict(version=1,cases=cases),tmp_path)
    status=plot_evaluation(result,tmp_path/'privacy_figures')
    assert any('fig_privacy_utility_' in f for f in status['produced'])
    feedback=[dict(round=1,layer='a',observed=True,freeze_next_round=False)]
    oracle=[dict(decision_round=2,layer='a',mse_active=1.,mse_freeze=2.,estimated_energy=-1.,
                 noiseless_energy=1.,estimator_target='sampled_clipped_aggregate_energy',epsilon=2.,dimension=10)]
    report=dict(oracle=analyze_oracle(feedback,oracle,2,offline_only=True))
    status=plot_oracle_report(report,tmp_path/'private_figures',offline_only=True)
    assert 'fig_freeze_decisions.pdf' in status['produced'] and 'fig_estimator_error_vs_dimension.pdf' in status['produced']


@pytest.mark.parametrize('change',[dict(noise_variance=-1.),dict(q=[0.,0.]),dict(active=[1,0]),
    dict(clipped=[[1.,1.],[1.,1.]]),dict(reference=[[1.],[2.]]),dict(q=[1.,2.])])
def test_invalid_conditional_inputs_fail_without_private_side_effects(change):
    values=dict(reference=[[1.,2.],[2.,1.]],masked_local=[[1.,0.],[2.,0.]],
                clipped=[[.5,0.],[1.,0.]],q=[.5,.5],active=[True,False],noise_variance=.1)
    values.update(change)
    with pytest.raises(ValueError):conditional_decomposition(**values,offline_only=True)


@pytest.mark.parametrize('problem',['future_pending','duplicate_oracle','unknown_mse','wrong_estimator_target','reactivation'])
def test_oracle_input_rejects_bad_timing_labels_and_targets(problem):
    feedback=[dict(round=1,layer='a',observed=True,freeze_next_round=True),
              dict(round=2,layer='a',observed=False,freeze_next_round=False)]
    oracle=[dict(decision_round=2,layer='a',mse_active=2.,mse_freeze=1.)]
    if problem=='future_pending':oracle[0]['decision_round']=3
    elif problem=='duplicate_oracle':oracle.append(oracle[0])
    elif problem=='unknown_mse':oracle[0]['mse_active']=None
    elif problem=='wrong_estimator_target':oracle[0].update(estimated_energy=1.,noiseless_energy=1.,estimator_target='unclipped')
    else:feedback[1]['observed']=True
    with pytest.raises(ValueError):analyze_oracle(feedback,oracle,2,offline_only=True)


def test_oracle_cli_requires_explicit_offline_flag_and_does_not_overwrite(tmp_path,monkeypatch):
    import scripts.analyze_p4_oracle as script
    path=tmp_path/'offline.json';path.write_text(json.dumps(dict(clients=[dict(client_slot=0,target=10.,accuracy=70.)])))
    output=tmp_path/'offline_report.json'
    monkeypatch.setattr('sys.argv',['oracle','--input',str(path),'--output',str(output)])
    with pytest.raises(SystemExit):script.main()
    assert not output.exists()
    monkeypatch.setattr('sys.argv',['oracle','--input',str(path),'--output',str(output),'--offline-only'])
    script.main();assert json.loads(output.read_text())['clients'][0]['mean_accuracy']==70.
    with pytest.raises(FileExistsError):script.main()


def test_registered_mask_replicates_reject_duplicates_and_wrong_provenance(tmp_path):
    p=make_run(tmp_path/'run');q=make_run(tmp_path/'other')
    manifest=manifest_for({'NF':[(1,'default',p)],'RAND':[(1,'11',q)]})
    method=manifest['cases'][0]['methods'][1];method['replicate_type']='mask_seed';method['runs'][0]['mask_seed']=11
    with pytest.raises(ValueError,match='mask seed'):evaluate_manifest(manifest,tmp_path)
    method['runs'].append(dict(seed=1,replicate='23',path='third',mask_seed=11))
    with pytest.raises(ValueError,match='distinct'):validate_manifest(manifest)
