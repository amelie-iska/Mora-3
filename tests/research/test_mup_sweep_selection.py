from dataclasses import replace
from decimal import Decimal as D
from hashlib import sha256

import pytest

from mora3.research.mup_sweep_selection import (FrozenProxySelection,FrozenValidationTask,ProxyRunOutcome,
    ValidationCoordinate,planned_proxy_trials,select_proxy_transfer)

def fixture():
    c=FrozenProxySelection('a'*64,'b'*64,4500,(
        FrozenValidationTask('bioelectric','c'*64,D('0'),D('0.01'),20),
        FrozenValidationTask('molecule','d'*64,D('-2'),D('.01'),100)))
    rows=[]
    for i,t in enumerate(planned_proxy_trials()):
        values=(ValidationCoordinate('bioelectric','c'*64,'validation',20,D('0.001')),
                ValidationCoordinate('molecule','d'*64,'validation',100,D('-2.4')))
        rows.append(ProxyRunOutcome(t.trial_id,'a'*64,sha256(t.trial_id.encode()).hexdigest(),'completed',5000,
            D(120 if i in (2,3) else 100),'b'*64,4500,values))
    return c,rows

def test_seed_mean_macro_with_zero_and_negative_reference_and_tie_breaking():
    c,rows=fixture()
    report=select_proxy_transfer(c,reversed(rows))
    assert report['frozen_transfer_hyperparameters']['hp_id']=='hp-01'
    assert D(report['selection_order'][0][1])==D('-0.05')
    assert report['selection_order'][-1][0]=='hp-02'
    assert report['eligible_hyperparameter_points']==12
    assert report['full_terminal_trial_count']==24 and not report['launch_authorized']

def test_retains_failed_run_and_never_chooses_its_successful_single_seed():
    c,rows=fixture()
    rows[0]=replace(rows[0],status='oom',completed_updates=12,failure_reason='bounded GPU allocation failed',
        validation_exposure_sha256=None,comparison_update=None,validation=())
    rows[2]=replace(rows[2],status='wall_time_censored',completed_updates=4900,gpu_seconds=D(1800))
    report=select_proxy_transfer(c,rows)
    assert report['ineligible_hyperparameter_points']==('hp-01',)
    assert report['retained_terminal_dispositions'][0][1]=='oom'
    assert report['frozen_transfer_hyperparameters']['hp_id']=='hp-03'

@pytest.mark.parametrize('case',['drop','duplicate','test','wrong-proof','incomplete','nan','budget','forged-zero-step'])
def test_incomparable_or_heldout_artifacts_fail_closed(case):
    c,rows=fixture()
    r=rows[0];a,b=r.validation
    if case=='drop':rows.pop()
    elif case=='duplicate':rows[-1]=r
    elif case=='test':rows[0]=replace(r,validation=(replace(a,split='test'),b))
    elif case=='wrong-proof':rows[0]=replace(r,validation=(replace(a,source_sha256='0'*64),b))
    elif case=='incomplete':rows[0]=replace(r,validation=(replace(a,observed_targets=19),b))
    elif case=='nan':rows[0]=replace(r,validation=(replace(a,mean_loss=D('NaN')),b))
    elif case=='budget':rows[0]=replace(r,gpu_seconds=D(1801))
    else:rows[0]=replace(r,completed_updates=0)
    with pytest.raises(ValueError):select_proxy_transfer(c,rows)


def test_macro_averages_within_family_before_equal_weighting_families():
    keys = ('forward','inverse','unpaired')
    tasks = tuple(FrozenValidationTask(key,char*64,D('1'),D('.01'),4)
                  for key,char in zip(keys,'cde',strict=True))
    contract = FrozenProxySelection('a'*64,'b'*64,4500,tasks)
    families = tuple(zip(keys,('paired','paired','single'),strict=True))
    grouped = replace(contract,task_families=families)
    outcomes = []
    for trial in planned_proxy_trials():
        if trial.hp_id=='hp-01':
            means = ('1','1','2.5')
        elif trial.hp_id=='hp-02':
            means = ('1.6','1.6','1.6')
        else:
            means = ('3','3','3')
        coordinates = tuple(ValidationCoordinate(t.task_key,t.validation_source_sha256,'validation',
                                                 t.eligible_observed_targets,D(mean))
                            for t,mean in zip(tasks,means,strict=True))
        outcomes.append(ProxyRunOutcome(trial.trial_id,'a'*64,sha256(trial.trial_id.encode()).hexdigest(),
                                        'completed',5000,D(100),'b'*64,4500,coordinates))
    legacy = select_proxy_transfer(contract,outcomes)
    balanced = select_proxy_transfer(grouped,outcomes)
    assert legacy['frozen_transfer_hyperparameters']['hp_id']=='hp-01'
    assert 'validation_aggregation' not in legacy
    assert balanced['frozen_transfer_hyperparameters']['hp_id']=='hp-02'
    assert balanced['validation_aggregation']==dict(policy='source-family-macro-v1',task_families=families)
    assert dict((key,D(value)) for key,value,_ in balanced['selection_order'])['hp-01']==D('.75')
    assert D(balanced['selection_order'][0][1])==D('.60')
    for malformed in (families[:2],families[::-1],families[:2]+(('unpaired',' '),)):
        with pytest.raises(ValueError,match='family'):
            replace(contract,task_families=malformed)
