"""New cross-checks of native validation-only selection; no campaigns launched."""
from dataclasses import replace
from decimal import Decimal as D
from hashlib import sha256
import pytest
from mora3.research.mup_sweep_selection import FrozenProxySelection,FrozenValidationTask,ProxyRunOutcome,ValidationCoordinate,planned_proxy_trials
from mora3.extensions.scri.transfer_audit import replay_primary_selection


def fixture():
    tasks=(FrozenValidationTask('a-spatial','c'*64,D('2'),D('.01'),64),FrozenValidationTask('b-bioelectric','d'*64,D('0'),D('.1'),7))
    contract=FrozenProxySelection('a'*64,'b'*64,4500,tasks,task_families=(('a-spatial','cells'),('b-bioelectric','simulation')))
    rows=[]
    for trial in planned_proxy_trials():
        score=D('.8')if trial.hp_id=='hp-04'else D('1.2')
        metrics=(ValidationCoordinate('a-spatial','c'*64,'validation',64,score),ValidationCoordinate('b-bioelectric','d'*64,'validation',7,D('.02')))
        rows.append(ProxyRunOutcome(trial.trial_id,'a'*64,sha256(trial.trial_id.encode()).hexdigest(),'completed',5000,D('100'),
            'b'*64,4500,metrics))
    return contract,rows


def test_all24_artifacts_select_only_complete_two_seed_validation_points():
    c,rows=fixture();r=replay_primary_selection(c,rows)
    assert r['full_terminal_trial_count']==24 and r['eligible_hyperparameter_points']==12
    assert r['frozen_transfer_hyperparameters']['hp_id']=='hp-04'
    assert r['target_validation_or_test_used'] is False and r['launch_authorized'] is False
    changed=list(rows);i=next(i for i,r in enumerate(changed)if r.trial_id.startswith('hp-04'))
    changed[i]=replace(changed[i],status='oom',completed_updates=0,validation=(),validation_exposure_sha256=None,comparison_update=None,failure_reason='synthetic memory refusal')
    r=replay_primary_selection(c,changed)
    assert 'hp-04'in r['ineligible_hyperparameter_points']
    assert r['frozen_transfer_hyperparameters']['hp_id']!='hp-04'


@pytest.mark.parametrize('kind',('test','denominator','exposure','duplicate-artifact','partial-campaign'))
def test_incomparable_evidence_cannot_select_transfer_settings(kind):
    c,rows=fixture();first=rows[0];v=first.validation
    if kind=='test':rows[0]=replace(first,validation=(replace(v[0],split='test'),v[1]))
    elif kind=='denominator':rows[0]=replace(first,validation=(replace(v[0],observed_targets=63),v[1]))
    elif kind=='exposure':rows[0]=replace(first,validation_exposure_sha256='e'*64)
    elif kind=='duplicate-artifact':rows[-1]=replace(rows[-1],run_artifact_sha256=rows[0].run_artifact_sha256)
    else:rows.pop()
    with pytest.raises(ValueError):replay_primary_selection(c,rows)
