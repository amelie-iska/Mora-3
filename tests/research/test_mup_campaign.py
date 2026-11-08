from dataclasses import replace
from decimal import Decimal

import pytest
import torch

from mora3.research.mup_campaign import (
    FrozenPrimaryProxyCampaign, NativeTrainRoutePlan, ValidationBatchReduction,
    reduce_matched_validation, run_primary_proxy_trial,
)
from mora3.research.mup_sweep_selection import planned_proxy_trials
from test_full_owner import _construct, actual_core_assets, cases
from test_primary_native_routes import fixture
from test_matched_validation import cohort


def plan(view):
    validation=cohort(view)
    native=(
        NativeTrainRoutePlan('fit-objective','a'*64,'b'*64,8),
        NativeTrainRoutePlan('train-only-objective','c'*64,'d'*64,4),
    )
    return FrozenPrimaryProxyCampaign(
        source_snapshot_sha256='a'*64,width_family_sha256='e'*64,proxy_width=36,proxy_parameter_count=9750884,
        source_role_schedule_sha256='1'*64,complete_owner_evidence_sha256='2'*64,
        runtime_profile_sha256='3'*64,existing_worker_sha256='4'*64,
        validation=validation,validation_adapter_sha256=(('fit-objective','5'*64),),
        train_routes=native,task_cycle=('fit-objective','train-only-objective','fit-objective'),
        gradient_clip_max_norm=1.,decay_aliases=('x.weight','y.weight'),
        max_one_update_seconds=Decimal('2'),max_matched_validation_seconds=Decimal('20'),
    )


def test_manifest_retains_exact_grid_finite_budgets_and_role_selection(tmp_path):
    p=plan(fixture(tmp_path));first,second,*_=planned_proxy_trials()
    assert p.selection_contract().validation_exposure_sha256==p.validation.sha256
    assert p.selection_contract().campaign_manifest_sha256==p.sha256
    assert p.require_next_trial(first.trial_id,())==first
    assert p.require_next_trial(second.trial_id,(first.trial_id,))==second
    with pytest.raises(ValueError):p.require_next_trial(second.trial_id,())
    with pytest.raises(ValueError):p.require_next_trial(first.trial_id,(first.trial_id,))
    assert len(p.binding()['exact_trial_grid'])==24
    assert p.binding()['per_trial_update_limit']==5000
    assert p.binding()['per_trial_wall_seconds']=='1800'


@pytest.mark.parametrize('drift',['unknown-cycle','lost-validation-disposition','decay-alias','invalid-width','budget'])
def test_incomplete_mixture_owner_or_resource_scope_refuses(tmp_path,drift):
    p=plan(fixture(tmp_path))
    changes={
        'unknown-cycle':{'task_cycle':('other',)},
        'lost-validation-disposition':{'validation':replace(p.validation,train_only_dispositions=())},
        'decay-alias':{'decay_aliases':('x.weight','x.weight')},
        'invalid-width':{'proxy_width':True},
        'budget':{'max_one_update_seconds':Decimal('1795')},
    }
    with pytest.raises(ValueError):replace(p,**changes[drift])


def test_native_reduction_uses_total_observed_denominator_and_missing_receipt(tmp_path):
    p=plan(fixture(tmp_path));a=p.validation.tasks[0]
    rows=(ValidationBatchReduction(Decimal('0.3'),1,None),
          ValidationBatchReduction(Decimal('0'),0,'missing native target'),
          ValidationBatchReduction(Decimal('0.9'),2,None))
    measured=reduce_matched_validation(a,rows)
    assert measured.observed_targets==3 and measured.mean_loss==Decimal('0.4') and measured.split=='validation'
    with pytest.raises(ValueError):reduce_matched_validation(a,rows[:-1])
    with pytest.raises(ValueError):ValidationBatchReduction(Decimal('0'),0,None)


def test_proxy_entry_calls_real_admission_before_source_validation_or_optimizer(tmp_path,actual_core_assets):
    from mora3.research.primary_training import NativeMuUpdateEngine
    model=_construct(tmp_path,actual_core_assets)
    engine=object.__new__(NativeMuUpdateEngine);engine.model=model
    before=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='admission'):
        run_primary_proxy_trial(engine,None,trial_id='unread',terminal_trial_ids=(),
                                bindings=None,validation_view=None,validation_evaluators=None,
                                metric_identity=None,root=None)
    assert torch.equal(before,torch.get_rng_state())


def test_exact_campaign_roundtrip_and_complete_validation_metric_export(tmp_path):
    from mora3.research.mup_campaign import ValidationTaskExecution, _validation_metrics
    from mora3.research.primary_native_routes import NativeMetricIdentity
    from mora3.research.metric_records import write_history, read_history

    p=plan(fixture(tmp_path))
    frozen=tmp_path/'campaign.json';frozen.write_bytes(p.wire())
    assert FrozenPrimaryProxyCampaign.load(frozen,p.sha256,validation=p.validation)==p
    with pytest.raises(ValueError):
        FrozenPrimaryProxyCampaign.load(frozen,p.sha256,validation=replace(p.validation,comparison_update=1))
    task=p.validation.tasks[0]
    reductions=(ValidationBatchReduction(Decimal('1.2'),3,None),)
    detail=ValidationTaskExecution(reduce_matched_validation(task,reductions),1,reductions)
    events=_validation_metrics(p.validation,(detail,),NativeMetricIdentity('run','trial','seg',5))
    assert all(r.scope=='validation' and r.step==2500 for r in events)
    assert events[0].value==.4 and events[0].numerator==1.2 and events[0].denominator==3
    out=tmp_path/'history.jsonl'
    assert write_history(out,events)==3 and read_history(out)==events


def test_new_campaign_binds_family_axis_and_legacy_manifest_roundtrips(tmp_path):
    from mora3.data.admet import canonical, digest

    p = plan(fixture(tmp_path))
    axis = tuple((task.task_key,task.source_family) for task in p.validation.tasks)
    assert p.selection_contract().task_families==axis
    assert p.binding()['schema']=='Mora3.FrozenPrimaryProxyCampaign/v2'
    assert p.binding()['validation_aggregation']==dict(policy='source-family-macro-v1',task_families=axis)
    wrong = p.binding()
    wrong['validation_aggregation']['task_families']=((axis[0][0],'forged'),)
    damaged = tmp_path/'changed-family-axis.json'
    raw = canonical(wrong)
    damaged.write_bytes(raw)
    with pytest.raises(ValueError,match='exact canonical'):
        FrozenPrimaryProxyCampaign.load(damaged,digest(raw),validation=p.validation)
    old = replace(p,validation_aggregation='task-macro-v1')
    assert old.selection_contract().task_families==()
    assert 'validation_aggregation' not in old.binding()
    assert old.binding()['schema']=='Mora3.FrozenPrimaryProxyCampaign/v1'
    assert old.sha256!=p.sha256
    original = tmp_path/'legacy-campaign.json'
    original.write_bytes(old.wire())
    recovered = FrozenPrimaryProxyCampaign.load(original,old.sha256,validation=p.validation)
    assert recovered==old and recovered.wire()==old.wire()
