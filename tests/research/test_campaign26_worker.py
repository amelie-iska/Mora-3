"""Counted worker tests use metadata fixtures, never real optimizer execution."""
from pathlib import Path
from hashlib import sha256
import json
import pytest
from mora3.research.campaign26 import Campaign26, RunBudget26
from mora3.research import campaign26_worker as worker
from mora3.research.mup_campaign_pin_store import create_pin_store,load_pinned_campaign,commit_returned_pin
from mora3.research.mup_campaign_publication import reserve_next_attempt,publish_terminal_receipt
from mora3.research.mup_trial_artifacts import seal_proxy_trial_execution
from test_mup_trial_artifacts import execution


@pytest.fixture
def harness(tmp_path,monkeypatch):
    selection,result=execution();events=[]
    from mora3.research.campaign26_transfer import TargetBlueprint
    from mora3.model.full_owner import PrimaryConfiguration
    blueprint=TargetBlueprint(PrimaryConfiguration(266,8,'b'*64),80304407,
        'd'*64,'e'*64,'f'*64,20000,86400,('native-task',),1.0,())
    class Model:
        def require_training_admission(self):events.append('admission')
    class Engine:model=Model()
    class Campaign:
        sha256=selection.campaign_manifest_sha256
        source_snapshot_sha256='b'*64
        proxy_width=8
        width_family_sha256='e'*64
        task_cycle=('native-task',)
        gradient_clip_max_norm=1.0
        decay_aliases=()
        def selection_contract(self):return selection
    monkeypatch.setattr(worker,'NativeMuUpdateEngine',Engine)
    monkeypatch.setattr(worker,'SourceBoundOwner',Model)
    monkeypatch.setattr(worker,'FrozenPrimaryProxyCampaign',Campaign)
    monkeypatch.setattr(worker,'require_current_directive',lambda *a:None)
    campaign=tmp_path/'campaign';campaign.mkdir();pins=tmp_path/'pins'
    create_pin_store(pins,campaign,selection=selection)
    budget=RunBudget26.create(tmp_path/'budget',Campaign26('c'*64,'b'*64,selection.campaign_manifest_sha256,
        target_specification_sha256=blueprint.sha256))
    def execute(directory,campaign_directory,**kwargs):
        events.append('execute')
        prior=load_pinned_campaign(directory,campaign_directory,selection=selection)
        r=reserve_next_attempt(campaign_directory,selection=selection,pins=prior.pins)
        digest=seal_proxy_trial_execution(r.artifact_directory,result,selection=selection)
        pin=publish_terminal_receipt(r,selection=selection,artifact_sha256=digest)
        return commit_returned_pin(directory,campaign_directory,selection=selection,previous=prior,pin=pin)
    monkeypatch.setattr(worker,'execute_one_pinned_trial',execute)
    kwargs=dict(budget=budget,pin_directory=pins,campaign_directory=campaign,campaign=Campaign(),
        trial_id=result.trial_id,engine=Engine(),root=tmp_path,failure_artifact=tmp_path/'failure.json',
        target_blueprint=blueprint)
    return kwargs,events


def test_one_returned_native_result_consumes_exactly_one_slot(harness):
    kw,events=harness;worker.execute_counted_proxy_trial(**kw)
    assert events==['admission','execute']
    assert kw['budget'].snapshot()['terminal_runs']==1
    with pytest.raises(ValueError):worker.execute_counted_proxy_trial(**kw)
    assert events.count('execute')==1


def test_actual_owner_refusal_precedes_any_budget_claim(harness,monkeypatch):
    kw,events=harness
    def refuse():raise ValueError('no complete source admission')
    monkeypatch.setattr(kw['engine'].model,'require_training_admission',refuse)
    with pytest.raises(ValueError,match='no complete'):worker.execute_counted_proxy_trial(**kw)
    assert kw['budget'].snapshot()['reserved_runs']==0
    assert not kw['failure_artifact'].exists()


def test_worker_exception_is_retained_and_does_not_refund_slot(harness,monkeypatch):
    kw,events=harness
    def fail(*a,**k):raise RuntimeError('sensitive exception detail must not be recorded')
    monkeypatch.setattr(worker,'execute_one_pinned_trial',fail)
    with pytest.raises(RuntimeError):worker.execute_counted_proxy_trial(**kw)
    assert kw['budget'].snapshot()['reserved_runs']==1
    assert kw['budget'].snapshot()['failed_runs']==1
    assert 'sensitive'not in kw['failure_artifact'].read_text()
    with pytest.raises(ValueError):worker.execute_counted_proxy_trial(**kw)


def test_missing_or_changed_current_directive_refuses(tmp_path):
    from mora3.data.admet import canonical
    p=tmp_path/'planning';p.mkdir()
    doc={'schema':'Mora3.Campaign26UserDirective/v1','maximum_training_runs':26,'proxy_sweep_runs':24,
        'full_target_training_runs':2,'target_seeds':[2026111001,2026111002],
        'standard_parameterization_baseline_runs':0,'intermediate_width_training_runs':0,
        'naive_standard_parameterization_target_runs':0,
        'primary_grid':dict(master_learning_rates=['0.0001','0.0003','0.001'],
            readout_multipliers=['0.5','1'],attention_key_multipliers=['0.5','1'],seeds=[2026110201,2026110202])}
    raw=canonical(doc);(p/'policy.json').write_bytes(raw)
    pointer={'schema':'Mora3.CurrentCampaignDirective/v1','path':'planning/policy.json','sha256':sha256(raw).hexdigest()}
    (p/'current_mup_campaign.json').write_text(json.dumps(pointer))
    plan=Campaign26(pointer['sha256'],'b'*64,'c'*64)
    worker.require_current_directive(tmp_path,plan)
    doc['maximum_training_runs']=44;(p/'policy.json').write_bytes(canonical(doc))
    with pytest.raises(ValueError):worker.require_current_directive(tmp_path,plan)


def test_unfrozen_target_blueprint_refuses_before_counted_worker_or_slot(harness):
    kw,events=harness
    with pytest.raises(ValueError,match='[Tt]arget'):
        worker.execute_counted_proxy_trial(**(kw|{'target_blueprint':None}))
    assert 'execute'not in events
    assert kw['budget'].snapshot()['reserved_runs']==0
