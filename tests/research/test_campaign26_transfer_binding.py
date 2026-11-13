"""Synthetic terminal histories exercise the REAL selector/accounting code.

These tests do not train a model or establish actual campaign completion.
"""
from dataclasses import replace
from hashlib import sha256
import json
import pytest
from mora3.data.admet import canonical
from mora3.research.campaign26 import Campaign26,RunBudget26,target_run_specs
from mora3.research.mup_sweep_selection import planned_proxy_trials
from mora3.research.mup_campaign_pin_store import (create_pin_store,load_pinned_campaign,
    commit_returned_pin,publish_pinned_transfer_selection)
from mora3.research.mup_campaign_publication import reserve_next_attempt,publish_terminal_receipt
from mora3.research.mup_campaign_recovery import recover_terminal_prefix
from mora3.research.mup_trial_artifacts import seal_proxy_trial_execution
from mora3.research.campaign26_worker import require_current_directive
from test_mup_trial_artifacts import execution


def synthetic_campaign(tmp_path,*,count=24,target_specification_sha256=None):
    selection,example=execution()
    planning=tmp_path/'planning';planning.mkdir()
    directive=dict(schema='Mora3.Campaign26UserDirective/v1',maximum_training_runs=26,
        proxy_sweep_runs=24,full_target_training_runs=2,target_seeds=[2026111001,2026111002],
        standard_parameterization_baseline_runs=0,intermediate_width_training_runs=0,
        naive_standard_parameterization_target_runs=0,
        primary_grid=dict(master_learning_rates=['0.0001','0.0003','0.001'],
            readout_multipliers=['0.5','1'],attention_key_multipliers=['0.5','1'],
            seeds=[2026110201,2026110202]))
    raw=canonical(directive);(planning/'directive.json').write_bytes(raw)
    (planning/'current_mup_campaign.json').write_bytes(canonical(dict(
        schema='Mora3.CurrentCampaignDirective/v1',path='planning/directive.json',sha256=sha256(raw).hexdigest())))
    options={}if target_specification_sha256 is None else dict(target_specification_sha256=target_specification_sha256)
    plan=Campaign26(sha256(raw).hexdigest(),'b'*64,selection.campaign_manifest_sha256,**options)
    budget=RunBudget26.create(tmp_path/'budget',plan)
    campaign=tmp_path/'campaign';campaign.mkdir();pins=tmp_path/'pins'
    state=create_pin_store(pins,campaign,selection=selection)
    for trial in planned_proxy_trials()[:count]:
        claim=budget.reserve(trial.trial_id);claim.enter()
        trial_result=replace(example,trial_id=trial.trial_id,metrics=tuple(replace(m,
            run_id='synthetic-fixture-'+trial.trial_id,trial_id=trial.trial_id,seed=trial.seed)for m in example.metrics))
        reservation=reserve_next_attempt(campaign,selection=selection,pins=state.pins)
        digest=seal_proxy_trial_execution(reservation.artifact_directory,trial_result,selection=selection)
        pin=publish_terminal_receipt(reservation,selection=selection,artifact_sha256=digest)
        state=commit_returned_pin(pins,campaign,selection=selection,previous=state,pin=pin)
        budget.finish(claim,status=trial_result.status,artifact_sha256=digest)
    publication=tmp_path/'selection.json'
    if count==24:
        digest=publish_pinned_transfer_selection(pins,campaign,publication,selection=selection)
        budget.bind_selection(digest)
    outcomes=recover_terminal_prefix(campaign,selection=selection,pins=state.pins).outcomes
    return dict(root=tmp_path,plan=plan,budget=budget,pin_directory=pins,campaign_directory=campaign,
        selection=selection,outcomes=outcomes,selection_path=publication,pins=state)


def target_arguments():
    return dict(source_snapshot_sha256='b'*64,target_width=266,target_parameter_count=80304407,
        complete_owner_sha256='d'*64,width_family_sha256='e'*64,runtime_profile_sha256='f'*64,
        maximum_updates=20000,maximum_seconds=86400)


def test_published_selection_and_target_specs_have_the_same_commitment(tmp_path):
    h=synthetic_campaign(tmp_path)
    specs=target_run_specs(h['plan'],h['selection'],h['outcomes'],selection_publication=h['selection_path'].read_bytes(),**target_arguments())
    assert {s['selection_sha256']for s in specs}=={h['budget'].snapshot()['selection_sha256']}


def test_directive_grid_must_match_the_native24_trials_not_only_counts(tmp_path):
    h=synthetic_campaign(tmp_path,count=0)
    doc=json.loads((tmp_path/'planning/directive.json').read_bytes())
    doc['primary_grid']['master_learning_rates']=['0.0001','0.0003','0.1']
    raw=canonical(doc);(tmp_path/'planning/directive.json').write_bytes(raw)
    pointer=json.loads((tmp_path/'planning/current_mup_campaign.json').read_bytes());pointer['sha256']=sha256(raw).hexdigest()
    (tmp_path/'planning/current_mup_campaign.json').write_bytes(canonical(pointer))
    changed=replace(h['plan'],directive_sha256=pointer['sha256'])
    with pytest.raises(ValueError,match='grid'):
        require_current_directive(tmp_path,changed)
