import json
from hashlib import sha256
from dataclasses import replace

import pytest

from test_metric_records import OfflineRun
from test_mup_trial_artifacts import execution
from mora3.research.metric_records import emit_wandb_offline
from mora3.research.mup_campaign_publication import reserve_next_attempt,publish_terminal_receipt
from mora3.research.mup_trial_artifacts import seal_proxy_trial_execution
from mora3.research import mup_campaign_pin_store as store
from mora3.research.mup_sweep_selection import planned_proxy_trials


@pytest.fixture
def setup(tmp_path):
    selection,run=execution()
    campaign=tmp_path/'campaign';campaign.mkdir()
    pins=tmp_path/'pins'
    store.create_pin_store(pins,campaign,selection=selection)
    return pins,campaign,selection,run


def terminal(campaign,selection,run):
    reservation=reserve_next_attempt(campaign,selection=selection,pins=())
    digest=seal_proxy_trial_execution(reservation.artifact_directory,run,selection=selection)
    return publish_terminal_receipt(reservation,selection=selection,artifact_sha256=digest)


def test_store_verifies_immutable_full_prefix(setup):
    pins,campaign,selection,run=setup
    before=store.load_pinned_campaign(pins,campaign,selection=selection)
    assert before.pins==()
    pin=terminal(campaign,selection,run)
    after=store.commit_returned_pin(pins,campaign,selection=selection,previous=before,pin=pin)
    assert after.pins==(pin,) and after==store.load_pinned_campaign(pins,campaign,selection=selection)
    original=(pins/'prefix-0001.json').read_bytes()
    with pytest.raises(ValueError): store.commit_returned_pin(pins,campaign,selection=selection,previous=before,pin=pin)
    assert (pins/'prefix-0001.json').read_bytes()==original


def test_published_receipt_without_pin_blocks_restart(setup):
    pins,campaign,selection,run=setup
    terminal(campaign,selection,run)
    with pytest.raises(ValueError): store.load_pinned_campaign(pins,campaign,selection=selection)


def test_interrupted_pin_publication_is_preserved(setup,monkeypatch):
    pins,campaign,selection,run=setup
    before=store.load_pinned_campaign(pins,campaign,selection=selection)
    pin=terminal(campaign,selection,run)
    def fail(*args): raise OSError('simulated pin publication interruption')
    monkeypatch.setattr(store.os,'link',fail)
    with pytest.raises(OSError): store.commit_returned_pin(pins,campaign,selection=selection,previous=before,pin=pin)
    assert (pins/'prefix-0001.json.pending').is_file()
    with pytest.raises(ValueError): store.load_pinned_campaign(pins,campaign,selection=selection)


@pytest.mark.parametrize('case',['missing-tip','unknown','modified','wrong-campaign'])
def test_ledger_tampering_or_ambiguity_blocks(setup,case,tmp_path):
    pins,campaign,selection,run=setup
    before=store.load_pinned_campaign(pins,campaign,selection=selection)
    pin=terminal(campaign,selection,run)
    store.commit_returned_pin(pins,campaign,selection=selection,previous=before,pin=pin)
    path=pins/'prefix-0001.json'
    if case=='missing-tip': path.rename(tmp_path/'preserved-tip.json')
    elif case=='unknown': (pins/'unexpected').write_bytes(b'')
    elif case=='modified':
        with path.open('ab') as stream: stream.write(b'\n')
    elif case=='wrong-campaign': campaign=tmp_path/'other';campaign.mkdir()
    with pytest.raises(ValueError): store.load_pinned_campaign(pins,campaign,selection=selection)


def test_owner_wrapper_commits_only_returned_pin(setup,monkeypatch):
    pins,campaign,selection,run=setup
    class Campaign:
        def selection_contract(self): return selection
    calls=[]
    def worker(directory,**kwargs):
        calls.append(kwargs['pins'])
        return terminal(directory,selection,run)
    monkeypatch.setattr(store,'execute_one_primary_proxy_trial',worker)
    state=store.execute_one_pinned_trial(pins,campaign,campaign=Campaign(),trial_id=run.trial_id)
    assert len(state.pins)==1 and calls==[()]
    assert store.load_pinned_campaign(pins,campaign,selection=selection)==state


def test_owner_never_calls_worker_for_orphan(setup,monkeypatch):
    pins,campaign,selection,run=setup
    terminal(campaign,selection,run)
    class Campaign:
        def selection_contract(self): return selection
    def forbidden(*args,**kwargs): pytest.fail('must not invoke worker')
    monkeypatch.setattr(store,'execute_one_primary_proxy_trial',forbidden)
    with pytest.raises(ValueError): store.execute_one_pinned_trial(pins,campaign,campaign=Campaign(),trial_id=run.trial_id)


def test_full_24_trial_prefix_remains_exact(setup):
    pins,campaign,selection,run=setup
    state=store.load_pinned_campaign(pins,campaign,selection=selection)
    for trial in planned_proxy_trials():
        changed=replace(run,trial_id=trial.trial_id,metrics=tuple(
            replace(row,trial_id=trial.trial_id,seed=trial.seed) for row in run.metrics))
        reservation=reserve_next_attempt(campaign,selection=selection,pins=state.pins)
        digest=seal_proxy_trial_execution(reservation.artifact_directory,changed,selection=selection)
        pin=publish_terminal_receipt(reservation,selection=selection,artifact_sha256=digest)
        state=store.commit_returned_pin(pins,campaign,selection=selection,previous=state,pin=pin)
    assert len(state.pins)==24
    assert state==store.load_pinned_campaign(pins,campaign,selection=selection)
    assert len(tuple(pins.iterdir()))==25
    path=pins.parent/'transfer.json'
    digest=store.publish_pinned_transfer_selection(pins,campaign,path,selection=selection)
    raw=path.read_bytes()
    assert sha256(raw).hexdigest()==digest
    published=json.loads(raw)
    assert published['terminal_pin_prefix_sha256']==state.tip_sha256
    result=published['selection']
    assert result['full_terminal_trial_count']==24 and result['eligible_hyperparameter_points']==12
    assert result['launch_authorized'] is False and result['frozen_transfer_hyperparameters'] is not None
    with pytest.raises(ValueError): store.publish_pinned_transfer_selection(pins,campaign,path,selection=selection)
    assert path.read_bytes()==raw


def test_post_link_interruption_retains_pending_blocker(setup,monkeypatch):
    pins,campaign,selection,run=setup
    state=store.load_pinned_campaign(pins,campaign,selection=selection)
    pin=terminal(campaign,selection,run)
    original=store._sync
    def fail(path):
        if (pins/'prefix-0001.json').exists(): raise OSError('simulated post-link interruption')
        original(path)
    monkeypatch.setattr(store,'_sync',fail)
    with pytest.raises(OSError): store.commit_returned_pin(pins,campaign,selection=selection,previous=state,pin=pin)
    assert (pins/'prefix-0001.json').exists() and (pins/'prefix-0001.json.pending').exists()
    with pytest.raises(ValueError): store.load_pinned_campaign(pins,campaign,selection=selection)


def test_incomplete_or_orphan_prefix_cannot_publish_selection(setup):
    pins,campaign,selection,run=setup
    path=pins.parent/'transfer.json'
    with pytest.raises(ValueError,match='24'):
        store.publish_pinned_transfer_selection(pins,campaign,path,selection=selection)
    terminal(campaign,selection,run)
    with pytest.raises(ValueError,match='orphan'):
        store.publish_pinned_transfer_selection(pins,campaign,path,selection=selection)
    assert not path.exists() and not path.with_name(path.name+'.pending').exists()


def test_pinned_trial_wandb_requires_exact_committed_history(setup):
    pins,campaign,selection,run=setup
    output=OfflineRun()
    prior=store.load_pinned_campaign(pins,campaign,selection=selection)
    with pytest.raises(ValueError,match='durably pinned'):
        store.emit_pinned_trial_wandb(pins,campaign,selection=selection,trial_id=run.trial_id,run=output)
    pin=terminal(campaign,selection,run)
    with pytest.raises(ValueError,match='orphan'):
        store.emit_pinned_trial_wandb(pins,campaign,selection=selection,trial_id=run.trial_id,run=output)
    assert not output.logs
    store.commit_returned_pin(pins,campaign,selection=selection,previous=prior,pin=pin)
    expected=OfflineRun()
    emit_wandb_offline(run.metrics,expected)
    count=store.emit_pinned_trial_wandb(pins,campaign,selection=selection,trial_id=run.trial_id,run=output)
    assert count==len(run.metrics) and output.logs==expected.logs
    online=OfflineRun('online')
    with pytest.raises(ValueError,match='offline'):
        store.emit_pinned_trial_wandb(pins,campaign,selection=selection,trial_id=run.trial_id,run=online)
    assert not online.logs
    history=campaign/run.trial_id/'attempt-001'/'metrics.jsonl'
    with history.open('ab') as f:f.write(b'\n')
    damaged=OfflineRun()
    with pytest.raises(ValueError):
        store.emit_pinned_trial_wandb(pins,campaign,selection=selection,trial_id=run.trial_id,run=damaged)
    assert not damaged.logs
