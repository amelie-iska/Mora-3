"""Immutable controller pin prefixes around the existing one-trial worker.

The dedicated external store is trusted controller state, not cryptographic
authentication against an actor who can rewrite every file. Hash chaining,
canonical encoding and full receipt readback detect inconsistent/corrupt state.
No directory-discovered receipt is ever adopted. Publication before durable pin
storage leaves an orphan and blocks restart pending explicit reconciliation.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from mora3.data.admet import canonical
from .metric_records import emit_wandb_offline
from .mup_campaign_publication import _sync_directory as _sync
from .mup_campaign_recovery import ReceiptPin,recover_terminal_prefix
from .mup_campaign_worker import execute_one_primary_proxy_trial
from .mup_sweep_selection import FrozenProxySelection,planned_proxy_trials,select_proxy_transfer
from .mup_trial_artifacts import load_proxy_trial_artifact


@dataclass(frozen=True)
class PinnedCampaignPrefix:
    pins: tuple[ReceiptPin,...]
    tip_sha256: str


def _publish_new(path,raw):
    pending=path.with_name(path.name+'.pending')
    with pending.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    _sync(path.parent)
    os.link(pending,path)
    _sync(path.parent)
    pending.unlink()  # only this successfully published temporary
    _sync(path.parent)


def _paths(directory,campaign_directory,selection):
    if type(selection) is not FrozenProxySelection:
        raise ValueError('Exact frozen selection required')
    selection.__post_init__()
    store=Path(directory).absolute();campaign=Path(campaign_directory).absolute()
    if (any(p.is_symlink() for p in (store,*store.parents,campaign,*campaign.parents))
            or store.resolve().is_relative_to(campaign.resolve())
            or campaign.resolve().is_relative_to(store.resolve())):
        raise ValueError('Separate nonsymlink controller and campaign directories required')
    return store,campaign


def _genesis(campaign,selection):
    return canonical(dict(schema='Mora3.ProxyCampaignPinStore/v1',
        campaign_manifest_sha256=selection.campaign_manifest_sha256,
        campaign_directory=str(campaign.resolve())))


def _pin_wire(pins):
    return [dict(trial_id=p.trial_id,receipt_sha256=p.receipt_sha256) for p in pins]


def _validate_pin(pin,index):
    grid=planned_proxy_trials()
    if (type(pin) is not ReceiptPin or index>=len(grid) or pin.trial_id!=grid[index].trial_id
            or type(pin.receipt_sha256) is not str
            or re.fullmatch('[0-9a-f]{64}',pin.receipt_sha256) is None):
        raise ValueError('Exact next ordered receipt pin required')


def _load_store(store,campaign,selection):
    if not store.is_dir(): raise ValueError('Existing initialized controller store required')
    entries=tuple(store.iterdir())
    count=len(entries)-1
    if (not 0<=count<=len(planned_proxy_trials())
            or {p.name for p in entries}!={'genesis.json',*(f'prefix-{i:04d}.json' for i in range(1,count+1))}
            or any(p.is_symlink() or not p.is_file() for p in entries)):
        raise ValueError('Incomplete/unknown controller entry; explicit reconciliation required')
    raw=(store/'genesis.json').read_bytes()
    if raw!=_genesis(campaign,selection): raise ValueError('Exact campaign-bound store genesis required')
    tip=sha256(raw).hexdigest();pins=()
    for index in range(1,count+1):
        raw=(store/f'prefix-{index:04d}.json').read_bytes()
        try:
            document=json.loads(raw)
            candidate=tuple(ReceiptPin(**row) for row in document['pins'])
        except (KeyError,TypeError,UnicodeError) as error:
            raise ValueError('Malformed controller pin prefix') from error
        if len(candidate)!=index or candidate[:-1]!=pins:
            raise ValueError('Immutable exact controller prefix differs')
        _validate_pin(candidate[-1],index-1)
        expected=canonical(dict(schema='Mora3.ProxyCampaignPinPrefix/v1',
            campaign_manifest_sha256=selection.campaign_manifest_sha256,
            previous_sha256=tip,pins=_pin_wire(candidate)))
        if raw!=expected: raise ValueError('Canonical hash-chained controller prefix differs')
        pins=candidate;tip=sha256(raw).hexdigest()
    return PinnedCampaignPrefix(pins,tip)


def create_pin_store(directory,campaign_directory,*,selection):
    """Explicit setup only: create a fresh store for an empty campaign directory."""
    store,campaign=_paths(directory,campaign_directory,selection)
    recover_terminal_prefix(campaign,selection=selection,pins=())
    store.mkdir()
    _sync(store.parent)
    _publish_new(store/'genesis.json',_genesis(campaign,selection))
    return load_pinned_campaign(store,campaign,selection=selection)


def load_pinned_campaign(directory,campaign_directory,*,selection):
    """Verify stored pins and every artifact; orphan receipts never become pins."""
    store,campaign=_paths(directory,campaign_directory,selection)
    state=_load_store(store,campaign,selection)
    recover_terminal_prefix(campaign,selection=selection,pins=state.pins)
    return state


def commit_returned_pin(directory,campaign_directory,*,selection,previous,pin):
    """Persist only the worker's explicitly returned pin against its exact prior state.

    This is not an orphan-adoption API: normal restart calls load_pinned_campaign,
    which refuses unpublished/unpinned attempts. Never reconstruct a returned pin
    from directory discovery after interruption. Existing worker owns sequencing.
    """
    store,campaign=_paths(directory,campaign_directory,selection)
    current=_load_store(store,campaign,selection)
    if type(previous) is not PinnedCampaignPrefix or previous!=current:
        raise ValueError('Controller prefix changed before returned-pin publication')
    _validate_pin(pin,len(previous.pins))
    candidate=(*previous.pins,pin)
    recover_terminal_prefix(campaign,selection=selection,pins=candidate)
    raw=canonical(dict(schema='Mora3.ProxyCampaignPinPrefix/v1',
        campaign_manifest_sha256=selection.campaign_manifest_sha256,
        previous_sha256=previous.tip_sha256,pins=_pin_wire(candidate)))
    _publish_new(store/f'prefix-{len(candidate):04d}.json',raw)
    return load_pinned_campaign(store,campaign,selection=selection)


def execute_one_pinned_trial(directory,campaign_directory,*,campaign,trial_id,**worker_kwargs):
    """Read exact trusted pins, invoke existing worker once, durably commit its pin.

    No retry, schedule, ownership acquisition, model construction or launch
    authorization. An exception after worker return deliberately leaves an orphan
    or partial controller file; subsequent invocation refuses before worker entry.
    """
    if 'pins' in worker_kwargs: raise ValueError('Pins must come from the dedicated controller store')
    selection=campaign.selection_contract()
    previous=load_pinned_campaign(directory,campaign_directory,selection=selection)
    pin=execute_one_primary_proxy_trial(campaign_directory,campaign=campaign,trial_id=trial_id,
        pins=previous.pins,**worker_kwargs)
    return commit_returned_pin(directory,campaign_directory,selection=selection,previous=previous,pin=pin)


def publish_pinned_transfer_selection(directory,campaign_directory,output,*,selection):
    """Publish one immutable selection from the exact complete stored trial prefix."""
    store,campaign=_paths(directory,campaign_directory,selection)
    path=Path(output).absolute()
    if (path.exists() or any(p.is_symlink() for p in (path,*path.parents))
            or path.resolve().is_relative_to(store.resolve())
            or path.resolve().is_relative_to(campaign.resolve())):
        raise ValueError('New selection path outside the controller and campaign directories required')
    state=_load_store(store,campaign,selection)
    prefix=recover_terminal_prefix(campaign,selection=selection,pins=state.pins)
    result=select_proxy_transfer(selection,prefix.outcomes)
    raw=canonical(dict(schema='Mora3.PinnedProxyTransferSelection/v1',
        terminal_pin_prefix_sha256=state.tip_sha256,selection=result))
    _publish_new(path,raw)
    if path.read_bytes()!=raw:
        raise ValueError('Published selection bytes differ from the frozen terminal prefix')
    return sha256(raw).hexdigest()


def emit_pinned_trial_wandb(directory,campaign_directory,*,selection,trial_id,run):
    """Replay the exact committed terminal history into a fresh caller-owned offline run."""
    store,campaign=_paths(directory,campaign_directory,selection)
    state=_load_store(store,campaign,selection)
    prefix=recover_terminal_prefix(campaign,selection=selection,pins=state.pins)
    outcome=next((row for row in prefix.outcomes if row.trial_id==trial_id),None)
    if outcome is None:
        raise ValueError('Exact durably pinned terminal trial required for reporting')
    artifact=campaign/outcome.trial_id/'attempt-001'
    loaded,history=load_proxy_trial_artifact(artifact,outcome.run_artifact_sha256,selection=selection)
    if loaded!=outcome:
        raise ValueError('Pinned terminal outcome changed before reporting')
    return emit_wandb_offline(history,run)
