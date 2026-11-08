from dataclasses import replace
from hashlib import sha256
import json

import pytest

from mora3.data.admet import canonical
from mora3.research.mup_sweep_selection import planned_proxy_trials
from mora3.research.mup_trial_artifacts import seal_proxy_trial_execution
from test_mup_trial_artifacts import execution
from mora3.research.mup_campaign_recovery import ReceiptPin, recover_terminal_prefix


def committed(root, contract, run):
    trial = root/run.trial_id
    trial.mkdir()
    digest = seal_proxy_trial_execution(trial/'attempt-001',run,selection=contract)
    raw = canonical(dict(schema='Mora3.ProxyTrialControllerReceipt/v1',
        campaign_manifest_sha256=contract.campaign_manifest_sha256,
        trial_id=run.trial_id,artifact_directory=f'{run.trial_id}/attempt-001',
        artifact_sha256=digest))
    (trial/'receipt.json').write_bytes(raw)
    return ReceiptPin(run.trial_id,sha256(raw).hexdigest())


def test_verified_prefix_and_read_only_next_path(tmp_path):
    contract,run=execution()
    empty=recover_terminal_prefix(tmp_path,selection=contract,pins=())
    assert empty.next_attempt_directory==tmp_path/run.trial_id/'attempt-001'
    assert list(tmp_path.iterdir())==[]
    pin=committed(tmp_path,contract,run)
    state=recover_terminal_prefix(tmp_path,selection=contract,pins=(pin,))
    assert state.terminal_trial_ids==(run.trial_id,)
    assert state.outcomes[0].trial_id==run.trial_id
    assert state.next_attempt_directory==tmp_path/planned_proxy_trials()[1].trial_id/'attempt-001'
    assert not state.next_attempt_directory.parent.exists()


@pytest.mark.parametrize('case',['orphan','partial','unknown','missing','digest','campaign','reordered','duplicate','metrics','symlink'])
def test_ambiguous_or_tampered_state_cannot_advance(tmp_path,case):
    contract,run=execution()
    pin=committed(tmp_path,contract,run)
    pins=(pin,)
    if case=='orphan': pins=()
    elif case=='partial': (tmp_path/planned_proxy_trials()[1].trial_id).mkdir()
    elif case=='unknown': (tmp_path/'unexpected').mkdir()
    elif case=='missing': pins=(pin,ReceiptPin(planned_proxy_trials()[1].trial_id,'a'*64))
    elif case=='digest': pins=(replace(pin,receipt_sha256='f'*64),)
    elif case=='campaign': contract=replace(contract,campaign_manifest_sha256='f'*64)
    elif case=='reordered': pins=(replace(pin,trial_id=planned_proxy_trials()[1].trial_id),)
    elif case=='duplicate': pins=(pin,pin)
    elif case=='metrics':
        with (tmp_path/run.trial_id/'attempt-001'/'metrics.jsonl').open('ab') as stream: stream.write(b'\n')
    elif case=='symlink':
        (tmp_path/'unexpected').symlink_to(tmp_path/run.trial_id,target_is_directory=True)
    with pytest.raises(ValueError): recover_terminal_prefix(tmp_path,selection=contract,pins=pins)


def test_complete_prefix_has_no_next_attempt(tmp_path):
    contract,run=execution()
    pins=[]
    for trial in planned_proxy_trials():
        changed=replace(run,trial_id=trial.trial_id,metrics=tuple(
            replace(row,trial_id=trial.trial_id,seed=trial.seed) for row in run.metrics))
        pins.append(committed(tmp_path,contract,changed))
    state=recover_terminal_prefix(tmp_path,selection=contract,pins=tuple(pins))
    assert len(state.outcomes)==24 and state.next_attempt_directory is None
