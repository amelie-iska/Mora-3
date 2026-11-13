"""Metadata-only 26-run limits; simulated receipts are not training results."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import json
import pytest
from mora3.research.campaign26 import Campaign26, RunBudget26, target_run_specs
from mora3.research.mup_sweep_selection import planned_proxy_trials, select_proxy_transfer
from mora3.data.admet import canonical
from test_mup_sweep_selection import fixture as selection_fixture


def plan():
    return Campaign26('a'*64,'b'*64,'c'*64)


def finish_proxy_budget(store):
    for trial in planned_proxy_trials():
        claim=store.reserve(trial.trial_id)
        claim.enter()
        store.finish(claim,status='completed',artifact_sha256='d'*64)


def test_fixed_24_plus2_no_baselines_and_no_implicit_widths():
    p=plan();assert len(p.run_ids)==26 and len(set(p.run_ids))==26
    assert p.run_ids[:24]==tuple(t.trial_id for t in planned_proxy_trials())
    assert p.run_ids[-2:]==('full80m-2026111001','full80m-2026111002')
    assert p.wire()['target_parameterization']=='muP'
    assert p.wire()['maximum_training_runs']==26
    assert p.wire()['standard_parameterization_runs']==0
    for bad in ((1,), (1,1), (True,2), (1,2,3)):
        with pytest.raises(ValueError):replace(p,target_seeds=bad)


def test_failed_start_consumes_slot_and_reuse_is_forbidden(tmp_path):
    store=RunBudget26.create(tmp_path/'budget',plan())
    claim=store.reserve(plan().run_ids[0]);claim.enter()
    store.finish(claim,status='failed',artifact_sha256='d'*64)
    with pytest.raises(ValueError):store.reserve(plan().run_ids[0])
    assert store.snapshot()['reserved_runs']==1
    assert store.snapshot()['failed_runs']==1
    assert store.reserve(plan().run_ids[1]).run_id==plan().run_ids[1]


def test_unfinished_attempt_blocks_later_runs_and_cannot_be_reentered(tmp_path):
    store=RunBudget26.create(tmp_path/'budget',plan());claim=store.reserve(plan().run_ids[0])
    with pytest.raises(ValueError):store.reserve(plan().run_ids[1])
    claim.enter()
    with pytest.raises(ValueError):claim.enter()
    assert store.snapshot()['reserved_runs']==1


def test_target_cannot_start_without_all_proxy_receipts_and_selection(tmp_path):
    store=RunBudget26.create(tmp_path/'budget',plan())
    with pytest.raises(ValueError):store.reserve(plan().run_ids[24])
    finish_proxy_budget(store)
    with pytest.raises(ValueError,match='selection'):store.reserve(plan().run_ids[24])
    assert store.snapshot()['reserved_runs']==24


def test_complete_budget_is_capped_at26_even_after_failure(tmp_path):
    store=RunBudget26.create(tmp_path/'budget',plan());finish_proxy_budget(store)
    store.bind_selection('e'*64)
    for rid in plan().run_ids[24:]:
        c=store.reserve(rid);c.enter();store.finish(c,status='failed',artifact_sha256='f'*64)
    assert store.snapshot()['reserved_runs']==26
    for rid in ('extra',plan().run_ids[0],plan().run_ids[-1]):
        with pytest.raises(ValueError):store.reserve(rid)
    assert len(list((tmp_path/'budget'/'runs').iterdir()))==26


def test_concurrent_reservation_allows_only_one_claim(tmp_path):
    store=RunBudget26.create(tmp_path/'budget',plan())
    def reserve():
        try:return store.reserve(plan().run_ids[0]).run_id
        except (ValueError,BlockingIOError):return None
    with ThreadPoolExecutor(max_workers=2)as pool:results=list(pool.map(lambda _:reserve(),range(2)))
    assert results.count(plan().run_ids[0])==1
    assert store.snapshot()['reserved_runs']==1


@pytest.mark.parametrize('damage',('genesis','unknown','torn','symlink'))
def test_inconsistent_or_symlink_budget_fails_closed(tmp_path,damage):
    root=tmp_path/'budget';store=RunBudget26.create(root,plan())
    if damage=='genesis':
        with(root/'genesis.json').open('ab')as f:f.write(b'\n')
    elif damage=='unknown':(root/'surprise').write_text('not a run')
    elif damage=='torn':(root/'runs'/plan().run_ids[0]).mkdir()
    else:
        (root/'runs').rename(tmp_path/'retained');(root/'runs').symlink_to(tmp_path/'retained',target_is_directory=True)
    with pytest.raises(ValueError):store.snapshot()


def test_selection_is_immutable_and_never_creates_a_run(tmp_path):
    store=RunBudget26.create(tmp_path/'budget',plan());finish_proxy_budget(store)
    store.bind_selection('e'*64)
    with pytest.raises((ValueError,FileExistsError)):store.bind_selection('f'*64)
    assert store.snapshot()['reserved_runs']==24
    assert store.snapshot()['selection_sha256']=='e'*64


def test_target_specs_preserve_winner_and_two_seeds_without_target_tuning():
    contract,outcomes=selection_fixture()
    p=Campaign26('a'*64,'b'*64,contract.campaign_manifest_sha256)
    specs=target_run_specs(p,contract,outcomes,source_snapshot_sha256='b'*64,
        target_width=266,target_parameter_count=80304407,complete_owner_sha256='d'*64,
        width_family_sha256='e'*64,runtime_profile_sha256='f'*64,
        maximum_updates=20000,maximum_seconds=86400,selection_publication=canonical(dict(
            schema='Mora3.PinnedProxyTransferSelection/v1',terminal_pin_prefix_sha256='a'*64,
            selection=select_proxy_transfer(contract,outcomes))))
    assert len(specs)==2 and {s['seed']for s in specs}=={2026111001,2026111002}
    assert all(s['parameterization']=='muP'and s['target_tuning_allowed']is False for s in specs)
    assert specs[0]['hyperparameters']==specs[1]['hyperparameters']
    assert specs[0]['hyperparameters']['hp_id']=='hp-01'
    assert all(s['launch_authorized']is False for s in specs)
    with pytest.raises(ValueError):target_run_specs(p,contract,outcomes[:-1],source_snapshot_sha256='b'*64,
        target_width=266,target_parameter_count=80304407,complete_owner_sha256='d'*64,
        width_family_sha256='e'*64,runtime_profile_sha256='f'*64,maximum_updates=20000,maximum_seconds=86400,selection_publication=canonical(dict(
            schema='Mora3.PinnedProxyTransferSelection/v1',terminal_pin_prefix_sha256='a'*64,
            selection=select_proxy_transfer(contract,outcomes))))


def test_mismatched_source_and_bad_target_budget_refuse():
    contract,outcomes=selection_fixture();p=Campaign26('a'*64,'b'*64,contract.campaign_manifest_sha256)
    common=dict(source_snapshot_sha256='b'*64,target_width=266,target_parameter_count=80304407,
        complete_owner_sha256='d'*64,width_family_sha256='e'*64,runtime_profile_sha256='f'*64,
        maximum_updates=20000,maximum_seconds=86400,selection_publication=canonical(dict(
            schema='Mora3.PinnedProxyTransferSelection/v1',terminal_pin_prefix_sha256='a'*64,
            selection=select_proxy_transfer(contract,outcomes))))
    for key,value in (('source_snapshot_sha256','0'*64),('target_parameter_count',1000000),
                      ('maximum_updates',True),('maximum_seconds',0),('target_width',3)):
        with pytest.raises(ValueError):target_run_specs(p,contract,outcomes,**(common|{key:value}))
