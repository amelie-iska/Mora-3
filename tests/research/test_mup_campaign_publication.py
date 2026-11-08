from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from test_mup_trial_artifacts import execution
from mora3.research.mup_trial_artifacts import seal_proxy_trial_execution
from mora3.research.mup_campaign_recovery import recover_terminal_prefix
from mora3.research import mup_campaign_publication as pub


def test_exclusive_reservation_and_verified_publication(tmp_path):
    selection,run=execution()
    def reserve():
        try: return pub.reserve_next_attempt(tmp_path,selection=selection,pins=())
        except (ValueError,FileExistsError): return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:reserve(),range(2)))
    reservations=[r for r in results if r is not None]
    assert len(reservations)==1
    reservation=reservations[0]
    assert not reservation.artifact_directory.exists()
    digest=seal_proxy_trial_execution(reservation.artifact_directory,run,selection=selection)
    pin=pub.publish_terminal_receipt(reservation,selection=selection,artifact_sha256=digest)
    state=recover_terminal_prefix(tmp_path,selection=selection,pins=(pin,))
    assert state.terminal_trial_ids==(run.trial_id,)
    with pytest.raises((ValueError,FileExistsError)):
        pub.publish_terminal_receipt(reservation,selection=selection,artifact_sha256=digest)


def test_reservation_parent_sync_failure_preserves_blocker(tmp_path,monkeypatch):
    selection,run=execution()
    def fail(_): raise OSError('simulated fsync interruption')
    monkeypatch.setattr(pub,'_sync_directory',fail)
    with pytest.raises(OSError): pub.reserve_next_attempt(tmp_path,selection=selection,pins=())
    assert (tmp_path/run.trial_id).is_dir()
    with pytest.raises(ValueError): recover_terminal_prefix(tmp_path,selection=selection,pins=())


def test_interrupted_publication_keeps_pending_and_refuses_retry(tmp_path,monkeypatch):
    selection,run=execution()
    reservation=pub.reserve_next_attempt(tmp_path,selection=selection,pins=())
    digest=seal_proxy_trial_execution(reservation.artifact_directory,run,selection=selection)
    def fail(*args): raise OSError('simulated interrupted hardlink')
    monkeypatch.setattr(pub.os,'link',fail)
    with pytest.raises(OSError): pub.publish_terminal_receipt(reservation,selection=selection,artifact_sha256=digest)
    assert (reservation.artifact_directory.parent/'receipt.pending').is_file()
    assert not (reservation.artifact_directory.parent/'receipt.json').exists()
    with pytest.raises(ValueError): recover_terminal_prefix(tmp_path,selection=selection,pins=())
    with pytest.raises((ValueError,FileExistsError)):
        pub.publish_terminal_receipt(reservation,selection=selection,artifact_sha256=digest)


def test_digest_mismatch_never_publishes(tmp_path):
    selection,run=execution()
    reservation=pub.reserve_next_attempt(tmp_path,selection=selection,pins=())
    seal_proxy_trial_execution(reservation.artifact_directory,run,selection=selection)
    with pytest.raises(ValueError):
        pub.publish_terminal_receipt(reservation,selection=selection,artifact_sha256='f'*64)
    assert {p.name for p in reservation.artifact_directory.parent.iterdir()}=={'attempt-001'}


def test_concurrent_publication_has_one_winner(tmp_path):
    selection,run=execution()
    reservation=pub.reserve_next_attempt(tmp_path,selection=selection,pins=())
    digest=seal_proxy_trial_execution(reservation.artifact_directory,run,selection=selection)
    def publish():
        try: return pub.publish_terminal_receipt(reservation,selection=selection,artifact_sha256=digest)
        except (ValueError,FileExistsError): return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:publish(),range(2)))
    winners=[r for r in results if r is not None]
    assert len(winners)==1
    assert recover_terminal_prefix(tmp_path,selection=selection,pins=tuple(winners)).terminal_trial_ids==(run.trial_id,)


def test_interruption_after_link_preserves_committed_orphan(tmp_path,monkeypatch):
    selection,run=execution()
    reservation=pub.reserve_next_attempt(tmp_path,selection=selection,pins=())
    digest=seal_proxy_trial_execution(reservation.artifact_directory,run,selection=selection)
    original=pub._sync_directory
    def fail_after_link(path):
        if (Path(path)/'receipt.json').exists(): raise OSError('simulated post-link interruption')
        original(path)
    monkeypatch.setattr(pub,'_sync_directory',fail_after_link)
    with pytest.raises(OSError): pub.publish_terminal_receipt(reservation,selection=selection,artifact_sha256=digest)
    assert (reservation.artifact_directory.parent/'receipt.json').is_file()
    assert (reservation.artifact_directory.parent/'receipt.pending').is_file()
    with pytest.raises(ValueError): recover_terminal_prefix(tmp_path,selection=selection,pins=())
