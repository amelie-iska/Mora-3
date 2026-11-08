"""Exclusive terminal attempt reservation and non-overwriting receipt publication.

No optimizer work, orphan adoption, retry, checkpoint resume or launch authority.
An interrupted operation preserves its directory/files and blocks new work.
"""
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path

from mora3.data.admet import canonical
from .mup_trial_artifacts import load_proxy_trial_artifact
from .mup_campaign_recovery import ReceiptPin, recover_terminal_prefix


def _sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


@dataclass(frozen=True)
class AttemptReservation:
    campaign_directory: Path
    campaign_manifest_sha256: str
    trial_id: str
    prior_pins: tuple[ReceiptPin, ...]

    @property
    def artifact_directory(self):
        return self.campaign_directory/self.trial_id/'attempt-001'


def reserve_next_attempt(directory, *, selection, pins):
    """Atomically claim the next trial and persist its parent before returning.

    Only the invocation that creates the directory receives a reservation. A
    caller must never reconstruct a reservation to restart optimizer work after
    failure: existing partial or orphan attempts require explicit reconciliation.
    """
    root=Path(directory).absolute()
    state=recover_terminal_prefix(root,selection=selection,pins=pins)
    if state.next_attempt_directory is None:
        raise ValueError('All planned trials already terminal')
    trial=state.next_attempt_directory.parent
    trial.mkdir()  # exclusive claim; no exist_ok, no deletion on failure
    _sync_directory(root)
    return AttemptReservation(root,selection.campaign_manifest_sha256,trial.name,pins)


def publish_terminal_receipt(reservation, *, selection, artifact_sha256):
    """Re-read a pinned exact artifact, then durably publish its canonical receipt.

    Exclusive pending-file creation also serializes concurrent publishers. No
    failure path cleans up pending files; they remain fail-closed evidence.
    A crash after publication but before external pin storage leaves an orphan
    that recovery deliberately refuses to adopt automatically.
    """
    if (type(reservation) is not AttemptReservation
            or selection.campaign_manifest_sha256!=reservation.campaign_manifest_sha256):
        raise ValueError('Exact campaign-bound reservation required')
    root=reservation.campaign_directory
    trial=reservation.artifact_directory.parent
    if (any(p.is_symlink() for p in (root,*root.parents,trial,reservation.artifact_directory))
            or not trial.is_dir() or {p.name for p in trial.iterdir()}!={'attempt-001'}
            or not reservation.artifact_directory.is_dir()
            or {p.name for p in reservation.artifact_directory.iterdir()}!={'terminal.json','metrics.jsonl'}
            or any(p.is_symlink() or not p.is_file() for p in reservation.artifact_directory.iterdir())):
        raise ValueError('Exact unpublished regular terminal artifact required')
    outcome,_=load_proxy_trial_artifact(reservation.artifact_directory,artifact_sha256,selection=selection)
    if outcome.trial_id!=reservation.trial_id:
        raise ValueError('Reservation and terminal artifact trial differ')
    _sync_directory(trial)
    raw=canonical(dict(schema='Mora3.ProxyTrialControllerReceipt/v1',
        campaign_manifest_sha256=reservation.campaign_manifest_sha256,
        trial_id=reservation.trial_id,
        artifact_directory=f'{reservation.trial_id}/attempt-001',artifact_sha256=artifact_sha256))
    pending=trial/'receipt.pending'
    with pending.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    # Persist pending evidence before linking its immutable bytes under the final name.
    _sync_directory(trial)
    os.link(pending,trial/'receipt.json')
    _sync_directory(trial)
    pending.unlink()  # own successfully published temporary; never an existing artifact
    _sync_directory(trial)
    pin=ReceiptPin(reservation.trial_id,sha256(raw).hexdigest())
    recover_terminal_prefix(root,selection=selection,pins=(*reservation.prior_pins,pin))
    return pin
