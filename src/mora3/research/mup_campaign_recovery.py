"""Read-only terminal-prefix recovery; never adopts or resumes an attempt.

The existing exclusive worker owns synchronization and durable receipt publication.
This is a snapshot check, not a lock or authorization to launch a trial. Pins must
come from independently trusted controller state, never from directory discovery.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

from mora3.data.admet import canonical
from .mup_sweep_selection import FrozenProxySelection, ProxyRunOutcome, planned_proxy_trials
from .mup_trial_artifacts import load_proxy_trial_artifact


@dataclass(frozen=True)
class ReceiptPin:
    trial_id: str
    receipt_sha256: str


@dataclass(frozen=True)
class TerminalPrefix:
    terminal_trial_ids: tuple[str, ...]
    outcomes: tuple[ProxyRunOutcome, ...]
    next_attempt_directory: Path | None


def recover_terminal_prefix(directory, *, selection, pins):
    """Verify an exact pinned prefix or refuse all advancement.

    Each trial directory contains ``attempt-001`` and ``receipt.json``. The
    receipt binds the campaign, trial, deterministic relative artifact directory,
    and terminal artifact hash. Any unpinned entry (including an orphan complete
    attempt) blocks recovery without modifying or automatically adopting it.
    The returned next path is advisory and has not been created or reserved.
    """
    if type(selection) is not FrozenProxySelection:
        raise ValueError('Exact frozen selection required')
    selection.__post_init__()
    grid=planned_proxy_trials()
    if (type(pins) is not tuple or len(pins)>len(grid)
            or any(type(pin) is not ReceiptPin for pin in pins)
            or tuple(pin.trial_id for pin in pins)!=tuple(t.trial_id for t in grid[:len(pins)])
            or any(type(pin.receipt_sha256) is not str
                   or re.fullmatch('[0-9a-f]{64}',pin.receipt_sha256) is None for pin in pins)):
        raise ValueError('Exact ordered independently pinned terminal prefix required')
    root=Path(directory)
    if any(path.is_symlink() for path in (root,*root.parents)) or not root.is_dir():
        raise ValueError('Existing dedicated nonsymlink campaign directory required')
    expected={pin.trial_id for pin in pins}
    if {path.name for path in root.iterdir()}!=expected:
        raise ValueError('Missing receipt or unpinned orphan/partial entry; explicit reconciliation required')
    outcomes=[]
    for pin in pins:
        trial=root/pin.trial_id
        artifact=trial/'attempt-001'
        receipt=trial/'receipt.json'
        if (trial.is_symlink() or not trial.is_dir()
                or {p.name for p in trial.iterdir()}!={'attempt-001','receipt.json'}
                or artifact.is_symlink() or not artifact.is_dir()
                or receipt.is_symlink() or not receipt.is_file()
                or {p.name for p in artifact.iterdir()}!={'terminal.json','metrics.jsonl'}
                or any(p.is_symlink() or not p.is_file() for p in artifact.iterdir())):
            raise ValueError('Exact regular committed receipt and terminal artifact required')
        raw=receipt.read_bytes()
        if sha256(raw).hexdigest()!=pin.receipt_sha256:
            raise ValueError('Pinned controller receipt digest differs')
        try:
            document=json.loads(raw)
            if (type(document) is not dict or set(document)!={
                    'schema','campaign_manifest_sha256','trial_id','artifact_directory','artifact_sha256'}
                    or canonical(document)!=raw
                    or document['schema']!='Mora3.ProxyTrialControllerReceipt/v1'
                    or document['campaign_manifest_sha256']!=selection.campaign_manifest_sha256
                    or document['trial_id']!=pin.trial_id
                    or document['artifact_directory']!=f'{pin.trial_id}/attempt-001'
                    or type(document['artifact_sha256']) is not str
                    or re.fullmatch('[0-9a-f]{64}',document['artifact_sha256']) is None):
                raise ValueError('Exact campaign-bound canonical controller receipt required')
        except (TypeError,UnicodeError) as error:
            raise ValueError('Malformed controller receipt') from error
        outcome,_=load_proxy_trial_artifact(artifact,document['artifact_sha256'],selection=selection)
        if outcome.trial_id!=pin.trial_id:
            raise ValueError('Receipt and terminal trial differ')
        outcomes.append(outcome)
    return TerminalPrefix(tuple(pin.trial_id for pin in pins),tuple(outcomes),
        root/grid[len(pins)].trial_id/'attempt-001' if len(pins)<len(grid) else None)
