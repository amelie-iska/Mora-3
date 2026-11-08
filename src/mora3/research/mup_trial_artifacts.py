"""Persist a canonical terminal trial and verify its metric history before selection."""

from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from mora3.data.admet import canonical

from .metric_records import read_history, write_history
from .mup_campaign import ProxyTrialExecution
from .mup_sweep_selection import (
    FrozenProxySelection, ProxyRunOutcome, ValidationCoordinate, _eligible_trial,
    planned_proxy_trials,
)

_SCHEMA = 'Mora3.ProxyTrialTerminalArtifact/v1'


def _outcome_wire(outcome):
    wire = asdict(outcome)
    del wire['run_artifact_sha256']
    wire['gpu_seconds'] = str(outcome.gpu_seconds)
    for row in wire['validation']:
        row['mean_loss'] = str(row['mean_loss'])
    return wire


def _file_sha256(path):
    with path.open('rb') as file:
        return file_digest(file,'sha256').hexdigest()


def _verify_selection_binding(outcome, metrics, selection):
    if type(selection) is not FrozenProxySelection:
        raise ValueError('Exact frozen selection contract required')
    selection.__post_init__()
    _eligible_trial(selection,outcome)
    trial = next((trial for trial in planned_proxy_trials() if trial.trial_id == outcome.trial_id),None)
    if trial is None or any(
        row.trial_id != trial.trial_id or row.seed != trial.seed or row.split == 'test'
        or row.x_axis == 'optimizer_step' and row.step > outcome.completed_updates
        for row in metrics
    ):
        raise ValueError('Metrics must bind to the planned trial, seed and observed update range')
    if len({row.run_id for row in metrics}) > 1:
        raise ValueError('Exact single-run metric history required')

    if outcome.validation:
        observed = [
            (row.task,row.source_sha256,row.denominator,row.value) for row in metrics
            if row.name == 'matched_native_validation_objective' and row.scope == 'validation'
            and row.split == 'validation' and row.x_axis == 'optimizer_step'
            and row.step == outcome.comparison_update
        ]
        expected = [
            (row.task_key,row.source_sha256,row.observed_targets,float(row.mean_loss))
            for row in outcome.validation
        ]
        if observed != expected:
            raise ValueError('Terminal comparison must match the complete ordered metric history')


def seal_proxy_trial_execution(directory, execution, *, selection):
    """Create a new artifact for an exact within-budget terminal execution.

    The final manifest is published only after the canonical metric history
    has been flushed and independently read back. An interrupted write leaves
    an inspectable directory without a committed manifest.
    """
    if type(execution) is not ProxyTrialExecution:
        raise ValueError('Exact terminal proxy trial execution required')
    provisional = execution.terminal_outcome(run_artifact_sha256='0'*64)
    _verify_selection_binding(provisional,execution.metrics,selection)
    directory = Path(directory)
    directory.mkdir()
    history = directory/'metrics.jsonl'
    count = write_history(history,execution.metrics)
    if read_history(history) != execution.metrics:
        raise ValueError('Canonical metric history readback differs from terminal execution')
    raw = canonical(dict(
        schema=_SCHEMA,terminal=_outcome_wire(provisional),
        metric_history_sha256=_file_sha256(history),metric_count=count,
    ))
    with NamedTemporaryFile(dir=directory,prefix='.terminal-',mode='w+b') as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())
        os.link(file.name,directory/'terminal.json')
    # Commit the manifest entry and then the new artifact directory entry.
    for path in (directory,directory.parent):
        fd = os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return sha256(raw).hexdigest()


def load_proxy_trial_artifact(directory, expected_manifest_sha256, *, selection):
    """Return the exact selector row and validated canonical metric history."""
    directory = Path(directory)
    raw = (directory/'terminal.json').read_bytes()
    digest = sha256(raw).hexdigest()
    if digest != expected_manifest_sha256:
        raise ValueError('Pinned terminal manifest digest differs')
    try:
        document = json.loads(raw)
        if (set(document) != {'schema','terminal','metric_history_sha256','metric_count'}
                or document['schema'] != _SCHEMA or canonical(document) != raw
                or type(document['metric_count']) is not int or document['metric_count'] < 0):
            raise ValueError('Exact canonical terminal manifest required')
        terminal = document['terminal']
        terminal['gpu_seconds'] = Decimal(terminal['gpu_seconds'])
        terminal['validation'] = tuple(
            ValidationCoordinate(**dict(row,mean_loss=Decimal(row['mean_loss'])))
            for row in terminal['validation']
        )
        outcome = ProxyRunOutcome(**terminal,run_artifact_sha256=digest)
        if canonical(_outcome_wire(outcome)) != canonical(json.loads(raw)['terminal']):
            raise ValueError('Exact terminal selector representation required')
    except (AttributeError,KeyError,TypeError,InvalidOperation) as error:
        raise ValueError('Invalid terminal manifest fields') from error
    history = directory/'metrics.jsonl'
    if _file_sha256(history) != document['metric_history_sha256']:
        raise ValueError('Pinned metric history digest differs')
    metrics = read_history(history)
    if len(metrics) != document['metric_count'] or _file_sha256(history) != document['metric_history_sha256']:
        raise ValueError('Pinned metric history count or bytes changed')
    _verify_selection_binding(outcome,metrics,selection)
    return outcome,metrics
