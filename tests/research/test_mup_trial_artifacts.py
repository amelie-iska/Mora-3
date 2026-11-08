from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
import os

import pytest

from mora3.research.metric_records import MetricRecord
from mora3.research.mup_campaign import ProxyTrialExecution
from mora3.research.mup_sweep_selection import planned_proxy_trials
from mora3.research.mup_trial_artifacts import load_proxy_trial_artifact, seal_proxy_trial_execution
from test_mup_sweep_selection import fixture


def execution():
    contract, runs = fixture()
    run = runs[0]
    spec = planned_proxy_trials()[0]
    metrics = tuple(
        MetricRecord(run_id='proxy-run-1',trial_id=spec.trial_id,seed=spec.seed,
            segment_id='original',step=4500,x_axis='optimizer_step',task=c.task_key,split='validation',
            scope='validation',source_sha256=c.source_sha256,evidence_sha256='e'*64,
            name='matched_native_validation_objective',value=float(c.mean_loss),unit='native loss',
            direction='minimize',denominator=c.observed_targets,
            numerator=float(c.mean_loss*c.observed_targets))
        for c in run.validation
    )
    return contract,ProxyTrialExecution(run.trial_id,run.campaign_manifest_sha256,run.status,
        run.completed_updates,run.gpu_seconds,run.validation_exposure_sha256,
        run.comparison_update,run.validation,(),metrics,None)


def test_readback_retains_exact_decimal_and_refuses_replacement(tmp_path):
    contract,run=execution()
    directory=tmp_path/'trial'
    digest=seal_proxy_trial_execution(directory,run,selection=contract)
    assert digest==sha256((directory/'terminal.json').read_bytes()).hexdigest()
    read,metrics=load_proxy_trial_artifact(directory,digest,selection=contract)
    assert read==run.terminal_outcome(run_artifact_sha256=digest)
    assert type(read.validation[0].mean_loss) is Decimal
    assert metrics==run.metrics
    with pytest.raises(FileExistsError):
        seal_proxy_trial_execution(directory,run,selection=contract)


def test_rejects_mismatched_comparison_before_artifact_creation(tmp_path):
    contract,run=execution()
    changed=replace(run.metrics[0],value=.5)
    directory=tmp_path/'trial'
    with pytest.raises(ValueError,match='comparison'):
        seal_proxy_trial_execution(directory,replace(run,metrics=(changed,*run.metrics[1:])),selection=contract)
    assert not directory.exists()


def test_forged_or_incomplete_artifact_is_rejected(tmp_path):
    contract,run=execution()
    directory=tmp_path/'trial'
    digest=seal_proxy_trial_execution(directory,run,selection=contract)
    with pytest.raises(ValueError,match='manifest'):
        load_proxy_trial_artifact(directory,'f'*64,selection=contract)
    with (directory/'metrics.jsonl').open('ab') as stream:
        stream.write(b'\n')
    with pytest.raises(ValueError,match='metric history'):
        load_proxy_trial_artifact(directory,digest,selection=contract)


def test_failure_retains_forensic_validation_metrics_but_not_selection_packet(tmp_path):
    contract,run=execution()
    run=replace(run,status='oom',validation_exposure_sha256=None,comparison_update=None,
                validation=(),completed_updates=4600,failure_reason='resource envelope exhausted')
    directory=tmp_path/'trial'
    digest=seal_proxy_trial_execution(directory,run,selection=contract)
    read,metrics=load_proxy_trial_artifact(directory,digest,selection=contract)
    assert read.status=='oom' and not read.validation and read.failure_reason==run.failure_reason
    assert metrics==run.metrics


def test_manifest_directory_and_parent_are_synced_in_order(tmp_path,monkeypatch):
    contract,run=execution()
    directory=tmp_path/'trial'
    directory_syncs=[]
    original_fsync=os.fsync

    def observe_directory_sync(fd):
        inode=os.fstat(fd).st_ino
        if inode in (directory.stat().st_ino,tmp_path.stat().st_ino):
            directory_syncs.append(inode)
            assert (directory/'terminal.json').is_file()
        original_fsync(fd)

    monkeypatch.setattr(os,'fsync',observe_directory_sync)
    seal_proxy_trial_execution(directory,run,selection=contract)
    assert directory_syncs==[directory.stat().st_ino,tmp_path.stat().st_ino]
