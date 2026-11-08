"""One admitted fresh proxy trial in the existing exclusive model worker.

No scheduler, retry, checkpoint resume, process owner or hard-timeout service is
created here. Authoritative pins come from external controller state. The owner
must persist the returned pin before attempting the next trial. This function
captures metric batches durably, but a killed process can still leave a
reserved partial/orphan directory, which recovery refuses to reuse. It does not
grant training permission or establish readiness beyond existing admission gates.
"""
import os
from pathlib import Path

from .runtime_capacity_gate import require_runtime_window_capacity

from mora3.data.admet import canonical
from mora3.model.full_owner import SourceBoundOwner
from mora3.research.primary_training import NativeMuUpdateEngine
from .mup_campaign import (
    FrozenPrimaryProxyCampaign, ProxyTrialExecution, MetricCaptureError, _require_runtime, run_primary_proxy_trial,
)
from .mup_campaign_recovery import recover_terminal_prefix
from .mup_campaign_publication import reserve_next_attempt, publish_terminal_receipt
from .mup_trial_artifacts import seal_proxy_trial_execution
from .mup_metric_journal import DurableMetricJournal, read_forensic_prefix


def _record_blocked(reservation,phase,error,execution,journal_path):
    """Best-effort non-overwriting disposition; never invent a selector outcome."""
    document=dict(schema='Mora3.ProxyTrialBlockedAttempt/v1',
        campaign_manifest_sha256=reservation.campaign_manifest_sha256,
        trial_id=reservation.trial_id,phase=phase,error_type=type(error).__name__,
        automatically_resumable=False,per_update_durable_journal=True,
        journal_path=str(journal_path),journal_may_be_partial=True,
        terminal_artifact_may_be_partial=True)
    if type(execution) is ProxyTrialExecution:
        document.update(observed_status=execution.status,
            observed_trial_id=execution.trial_id,completed_updates=execution.completed_updates,
            measured_wall_seconds=str(execution.measured_wall_seconds),
            in_memory_metric_count=len(execution.metrics))
    trial=reservation.artifact_directory.parent
    with (trial/'blocked.json').open('xb') as stream:
        stream.write(canonical(document));stream.flush();os.fsync(stream.fileno())
    fd=os.open(trial,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def execute_one_primary_proxy_trial(directory, *, engine,campaign,trial_id,pins,
        bindings,validation_view,validation_evaluators,metric_identity,root,journal_path,evidence_paths=None,capacity_request=None):
    """Preflight, reserve once, execute once, seal and publish one verified receipt.

    Errors propagate. Any error after reservation preserves the reserved attempt
    and attempts a non-sensitive blocked disposition; an over-budget execution
    remains forensic evidence, never a clamped or fabricated selector row. Hard
    kill/power loss may prevent that disposition; the reservation is the durable
    no-reuse guard. The separately designated journal file is exclusive and never
    reopened. Complete captured batches survive acknowledged fsync; a kill between
    optimizer work and capture can lose latest telemetry. Journal location must be
    retained in external attempt evidence, outside the strict campaign directory.
    Existing worker ownership and external hard time limits still apply.
    """
    if type(engine) is not NativeMuUpdateEngine or type(engine.model) is not SourceBoundOwner:
        raise ValueError('Exact native engine and source-bound owner required')
    engine.model.require_training_admission()
    if type(campaign) is not FrozenPrimaryProxyCampaign:
        raise ValueError('Exact frozen primary campaign required')
    require_runtime_window_capacity(root,campaign=campaign,bindings=bindings,request=capacity_request)
    journal_path=Path(journal_path).absolute()
    campaign_path=Path(directory).resolve()
    if (journal_path.resolve().is_relative_to(campaign_path)
            or any(p.is_symlink() for p in (journal_path,*journal_path.parents))
            or not journal_path.parent.is_dir()):
        raise ValueError('Separate existing regular journal parent outside campaign directory required')
    selection=campaign.selection_contract()
    prefix=recover_terminal_prefix(directory,selection=selection,pins=pins)
    trial=campaign.require_next_trial(trial_id,prefix.terminal_trial_ids)
    _require_runtime(engine,campaign,trial,bindings,validation_view,validation_evaluators,
        metric_identity,root,evidence_paths)
    # Repeat recovery inside exclusive reservation; never infer terminal IDs from names.
    reservation=reserve_next_attempt(directory,selection=selection,pins=pins)
    execution=None
    phase='reserved'
    try:
        if reservation.trial_id!=trial_id:
            raise ValueError('Reserved trial differs from requested admitted trial')
        phase='journal-create'
        capture_failed=False
        with DurableMetricJournal(journal_path) as journal:
            def capture(rows):
                nonlocal capture_failed
                try:
                    journal.append_batch(rows)
                except BaseException as error:
                    capture_failed=True
                    if isinstance(error,Exception):
                        raise MetricCaptureError('Durable metric capture failed') from error
                    raise
            phase='execute'
            execution=run_primary_proxy_trial(engine,campaign,trial_id=trial_id,
                terminal_trial_ids=prefix.terminal_trial_ids,bindings=bindings,
                validation_view=validation_view,validation_evaluators=validation_evaluators,
                metric_identity=metric_identity,root=root,evidence_paths=evidence_paths,
                metric_capture=capture)
            if capture_failed:
                raise MetricCaptureError('Interrupted durable metric capture forbids terminal publication')
        if (type(execution) is not ProxyTrialExecution or execution.trial_id!=trial_id
                or execution.campaign_manifest_sha256!=campaign.sha256):
            raise ValueError('Exact reserved terminal trial execution required')
        phase='journal-readback'
        observed=read_forensic_prefix(journal_path)
        if observed.torn_tail_bytes or observed.metrics!=execution.metrics:
            raise ValueError('Complete durable journal must exactly match terminal metrics')
        phase='seal'
        digest=seal_proxy_trial_execution(reservation.artifact_directory,execution,selection=selection)
        phase='publish'
        return publish_terminal_receipt(reservation,selection=selection,artifact_sha256=digest)
    except BaseException as error:
        try: _record_blocked(reservation,phase,error,execution,journal_path)
        except Exception as recording_error:
            error.add_note('Blocked disposition could not be persisted: '+type(recording_error).__name__+
                '; reserved attempt remains non-reusable.')
        raise
