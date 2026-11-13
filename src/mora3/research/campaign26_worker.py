"""Count the existing one-trial worker under the user's26-run campaign.

No new worker, supervisor, thread, optimizer loop, data admission or GPU lease.
Only an already admitted native owner can consume a real run slot. Failure after
reservation consumes that slot and retains evidence; there is no automatic retry.
"""
from hashlib import sha256
import json
import os
from pathlib import Path

from mora3.data.admet import canonical
from mora3.model.full_owner import SourceBoundOwner
from .primary_training import NativeMuUpdateEngine
from .mup_campaign import FrozenPrimaryProxyCampaign
from .mup_campaign_pin_store import (load_pinned_campaign,execute_one_pinned_trial,
                                     publish_pinned_transfer_selection)
from .mup_campaign_recovery import recover_terminal_prefix
from .mup_sweep_selection import select_proxy_transfer, LR, READOUT, ATTENTION, SEEDS
from .campaign26 import Campaign26,RunBudget26,_safe,_publish


def _read_document(path):
    path=_safe(path)
    if not path.is_file():raise ValueError('Existing regular directive/selection artifact required')
    with path.open('rb')as f:raw=f.read(1_048_577)
    if len(raw)>1_048_576:raise ValueError('Directive/selection artifact exceeds bounded size')
    try:doc=json.loads(raw)
    except (ValueError,UnicodeError)as exc:raise ValueError('Complete directive/selection JSON required')from exc
    if type(doc)is not dict:raise ValueError('Directive/selection object required')
    return raw,doc


def require_current_directive(root,plan):
    if type(plan)is not Campaign26:raise ValueError('Exact26-run plan required')
    root=_safe(root);_,pointer=_read_document(root/'planning/current_mup_campaign.json')
    if set(pointer)!={'schema','path','sha256'}or pointer['schema']!='Mora3.CurrentCampaignDirective/v1':
        raise ValueError('Current campaign directive pointer required')
    name=pointer['path']
    if (type(name)is not str or Path(name).is_absolute()or '..'in Path(name).parts
            or not name.startswith('planning/')or not name.endswith('.json')):
        raise ValueError('Exact project planning directive path required')
    raw,doc=_read_document(root/name)
    if sha256(raw).hexdigest()!=pointer['sha256']or pointer['sha256']!=plan.directive_sha256:
        raise ValueError('Current user directive changed or does not match the run budget')
    expected=dict(maximum_training_runs=26,proxy_sweep_runs=24,full_target_training_runs=2,
        standard_parameterization_baseline_runs=0,intermediate_width_training_runs=0,
        naive_standard_parameterization_target_runs=0)
    if (doc.get('schema')!='Mora3.Campaign26UserDirective/v1'
            or any(type(doc.get(k))is not int or doc[k]!=v for k,v in expected.items())
            or doc.get('target_seeds')!=list(plan.target_seeds)):
        raise ValueError('The active user directive must retain exactly24+2 runs and no baselines')
    grid=dict(master_learning_rates=[str(x)for x in LR],readout_multipliers=[str(x)for x in READOUT],
        attention_key_multipliers=[str(x)for x in ATTENTION],seeds=list(SEEDS))
    if canonical(doc.get('primary_grid'))!=canonical(grid):
        raise ValueError('Current user directive grid differs from the exact native24-trial grid')
    return pointer['sha256']


def _prefix(budget,pin_directory,campaign_directory,campaign):
    if type(budget)is not RunBudget26 or type(campaign)is not FrozenPrimaryProxyCampaign:
        raise ValueError('Exact26-run budget and native primary campaign required')
    if (budget.plan.proxy_campaign_sha256!=campaign.sha256
            or budget.plan.source_snapshot_sha256!=campaign.source_snapshot_sha256):
        raise ValueError('Run budget and native campaign/source snapshot differ')
    selection=campaign.selection_contract()
    pins=load_pinned_campaign(pin_directory,campaign_directory,selection=selection)
    prefix=recover_terminal_prefix(campaign_directory,selection=selection,pins=pins.pins)
    with budget._locked():states,_=budget._load()
    if len(states)!=len(prefix.outcomes):
        raise ValueError('Run-budget/native terminal prefixes disagree; explicit reconciliation required')
    for state,outcome in zip(states,prefix.outcomes,strict=True):
        claim,entered,terminal=state
        if (not entered or terminal is None or claim['run_id']!=outcome.trial_id
                or terminal['status']!=outcome.status or terminal['artifact_sha256']!=outcome.run_artifact_sha256):
            raise ValueError('Native terminal outcome does not match its counted run')
    return pins,prefix


def execute_counted_proxy_trial(*,budget,pin_directory,campaign_directory,campaign,trial_id,
        engine,root,failure_artifact,target_blueprint=None,**worker_kwargs):
    """Execute at most one already-qualified primary run through the original worker."""
    if type(engine)is not NativeMuUpdateEngine or type(engine.model)is not SourceBoundOwner:
        raise ValueError('Exact native update engine and registered source-bound owner required')
    engine.model.require_training_admission()  # no reservation while the complete data gate is closed
    if type(budget)is not RunBudget26:raise ValueError('Persistent26-run budget required')
    require_current_directive(root,budget.plan)
    from .campaign26_transfer import require_target_blueprint
    require_target_blueprint(budget.plan,target_blueprint)
    if (target_blueprint.configuration.base_width!=campaign.proxy_width
            or target_blueprint.width_family_sha256!=campaign.width_family_sha256
            or target_blueprint.task_cycle!=campaign.task_cycle
            or target_blueprint.gradient_clip_max_norm!=campaign.gradient_clip_max_norm
            or target_blueprint.decay_aliases!=campaign.decay_aliases):
        raise ValueError('Target blueprint differs from the fixed native proxy width family or training policy')
    _,prefix=_prefix(budget,pin_directory,campaign_directory,campaign)
    if len(prefix.outcomes)>=24 or trial_id!=budget.plan.run_ids[len(prefix.outcomes)]:
        raise ValueError('Only the next of24 primary trials may execute')
    error_path=_safe(failure_artifact)
    if error_path.exists()or not error_path.parent.is_dir():raise ValueError('Fresh failure artifact path required')
    for protected in(budget.directory,Path(pin_directory),Path(campaign_directory)):
        if error_path.resolve().is_relative_to(protected.resolve()):
            raise ValueError('Failure artifact must remain outside strict controller/attempt namespaces')
    claim=budget.reserve(trial_id);claim.enter()
    try:
        state=execute_one_pinned_trial(pin_directory,campaign_directory,campaign=campaign,
            trial_id=trial_id,engine=engine,root=root,**worker_kwargs)
        result=recover_terminal_prefix(campaign_directory,selection=campaign.selection_contract(),pins=state.pins)
        if len(result.outcomes)!=len(prefix.outcomes)+1 or result.outcomes[-1].trial_id!=trial_id:
            raise ValueError('Worker did not return exactly one new native terminal outcome')
        outcome=result.outcomes[-1]
        budget.finish(claim,status=outcome.status,artifact_sha256=outcome.run_artifact_sha256)
        return state
    except BaseException as error:
        document=dict(schema='Mora3.Campaign26ExecutionFailure/v1',campaign26_sha256=budget.plan.sha256,
            trial_id=trial_id,error_type=type(error).__name__,run_slot_consumed=True,
            automatically_resumable=False,source_admission_issued=False)
        try:
            _publish(error_path,document)
            budget.finish(claim,status='failed',artifact_sha256=sha256(canonical(document)).hexdigest())
        except Exception as recording_error:
            error.add_note('Run reservation remains consumed; failure receipt publication failed: '+type(recording_error).__name__)
        raise


def publish_counted_transfer_selection(*,budget,pin_directory,campaign_directory,campaign,root,output):
    """Use the full native24-result prefix, then bind one selection to both targets."""
    require_current_directive(root,budget.plan)
    _,prefix=_prefix(budget,pin_directory,campaign_directory,campaign)
    result=select_proxy_transfer(campaign.selection_contract(),prefix.outcomes)
    if result['frozen_transfer_hyperparameters']is None:
        raise ValueError('No eligible two-seed proxy winner; targets must not start')
    digest=publish_pinned_transfer_selection(pin_directory,campaign_directory,output,
        selection=campaign.selection_contract())
    budget.bind_selection(digest)
    return digest


def read_counted_transfer_selection(*,budget,root,selection_path):
    """Verify the immutable selected artifact before preparing either target run."""
    require_current_directive(root,budget.plan)
    current=budget.snapshot();raw,document=_read_document(selection_path)
    if sha256(raw).hexdigest()!=current['selection_sha256']:
        raise ValueError('Target must use the budget-bound immutable selection artifact')
    if (set(document)!={'schema','terminal_pin_prefix_sha256','selection'}
            or document['schema']!='Mora3.PinnedProxyTransferSelection/v1'):
        raise ValueError('Exact existing native selector publication required')
    result=document['selection']
    if (result.get('campaign_manifest_sha256')!=budget.plan.proxy_campaign_sha256
            or result.get('full_terminal_trial_count')!=24 or result.get('target_validation_or_test_used')is not False
            or result.get('launch_authorized')is not False or result.get('frozen_transfer_hyperparameters')is None):
        raise ValueError('Complete native validation-only winner required')
    return result
