"""Metadata-only exact finite59/MCSA/reviewed-text composition; never admission."""
from dataclasses import dataclass
from pathlib import Path
from mora3.data.admet import digest
from . import combined60_binding as legacy
from .source_text_campaign import ReviewedTextCampaignRoute,SPECS
from .matched_validation import TextCampaignValidationSources,TextValidationSourceRow,matched_validation_source_policy_sha256
from .mup_campaign import FrozenPrimaryProxyCampaign,NativeTrainRoutePlan,executable_adapter_sha256
from .primary_campaign_assembly import _reference,_read
from .mup_sweep_selection import MAX_UPDATES

@dataclass(frozen=True)
class Combined62:
    bindings:tuple
    plans:tuple
    validation:object
    evaluators:tuple
    training_authorized:bool=False
    allocation_authorized:bool=False
    complete_campaign:bool=False

def text_plans(root,*,routes,scan_limits):
    routes=tuple(routes)
    if (len(routes)!=2 or any(type(r) is not ReviewedTextCampaignRoute for r in routes)
        or {r.family for r in routes}!=set(SPECS) or len({r.task_key for r in routes})!=2):
        raise ValueError('Exactly two reviewed native text routes required')
    if type(scan_limits) is not dict or set(scan_limits)!={r.task_key for r in routes} or any(type(v) is not int or v<1 for v in scan_limits.values()):
        raise ValueError('Explicit positive per-update scan limits for both text routes required')
    return tuple(sorted((NativeTrainRoutePlan(r.task_key,digest(r.initial_cursor),executable_adapter_sha256(r.prepare,root=root),scan_limits[r.task_key]) for r in routes),key=lambda p:p.task_key))

def bind_combined62(root,*,combined60_inputs,text_routes,text_scan_limits,max_selected_text_bytes,campaign,campaign_reference):
    """Bind executable identities to the campaign root without allocating.

    Each route's TextBudget separately binds batch/bytes/materialization/passes.
    Scan limits are per update, never a lifetime allowance or sampling weight.
    """
    root=Path(root).resolve()
    if type(campaign) is not FrozenPrimaryProxyCampaign:raise ValueError('Exact frozen campaign required')
    campaign.__post_init__();ref=_reference(campaign_reference);raw=_read(root,ref)
    if raw!=campaign.wire() or ref.sha256!=campaign.sha256:raise ValueError('Exact pinned combined62 campaign required')
    previous=combined60_inputs['campaign'];routes=tuple(sorted(text_routes,key=lambda r:r.task_key))
    plans=text_plans(root,routes=routes,scan_limits=text_scan_limits)
    for name in ('source_snapshot_sha256','width_family_sha256','proxy_width','proxy_parameter_count','source_role_schedule_sha256','complete_owner_evidence_sha256','runtime_profile_sha256','existing_worker_sha256','gradient_clip_max_norm','decay_aliases','max_one_update_seconds','max_matched_validation_seconds','validation_aggregation'):
        if getattr(campaign,name)!=getattr(previous,name):raise ValueError('Prior campaign identity/policy changed: '+name)
    for name in ('source_snapshot_sha256','same_snapshot_group_guard_sha256','initial_reference_artifact_sha256','comparison_update','read_batch_size','max_selected_arrow_bytes'):
        if getattr(campaign.validation,name)!=getattr(previous.validation,name):raise ValueError('Prior validation policy changed: '+name)
    keys={p.task_key for p in previous.train_routes};newkeys={r.task_key for r in routes}
    if len(keys)!=60 or keys&newkeys:raise ValueError('Exact disjoint sixty plus two route keys required')
    if tuple(k for k in campaign.task_cycle if k in keys)!=previous.task_cycle:raise ValueError('Prior sampling cycle order/multiplicity changed')
    expected=tuple(sorted(previous.train_routes+plans,key=lambda p:p.task_key))
    if campaign.train_routes!=expected:raise ValueError('Exact sixty-two cursor/executable/scan plans required')
    tasks={t.task_key:t for t in campaign.validation.tasks}
    if set(tasks)!={t.task_key for t in previous.validation.tasks}|newkeys:raise ValueError('Exact five validation scopes required')
    for task in previous.validation.tasks:
        if tasks[task.task_key]!=task:raise ValueError('Prior validation coordinate/reference changed')
    if campaign.validation.train_only_dispositions!=previous.validation.train_only_dispositions:raise ValueError('Prior train-only dispositions changed')
    for r in routes:
        r.executable_binding()
        if r.source_snapshot_sha256!=campaign.source_snapshot_sha256:raise ValueError('Text source snapshot differs')
        calls=sum(campaign.task_cycle[i%len(campaign.task_cycle)]==r.task_key for i in range(MAX_UPDATES))
        if calls>r.maximum_windows:raise ValueError('Explicit text finite pass budget cannot supply schedule')
    base=legacy.bind_combined60(root,**combined60_inputs)
    sources=TextCampaignValidationSources(base.validation.sources,routes,max_selected_text_bytes)
    evaluators=tuple(sorted(base.validation.evaluators+tuple((r.task_key,r.evaluate_validation) for r in routes)))
    hashes=tuple(sorted(base.validation.hashes+tuple((r.task_key,executable_adapter_sha256(r.evaluate_validation,root=root)) for r in routes)))
    if campaign.validation.view_policy_sha256!=matched_validation_source_policy_sha256(sources) or campaign.validation_adapter_sha256!=hashes:raise ValueError('Exact mixed source/evaluator identities required')
    for r in routes:
        task=tasks[r.task_key]
        coords=tuple(TextValidationSourceRow(**(c|{'native_group_key_digests':tuple(c['native_group_key_digests'])})) for c in r.validation_coordinates)
        if task.coordinates!=coords or (task.loss_unit,task.target_unit)!=(r.loss_unit,r.target_unit):raise ValueError('Exact text coordinates and units required')
        if sum(r.observed_targets(b) for b in campaign.validation.batches(sources,r.task_key))!=task.observed_targets:raise ValueError('Text denominator differs')
    if _read(root,ref)!=raw:raise ValueError('Pinned campaign changed')
    return Combined62(tuple(sorted(base.bindings+tuple((r.task_key,r.train_binding()) for r in routes))),expected,sources,evaluators)
