"""Finite59 plus scoped MCSA; exact Tox21/MCSA/FreeSolv validation, no admission."""
from dataclasses import dataclass,replace,asdict
from pathlib import Path
import json
from mora3.data.admet import canonical,digest
from mora3.research import finite59_assembly as finite
from mora3.research.primary_campaign_assembly import _read,_reference
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.mup_campaign import FrozenPrimaryProxyCampaign,executable_adapter_sha256
from mora3.research.matched_validation import ValidationSourceRow,matched_validation_source_policy_sha256
from mora3.research.tox21_inverse_validation_binding import bind_pending_inverse_validation
from mora3.research.assay_campaign_routes import FreeSolvCampaignAdapter
from mora3.research import mcsa_campaign_bridge as mcsa

FREESOLV=('outputs/implementation/readiness-20261108-freesolv-exact-pending-validation-v2/receipt.json','018ebb67aff53ec86757d0dc7231cebaf7a742438ccccb72382ca688261f8358')
FREE=FreeSolvCampaignAdapter.task_key
@dataclass(frozen=True)
class PendingThreeEvaluators:
    sources:object
    evaluators:tuple
    hashes:tuple
    coordinates:tuple
    denominators:tuple
    units:tuple
    @property
    def training_authorized(self):return False

@dataclass(frozen=True)
class Combined60:
    bindings:tuple
    plans:tuple
    validation:PendingThreeEvaluators
    @property
    def training_authorized(self):return False
    @property
    def allocation_authorized(self):return False
    @property
    def complete_campaign(self):return False

def prepare_three_evaluators(root,*,finite_inputs,mcsa_bridge,read_batch_size,max_selected_bytes):
    root=Path(root).resolve()
    if type(mcsa_bridge) is not mcsa.MCSABridge:raise ValueError('Exact scoped MCSA bridge required')
    if any(type(x) is not int or x<1 for x in (read_batch_size,max_selected_bytes)):raise ValueError('Explicit validation budgets required')
    snapshot=_reference(finite_inputs['source_snapshot']);_read(root,snapshot)
    route=mcsa_bridge.binding.route
    if mcsa_bridge.validation_sources.mcsa is not route or mcsa_bridge.plan.task_key!=route.task_key:
        raise ValueError('Same exact MCSA train/evaluator route required')
    if route.source_snapshot_sha256!=snapshot.sha256:raise ValueError('Common exact source snapshot required')
    # Revalidate MCSA coordinates through authoritative source reader.
    coords,count=route.validation_enrollment(max_selected_bytes=max_selected_bytes)
    if coords!=mcsa_bridge.coordinates or count!=mcsa_bridge.observed_targets:raise ValueError('MCSA enrollment changed')
    pending=bind_pending_inverse_validation(root,read_batch_size=read_batch_size,max_selected_arrow_bytes=max_selected_bytes)
    tox=replace(pending.adapter,source_snapshot_sha256=snapshot.sha256)
    f=dict(finite_inputs['remaining_result'].bindings)[FREE].prepare.__self__
    if type(f) is not FreeSolvCampaignAdapter or f.source_snapshot_sha256!=snapshot.sha256:raise ValueError('Exact common FreeSolv train/eval adapter required')
    ref=_reference(dict(path=FREESOLV[0],sha256=FREESOLV[1]));raw=_read(root,ref);receipt=json.loads(raw)
    cref=_reference({k:receipt['coordinates'][k] for k in ('path','sha256')});freecoords=tuple(ValidationSourceRow(**r) for r in json.loads(_read(root,cref)))
    if (receipt['schema']!='Mora3.FreeSolvExactPendingValidation/v1' or receipt['role']!='validation'
        or receipt['initial_reference_loss'] is not None or receipt['production_evaluation_enrolled'] is not False
        or len(freecoords)!=42 or receipt['observed_reported_experimental_targets']!=42):raise ValueError('Exact pending FreeSolv authority scope required')
    scoped=replace(f,source_snapshot_sha256=receipt['scoped_source']['sha256'])
    if (canonical(scoped.executable_binding())!=canonical(receipt['exact_adapter_binding'])
        or executable_adapter_sha256(scoped.evaluate_validation,root=root)!=receipt['validation_adapter_sha256']):raise ValueError('Exact FreeSolv receipt executable/source binding changed')
    # Runtime has one original finite validation view. Verify each FreeSolv source
    # descriptor and original exclusion set independently in both exact views.
    view=mcsa_bridge.validation_sources.original
    for path in {c.source_path for c in freecoords}:
        if (view._sealed_by_path.get(path)!=f.current_view._sealed_by_path.get(path)
            or {x for x in view._ordinal_exclusions if x[0]==path}!={x for x in f.current_view._ordinal_exclusions if x[0]==path}):raise ValueError('FreeSolv exact base/inverse validation coordinates differ')
    if finite_view_policy_sha256(view)!=finite.INVERSE_POLICY:raise ValueError('Exact guarded original validation view required')
    evaluators=tuple(sorted(((finite.INVERSE,tox.evaluate_validation),(FREE,f.evaluate_validation),(route.task_key,route.evaluate_validation))))
    units=tuple(sorted(((finite.INVERSE,(tox.loss_unit,tox.target_unit)),(FREE,(f.loss_unit,f.target_unit)),(route.task_key,(route.loss_unit,route.target_unit)))))
    return PendingThreeEvaluators(mcsa_bridge.validation_sources,evaluators,
        tuple((k,executable_adapter_sha256(e,root=root)) for k,e in evaluators),
        tuple(sorted(((finite.INVERSE,pending.coordinates),(FREE,freecoords),(route.task_key,coords)))),
        tuple(sorted(((finite.INVERSE,5225),(FREE,42),(route.task_key,2043)))),units)

def bind_combined60(root,*,finite_inputs,mcsa_bridge,campaign,campaign_reference):
    root=Path(root).resolve()
    finite_result=finite.bind_finite59_results(root,**finite_inputs)
    if type(campaign) is not FrozenPrimaryProxyCampaign:raise ValueError('Exact frozen combined campaign required')
    campaign.__post_init__();ref=_reference(campaign_reference);raw=_read(root,ref)
    if raw!=campaign.wire() or ref.sha256!=campaign.sha256:raise ValueError('Exact pinned combined campaign required')
    previous=finite_inputs['campaign'];route=mcsa_bridge.binding.route
    if campaign.source_snapshot_sha256!=previous.source_snapshot_sha256:raise ValueError('Combined source identity differs')
    # Unchanged resource/width/optimizer identities; only explicit task/cohort composition differs.
    for name in ('width_family_sha256','proxy_width','proxy_parameter_count','source_role_schedule_sha256','complete_owner_evidence_sha256','runtime_profile_sha256','existing_worker_sha256','gradient_clip_max_norm','decay_aliases','max_one_update_seconds','max_matched_validation_seconds','validation_aggregation'):
        if getattr(campaign,name)!=getattr(previous,name):raise ValueError('Combined campaign policy changed: '+name)
    if (digest(route.initial_cursor)!=mcsa_bridge.plan.initial_cursor_sha256 or executable_adapter_sha256(mcsa_bridge.binding.prepare,root=root)!=mcsa_bridge.plan.adapter_sha256):raise ValueError('MCSA cursor/executable changed')
    plans=tuple(sorted(finite_result.plans+(mcsa_bridge.plan,),key=lambda p:p.task_key))
    if campaign.train_routes!=plans:raise ValueError('Exact sixty route plans and capacities required')
    from mora3.research.mup_sweep_selection import MAX_UPDATES
    route.require_task_cycle_capacity(campaign.task_cycle,MAX_UPDATES)
    pending=prepare_three_evaluators(root,finite_inputs=finite_inputs,mcsa_bridge=mcsa_bridge,read_batch_size=campaign.validation.read_batch_size,max_selected_bytes=campaign.validation.max_selected_arrow_bytes)
    if campaign.validation.view_policy_sha256!=matched_validation_source_policy_sha256(pending.sources) or campaign.validation_adapter_sha256!=pending.hashes:raise ValueError('Exact mixed validation source and evaluators required')
    tasks={t.task_key:t for t in campaign.validation.tasks}
    if set(tasks)!=set(dict(pending.coordinates)):raise ValueError('Exactly Tox21, FreeSolv and MCSA validation tasks required')
    if tasks[finite.INVERSE]!=previous.validation.tasks[0]:raise ValueError('Existing exact Tox21 task/reference changed')
    for key,coords in pending.coordinates:
        task=tasks[key]
        if task.coordinates!=coords or task.observed_targets!=dict(pending.denominators)[key] or (task.loss_unit,task.target_unit)!=dict(pending.units)[key]:raise ValueError('Exact independent validation coordinate/denominator/unit differs')
        batches=campaign.validation.batches(pending.sources,key)
        evaluator=dict(pending.evaluators)[key]
        if key==route.task_key:observed=sum(b.observed_targets for b in batches)
        else:observed=sum(evaluator.__self__.validation_observed_targets(b) for b in batches)
        if observed!=task.observed_targets:raise ValueError('Actual native validation target count differs')
    if _read(root,ref)!=raw:raise ValueError('Pinned combined campaign changed')
    return Combined60(tuple(sorted(finite_result.bindings+((route.task_key,mcsa_bridge.binding),))),plans,pending)
