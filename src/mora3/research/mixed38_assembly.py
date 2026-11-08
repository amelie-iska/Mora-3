"""Exact scoped composition for seven native source factories."""
from dataclasses import dataclass
from pathlib import Path
import json
from mora3.data.admet import digest,canonical
from mora3.research.primary_campaign_assembly import _reference,_read
from mora3.research.mup_campaign import FrozenPrimaryProxyCampaign,NativeTrainRoutePlan,executable_adapter_sha256
from mora3.research.primary_native_routes import NativeRouteBinding
from mora3.research import native_prior_registry as prior,reactzyme_route_factory as reaction,finite_assay_registry as assay,pdb_route_factory as pdb
from mora3.research.finite_view_compatibility import BASE_POLICY,INVERSE_POLICY,SUPPORTED as BASE_AXES,INVERSE,FILTER,verify_exact_view_compatibility
from mora3.research import uniprot_full_record_route_factory as uniprot

from mora3.research import native_annotation_registry as annotation
from mora3.research import rfam_goa_registry as rg

POLICIES={**{k:BASE_POLICY for k in BASE_AXES},pdb.TASK:BASE_POLICY,**{k:INVERSE_POLICY for k in uniprot.ROUTES},**{k:BASE_POLICY for k in annotation.ROUTES},**{k:rg.RFAM_VIEW for k in rg.RFAM_KEYS},rg.GOA_TASK:BASE_POLICY}
AXES=tuple(sorted(POLICIES))

@dataclass(frozen=True)
class Mixed38Routes:
    bindings:tuple
    plans:tuple
    per_route_view_policies:tuple
    validation_compatibility:dict
    @property
    def complete_campaign(self):return False
    @property
    def allocation_authorized(self):return False
    @property
    def training_authorized(self):return False

def _merge(root,plans,groups,*,snapshot):
    if len(AXES)!=38 or type(plans) is not tuple or tuple(p.task_key for p in plans)!=AXES:
        raise ValueError('Exact sorted unique 38-key campaign plan required')
    actual={};observed={}
    for bindings,group_plans in groups:
        if type(bindings) is not tuple or type(group_plans) is not tuple or len(bindings)!=len(group_plans):
            raise ValueError('Exact paired bindings and plans required')
        for (key,binding),plan in zip(bindings,group_plans,strict=True):
            if key in actual or key not in POLICIES:raise ValueError('Duplicate or unsupported mixed-view route')
            if (type(binding) is not NativeRouteBinding or type(plan) is not NativeTrainRoutePlan
                    or key!=plan.task_key or binding.route.task_key!=key
                    or binding.route.source_snapshot_sha256!=snapshot
                    or binding.route.native_view_policy_sha256!=POLICIES[key]):
                raise ValueError('Exact per-route identity, snapshot or mandatory inverse guard differs')
            binding.__post_init__();plan.__post_init__()
            if digest(binding.route.initial_cursor)!=plan.initial_cursor_sha256 or executable_adapter_sha256(binding.prepare,root=root)!=plan.adapter_sha256:
                raise ValueError('Live cursor/executable binding differs')
            actual[key]=binding;observed[key]=plan
    if set(actual)!=set(AXES) or tuple(observed[k] for k in AXES)!=plans:
        raise ValueError('Exact full scoped axis/cursor/adapter/window-budget plan differs')
    return tuple((k,actual[k]) for k in AXES)

def _rfam_goa_budget_digest(bindings,plans):
    adapters={k:b.prepare.__self__ for k,b in bindings}
    windows={p.task_key:p.max_windows for p in plans}
    pool=adapters[rg.GOA_TASK].pool
    if pool.max_batch_size!=1:raise ValueError('Fixed single-row GOA budget changed')
    return digest(canonical(dict(rfam={k:dict(batch_size=adapters[k].batch_size,max_windows=windows[k]) for k in rg.RFAM_KEYS},
        goa=dict(max_windows=windows[rg.GOA_TASK],max_cached_kb_parts=pool.max_cached_kb_parts),fixed_goa_batch_size=1)))

def bind_mixed38_results(root,*,campaign,campaign_reference,source_snapshot,
        prior_result,reactzyme_result,assay_result,pdb_result,uniprot_result,annotation_result,rfam_goa_result,pdb_resource_binding_sha256,rfam_goa_budget_binding_sha256):
    """Consume existing checked factories; require exact pending Tox21 coordinates.

    This delta does not load or invent the missing production campaign, snapshot,
    reference losses or measured owner resources. Caller constructs factories
    with explicit budgets first; their exact plans must equal the pinned campaign.
    """
    results=(prior_result,reactzyme_result,assay_result,pdb_result,uniprot_result,annotation_result,rfam_goa_result)
    kinds=(prior.PartialNativePriorRegistry,reaction.PartialReactZymeRegistry,assay.PartialAssayRegistry,pdb.PartialPDBRegistry,uniprot.PartialFullRecordRegistry,annotation.PartialAnnotationRegistry,rg.PartialRfamGOARegistry)
    if any(type(x) is not k for x,k in zip(results,kinds,strict=True)) or type(campaign) is not FrozenPrimaryProxyCampaign:
        raise ValueError('Exact seven source-factory result types and frozen campaign required')
    campaign.__post_init__()
    if (campaign.validation.view_policy_sha256!=INVERSE_POLICY
            or tuple(t.task_key for t in campaign.validation.tasks)!=(INVERSE,)):
        raise ValueError('Only exact enrolled inverse-filter Tox21 validation is supported by this delta')
    root=Path(root).resolve();cr=_reference(campaign_reference);sr=_reference(source_snapshot)
    craw=_read(root,cr);sraw=_read(root,sr)
    if cr.sha256!=campaign.sha256 or craw!=campaign.wire() or sr.sha256!=campaign.source_snapshot_sha256:
        raise ValueError('Pinned campaign and full source identity differ')
    p,r,a,d,u,n,g=results
    if (rfam_goa_budget_binding_sha256!=g.budget_binding_sha256 or
            _rfam_goa_budget_digest(g.bindings,g.plans)!=g.budget_binding_sha256):
        raise ValueError("Exact external Rfam/GOA cache/window budget binding required")
    if pdb_resource_binding_sha256!=d.resource_binding_sha256 or pdb._resource_binding(d.binding.prepare.__self__.axis,d.source_availability,d.budget)!=pdb_resource_binding_sha256:
        raise ValueError('Exact explicitly supplied PDB resource binding required')
    groups=((p.bindings,p.plans),(((r.plan.task_key,r.binding),),(r.plan,)),(a.bindings,a.plans),(((d.plan.task_key,d.binding),),(d.plan,)),(u.bindings,u.plans),(n.bindings,n.plans),(g.bindings,g.plans))
    bindings=_merge(root,campaign.train_routes,groups,snapshot=sr.sha256)
    witness=verify_exact_view_compatibility(root,train_policy=BASE_POLICY,validation_policy=INVERSE_POLICY,
        train_axes=BASE_AXES,validation_axes=(INVERSE,),validation_tasks=campaign.validation.tasks)
    filter_doc=json.loads(_read(root,_reference(dict(path=FILTER[0],sha256=FILTER[1]))))
    ref=filter_doc['source_inverse_literal_guard_proposal'];proposal=json.loads(_read(root,_reference({k:ref[k] for k in ('path','sha256')})))
    candidate,=proposal['candidates'];coordinate=(candidate['original_shard']['path'],candidate['absolute_original_row_ordinal'])
    if coordinate[0] in d.binding.route._paths:raise ValueError('PDB source unexpectedly intersects guarded inverse omission')
    for key,binding in n.bindings+g.bindings:
        if coordinate[0] in binding.route._paths:
            raise ValueError('Added original source intersects inverse omission')
    for key,binding in u.bindings:
        if coordinate not in binding.route._view._ordinal_exclusions:
            raise ValueError('Guarded inverse omission absent from full-record route view')
    pins={}
    for result in results:
        for _,ref in result.asset_pins:
            if ref.path in pins and pins[ref.path]!=ref.sha256:raise ValueError('Conflicting shared asset pin')
            pins[ref.path]=ref.sha256;_read(root,ref)
    if _read(root,cr)!=craw or _read(root,sr)!=sraw:raise ValueError('Pinned assembly artifacts changed')
    return Mixed38Routes(bindings,campaign.train_routes,tuple(sorted(POLICIES.items())),dict(
        base18=witness,pdb_original_source_nonintersection=True,full_record_routes_keep_inverse_guard=True,
        scope='38 executable keys across 27 of 45 native census task axes; six added train-only bindings; no general filter equivalence.',distinct_native_source_axes=27,executable_keys=38,added_train_only_nonintersection=True))
