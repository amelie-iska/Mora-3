"""Exact 18-axis metadata assembly. Does not establish full-owner completeness."""
from dataclasses import dataclass
from pathlib import Path
from mora3.data.admet import digest
from .mup_campaign import FrozenPrimaryProxyCampaign,NativeTrainRoutePlan,executable_adapter_sha256
from .primary_native_routes import NativeRouteBinding
from .primary_campaign_assembly import _reference,_read
from . import native_prior_registry as prior
from . import reactzyme_route_factory as reaction
from . import finite_assay_registry as assay
from .finite_view_compatibility import resolve_view_compatibility, require_policy_scope, INVERSE

SUPPORTED=tuple(sorted(tuple('native-prior:'+r for r in prior.ROUTES)+(reaction.TASK,)+tuple(assay.ROUTES)))

@dataclass(frozen=True)
class ScopedRuntimeRoutes:
    campaign_sha256:str
    source_snapshot_sha256:str
    view_policy_sha256:str
    bindings:tuple
    plans:tuple
    factory_results:tuple
    validation_view_policy_sha256:str
    view_compatibility:dict | None
    @property
    def schema(self):return 'Mora3.Scoped18RuntimeRoutes/v2'
    @property
    def complete_campaign(self):return False
    @property
    def allocation_authorized(self):return False
    @property
    def training_authorized(self):return False
    @property
    def unresolved(self):return ('complete_owner_axis_authority','validation_registry_and_cohort','runtime_and_owner_admission')

def _check_axes(plans):
    if type(plans) is not tuple or any(type(p) is not NativeTrainRoutePlan for p in plans):
        raise ValueError('Exact frozen native route plans required')
    keys=tuple(p.task_key for p in plans)
    if len(SUPPORTED)!=18 or len(set(SUPPORTED))!=18 or keys!=SUPPORTED:
        raise ValueError('Exactly 18 unique sorted supported planned axes required; missing, duplicate or unsupported axis')
    for p in plans:p.__post_init__()

def _merge_exact(root,planned,groups,*,snapshot,view_policy):
    _check_axes(planned)
    actual={};actual_plans={}
    for bindings,plans in groups:
        if type(bindings) is not tuple or type(plans) is not tuple or len(bindings)!=len(plans):
            raise ValueError('Exact paired factory route bindings and plans required')
        for pair,plan in zip(bindings,plans,strict=True):
            if type(pair) is not tuple or len(pair)!=2:raise ValueError('Exact route binding pair required')
            key,binding=pair
            if key in actual:raise ValueError('Duplicate runtime route axis')
            if (type(binding) is not NativeRouteBinding or type(plan) is not NativeTrainRoutePlan
                    or key!=plan.task_key or binding.route.task_key!=key
                    or binding.route.source_snapshot_sha256!=snapshot
                    or binding.route.native_view_policy_sha256!=view_policy):
                raise ValueError('Runtime route identity/snapshot/view differs')
            binding.__post_init__();plan.__post_init__()
            if (digest(binding.route.initial_cursor)!=plan.initial_cursor_sha256
                    or executable_adapter_sha256(binding.prepare,root=root)!=plan.adapter_sha256):
                raise ValueError('Live cursor or executable adapter differs from factory plan')
            actual[key]=binding;actual_plans[key]=plan
    if set(actual)!=set(SUPPORTED):raise ValueError('Incomplete or unsupported runtime axis union')
    if tuple(actual_plans[k] for k in SUPPORTED)!=planned:
        raise ValueError('Exact campaign planned cursor/adapter/window budget differs')
    return tuple((k,actual[k]) for k in SUPPORTED)

def assemble_scoped_runtime_routes(root,*,campaign,campaign_reference,source_snapshot,
        expected_view_policy_sha256,prior_budgets,reactzyme_budget,assay_budgets):
    """Pinned external campaign plus explicit budgets; no discovery or allocation.

    Accepts only this exact 18-axis scope. A broader campaign is refused rather
    than silently returned as a partial mapping. Repeated task-cycle entries
    retain their campaign weighting; duplicate plan/binding axes are refused.
    The caller still owns full-owner authority and validation assembly.
    """
    if type(campaign) is not FrozenPrimaryProxyCampaign:raise ValueError('Exact frozen campaign required')
    campaign.__post_init__();_check_axes(campaign.train_routes)
    validation_axes=tuple(t.task_key for t in campaign.validation.tasks)
    if INVERSE in validation_axes or campaign.validation.view_policy_sha256!=expected_view_policy_sha256:
        require_policy_scope(expected_view_policy_sha256,campaign.validation.view_policy_sha256,SUPPORTED,validation_axes)
    prior.checked_budgets(prior_budgets);assay.checked_budgets(assay_budgets)
    if type(reactzyme_budget) is not reaction.ReactZymeBudget:raise ValueError('Exact ReactZyme budget required')
    reactzyme_budget.__post_init__()
    expected_windows={'native-prior:'+k:v.max_windows for k,v in prior_budgets.items()}
    expected_windows.update({k:v.max_windows for k,v in assay_budgets.items()})
    expected_windows[reaction.TASK]=reactzyme_budget.max_windows
    if any(expected_windows[p.task_key]!=p.max_windows for p in campaign.train_routes):
        raise ValueError('Explicit route window budgets differ from frozen campaign')
    root=Path(root).resolve();campaign_ref=_reference(campaign_reference);snapshot_ref=_reference(source_snapshot)
    if (campaign_ref.sha256!=campaign.sha256 or snapshot_ref.sha256!=campaign.source_snapshot_sha256):
        raise ValueError('Pinned campaign/full source snapshot identity differs')
    campaign_raw=_read(root,campaign_ref);snapshot_raw=_read(root,snapshot_ref)
    if campaign_raw!=campaign.wire():raise ValueError('Pinned campaign bytes differ from declared plan')
    compatibility=resolve_view_compatibility(root,train_policy=expected_view_policy_sha256,
        validation_policy=campaign.validation.view_policy_sha256,train_axes=SUPPORTED,
        validation_axes=validation_axes,validation_tasks=campaign.validation.tasks)
    common=dict(source_snapshot=source_snapshot,expected_view_policy_sha256=expected_view_policy_sha256)
    p=prior.build_native_prior_registry(root,budgets=prior_budgets,**common)
    r=reaction.build_original_reactzyme_registry(root,budget=reactzyme_budget,**common)
    a=assay.build_original_assay_registry(root,budgets=assay_budgets,**common)
    if type(p) is not prior.PartialNativePriorRegistry or type(r) is not reaction.PartialReactZymeRegistry or type(a) is not assay.PartialAssayRegistry:
        raise ValueError('Exact three production factory results required')
    groups=((p.bindings,p.plans),(((r.plan.task_key,r.binding),),(r.plan,)),(a.bindings,a.plans))
    bindings=_merge_exact(root,campaign.train_routes,groups,snapshot=snapshot_ref.sha256,view_policy=expected_view_policy_sha256)
    if _read(root,campaign_ref)!=campaign_raw or _read(root,snapshot_ref)!=snapshot_raw:
        raise ValueError('Pinned campaign or snapshot changed during assembly')
    # Factories may share assets: every repeated key must still bind identical bytes.
    by_path={}
    for result in (p,r,a):
        for _,ref in result.asset_pins:
            if ref.path in by_path and by_path[ref.path]!=ref.sha256:raise ValueError('Conflicting shared source asset pins')
            by_path[ref.path]=ref.sha256
            _read(root,ref)
    return ScopedRuntimeRoutes(campaign.sha256,snapshot_ref.sha256,expected_view_policy_sha256,
        bindings,campaign.train_routes,(p,r,a),campaign.validation.view_policy_sha256,compatibility)
