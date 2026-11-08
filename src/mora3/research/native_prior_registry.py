"""Seven native-prior factories only; never full campaign or owner admission."""
from dataclasses import dataclass
from pathlib import Path
import json

from mora3.data.admet import canonical, digest
from mora3.data.finite_union_native_rows import full9
from mora3.data.finite_union_annotation_study_filter import load_study_filtered_native_view
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.model.source_assembly_inputs import PINS, CURRENT_VIEW_PINS
from .native_prior_campaign import NativePriorCampaignAdapter, _SOURCE_ROUTES
from .primary_native_routes import finite_view_policy_sha256
from .primary_campaign_assembly import _reference, _read
from .mup_campaign import NativeTrainRoutePlan, executable_adapter_sha256

ROUTES=tuple(sorted(_SOURCE_ROUTES))

@dataclass(frozen=True)
class PriorBudget:
    batch_size: int
    max_total_target_bytes: int
    chunk_size: int
    max_windows: int

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in (self.batch_size,self.max_total_target_bytes,self.chunk_size,self.max_windows)):
            raise ValueError('Explicit positive integer per-route budgets required')

@dataclass(frozen=True)
class PartialNativePriorRegistry:
    bindings: tuple
    plans: tuple
    asset_pins: tuple
    source_snapshot_sha256: str
    native_view_policy_sha256: str

    @property
    def complete_campaign(self): return False

    @property
    def allocation_authorized(self): return False

    @property
    def training_authorized(self): return False

def checked_budgets(budgets):
    if type(budgets) is not dict or set(budgets)!=set(ROUTES):
        raise ValueError('Exactly seven supported native-prior axes and all budgets required')
    for budget in budgets.values():
        if type(budget) is not PriorBudget:
            raise ValueError('Exact typed per-route budget required')
        budget.__post_init__()
    return tuple((route,budgets[route]) for route in ROUTES)

def _construct(root, view, vocabulary, snapshot_sha256, budgets):
    """Internal assembly; public entry first verifies assets and immutable view."""
    bindings=[];plans=[]
    for route,budget in budgets:
        adapter=NativePriorCampaignAdapter(view,vocabulary,snapshot_sha256,route,
            budget.max_total_target_bytes,budget.chunk_size)
        binding=adapter.train_binding(batch_size=budget.batch_size)
        plan=NativeTrainRoutePlan(binding.route.task_key,digest(binding.route.initial_cursor),
            executable_adapter_sha256(binding.prepare,root=root),budget.max_windows)
        bindings.append((plan.task_key,binding));plans.append(plan)
    return tuple(bindings),tuple(plans)

def build_native_prior_registry(root, *, source_snapshot, expected_view_policy_sha256, budgets):
    """Construct metadata-only bindings, no source iteration or model allocation.

    source_snapshot is an external exact {path,sha256} full-owner snapshot
    reference, not an invented snapshot or an admission assertion. The caller
    must separately validate its semantics via the campaign construction path.
    Only the live pinned v2 view and source-fitted vocabulary are accepted.
    Single finite passes only; no implicit repeat schedule or budget defaults.
    """
    ordered=checked_budgets(budgets)  # refuse missing axes before filesystem I/O
    root=Path(root).resolve()
    snapshot=_reference(source_snapshot)
    refs=dict(source_snapshot=snapshot)
    for name,pin in (('effective_source_view',CURRENT_VIEW_PINS['effective_source_view']),
                     ('payload_vocabulary',PINS['payload_vocabulary']),('payload_fit',PINS['payload_fit'])):
        refs[name]=_reference(dict(path=pin[0],sha256=pin[1]))
    raw={name:_read(root,ref) for name,ref in refs.items()}
    view_ref=refs['effective_source_view']
    view=load_study_filtered_native_view(root/view_ref.path,view_ref.sha256,root=root)
    fit=json.loads(raw['payload_fit'])
    if (fit.get('status')!='complete-source-fit' or fit.get('train_views')!=602
            or fit.get('target_fields_used') is not False or fit.get('heldout_rows_used') is not False
            or fit.get('payload_vocabulary_artifact',{}).get('sha256')!=refs['payload_vocabulary'].sha256):
        raise ValueError('Exact original source-fitted vocabulary proof required')
    if (canonical(getattr(view,'effective_filter',None))!=canonical(json.loads(raw['effective_source_view']))
            or finite_view_policy_sha256(view)!=expected_view_policy_sha256
            or tuple(canonical(d) for d in view.descriptors)!=view._sealed_descriptors
            or view._sealed_by_path!={d['source_shard']['path']:s for d,s in zip(view.descriptors,view._sealed_descriptors,strict=True)}):
        raise ValueError('Pinned effective view or sealed source descriptors changed')
    # Native payload hashes remain checked by the existing reader at consumption.
    # The original metadata identity is checked before returning any binding.
    identities={(family,task,revision) for family,task,revision in _SOURCE_ROUTES.values()}
    relevant=[]
    for d in view.descriptors:
        if any((r['source_family'],r['task_name'],r['source_revision']) in identities
               and r['split']=='train' and r['eligible_view_disposition']==0 for r in d['source_task_schema_counts']):
            p=root/d['source_shard']['path']
            if full9(p)!=tuple(d['source_full9']):
                raise ValueError('Original native source fingerprint changed')
            relevant.append((p,tuple(d['source_full9'])))
    vocabulary=AnnotationVocabulary(raw['payload_vocabulary'].decode('utf8'))
    bindings,plans=_construct(root,view,vocabulary,snapshot.sha256,ordered)
    for name,ref in refs.items():
        if _read(root,ref)!=raw[name]:raise ValueError('Pinned asset changed during construction')
    if any(full9(path)!=stamp for path,stamp in relevant):
        raise ValueError('Original source changed during construction')
    return PartialNativePriorRegistry(bindings,plans,tuple(sorted(refs.items())),snapshot.sha256,expected_view_policy_sha256)
