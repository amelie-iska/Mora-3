"""Original BioGRID chemical, Reactome static and STRING score train factories."""
from dataclasses import dataclass
from pathlib import Path
import json
from mora3.data.admet import canonical,digest
from mora3.data.finite_union_native_rows import full9
from mora3.data.finite_union_annotation_study_filter import load_study_filtered_native_view
from mora3.model.source_assembly_inputs import CURRENT_VIEW_PINS
from .primary_campaign_assembly import _reference,_read
from .primary_native_routes import finite_view_policy_sha256
from .native_annotation_campaign_routes import _SPECS,bind_string,bind_reactome,bind_biogrid_chemical
from .mup_campaign import NativeTrainRoutePlan,executable_adapter_sha256

BASE='outputs/implementation/readiness-20261106-model-mup/'
SPECS={
 'string':('current-string-scores-v1/source-census.json','10c7917c99e7e37ece1dfffaa05688050b3d95b432e3690a39da2d8be897ff69',474656,'complete_current_original_roles',bind_string),
 'reactome':('current-reactome-v1/source-census.json','b2b39a726eac28f3c074470c7eeaa87ecf4e7a8818f281a5c69369a64123c810',1653,'complete_original_roles',bind_reactome),
 'biogrid_chemical':('current-biogrid-chemical-v1/source-census.json','c9c2ac3ff52746665e673e36a7faa4cdc58dfc96fdeb2c2c28c589844d86abaa',2706,'complete_current_original_roles',bind_biogrid_chemical),
}
ROUTES=tuple(sorted(_SPECS[k].key for k in SPECS))
VIEW='c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'

@dataclass(frozen=True)
class AnnotationRouteBudget:
    batch_size:int
    max_windows:int
    def __post_init__(self):
        if any(type(x) is not int or x<1 for x in (self.batch_size,self.max_windows)):
            raise ValueError('Explicit positive annotation route batch/window budgets required')

@dataclass(frozen=True)
class PartialAnnotationRegistry:
    bindings:tuple
    plans:tuple
    budgets:tuple
    source_availability:dict
    asset_pins:tuple
    @property
    def complete_campaign(self):return False
    @property
    def allocation_authorized(self):return False
    @property
    def training_authorized(self):return False
    @property
    def unresolved(self):return ('All three exact sources are native train-only; explicit campaign dispositions required',
        'No heldout/evaluation role created or external heldout clearance inferred',
        'Full graph/sequence, batch and owner runtime resource qualification remains separate',
        'One finite pass; missing fields and original occurrence multiplicities remain unchanged')

def checked_budgets(budgets):
    if type(budgets) is not dict or set(budgets)!=set(ROUTES):raise ValueError('Exactly three supported annotation route keys and budgets required')
    for budget in budgets.values():
        if type(budget) is not AnnotationRouteBudget:raise ValueError('Exact typed annotation budget required')
        budget.__post_init__()
    return tuple(sorted(budgets.items()))

def _fingerprints(root,view,sources):
    descriptors={d['source_shard']['path']:d for d in view.descriptors}
    paths={p for source in sources.values() for p in source.source_paths}
    if any(p not in descriptors or full9(root/p)!=tuple(descriptors[p]['source_full9']) for p in paths):
        raise ValueError('Original native annotation source fingerprint changed')

def load_original_annotation_assets(root):
    root=Path(root).resolve();p,h=CURRENT_VIEW_PINS['effective_source_view']
    refs={'view':_reference(dict(path=p,sha256=h))}
    for kind,(path,sha,*_) in SPECS.items():refs[kind]=_reference(dict(path=BASE+path,sha256=sha))
    raws={k:_read(root,r) for k,r in refs.items()}
    view=load_study_filtered_native_view(root/refs['view'].path,refs['view'].sha256,root=root)
    if finite_view_policy_sha256(view)!=VIEW:raise ValueError('Exact current annotation view required')
    sources={};availability={}
    for kind,(_,_,count,role_field,_) in SPECS.items():
        ref=refs[kind];proof=json.loads(raws[kind])
        if (proof[role_field]!={'train':count,'validation':0,'test':0}
                or canonical(json.loads(_read(root,_reference({k:proof['effective_filter'][k] for k in ('path','sha256')}))))!=canonical(view.effective_filter)):
            raise ValueError('Exact guarded original train-only role census required')
        source=_SPECS[kind].source_type(view,ref.path,ref.sha256)
        if source.axes.native_view_policy_sha256!=VIEW:raise ValueError('Source axes differ from pinned view')
        sources[kind]=source
        availability[_SPECS[kind].key]=dict(source_family=_SPECS[kind].family,native_task=_SPECS[kind].task,
            train_rows=count,validation_rows=0,test_rows=0,source_paths=len(source.source_paths),source_axes=source.axes.binding())
    _fingerprints(root,view,sources)
    return view,sources,availability,refs,raws

def _construct(root,view,sources,snapshot,ordered):
    by_key={_SPECS[k].key:k for k in SPECS};bindings=[];plans=[]
    for key,budget in ordered:
        kind=by_key[key];builder=SPECS[kind][4]
        binding=builder(sources[kind],view,snapshot_sha256=snapshot,batch_size=budget.batch_size)
        if binding.route.task_key!=key:raise ValueError('Exact requested native annotation task differs')
        plans.append(NativeTrainRoutePlan(key,digest(binding.route.initial_cursor),executable_adapter_sha256(binding.prepare,root=root),budget.max_windows))
        bindings.append((key,binding))
    return tuple(bindings),tuple(plans)

def build_original_annotation_registry(root,*,source_snapshot,expected_view_policy_sha256,budgets):
    ordered=checked_budgets(budgets)
    if expected_view_policy_sha256!=VIEW:raise ValueError('Exact source-qualified annotation view required')
    root=Path(root).resolve();snapshot=_reference(source_snapshot);before=_read(root,snapshot)
    view,sources,availability,refs,raws=load_original_annotation_assets(root)
    bindings,plans=_construct(root,view,sources,snapshot.sha256,ordered)
    _fingerprints(root,view,sources)
    if _read(root,snapshot)!=before or any(_read(root,r)!=raws[k] for k,r in refs.items()):raise ValueError('Pinned native annotation assets changed')
    refs['source_snapshot']=snapshot
    return PartialAnnotationRegistry(bindings,plans,ordered,availability,tuple(sorted(refs.items())))
