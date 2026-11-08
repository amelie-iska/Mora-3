"""Two original Rfam task axes and one GOA axis; no native evaluation roles."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
from mora3.data.admet import digest
from mora3.data.finite_union_native_rows import full9
from mora3.data.finite_union_inverse_literal_filter import load_inverse_literal_filtered_native_view
from mora3.data.annotation_axes import read_annotation_axes
from mora3.data.goa_streamed import StreamedGOASourcePool
from mora3.model.source_assembly_inputs import PINS,CURRENT_VIEW_PINS
from mora3.research.primary_campaign_assembly import _reference,_read
from mora3.research.rfam_campaign_routes import RfamCampaignAdapter,VIEW as RFAM_VIEW
from mora3.research.goa_campaign_route import GOACampaignAdapter,TASK as GOA_TASK
from mora3.research.mup_campaign import NativeTrainRoutePlan,executable_adapter_sha256
from mora3.research.primary_native_routes import finite_view_policy_sha256

FILTER=('outputs/data-intake/readiness-20261106-corpus-continuation/current32-effective-inverse-target-fragment-filter-v2/effective-filter-descriptor.json','2ac36e62d203f85a5db8f8a17d28bfabe0db2eba94f069f095262d3fa5ad5aba')
GOA_VIEW='c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'
RFAM_KEYS=tuple('Rfam_source_family_'+d+'_prior' for d in ('generation','prediction'))
KEYS=tuple(sorted(RFAM_KEYS+(GOA_TASK,)))

@dataclass(frozen=True)
class RfamBudget:
    batch_size:int
    max_windows:int
    def __post_init__(self):
        if any(type(x) is not int or x<1 for x in (self.batch_size,self.max_windows)):raise ValueError('Explicit positive Rfam budgets required')

@dataclass(frozen=True)
class GOABudget:
    max_windows:int
    max_cached_kb_parts:int
    def __post_init__(self):
        if any(type(x) is not int or x<1 for x in (self.max_windows,self.max_cached_kb_parts)):raise ValueError('Explicit positive GOA window/cache budgets required')

@dataclass(frozen=True)
class PartialRfamGOARegistry:
    bindings:tuple
    plans:tuple
    budget_binding_sha256:str
    availability:dict
    asset_pins:tuple
    @property
    def native_source_axes(self):return 3
    @property
    def executable_keys(self):return 3
    @property
    def complete_campaign(self):return False
    @property
    def training_authorized(self):return False
    @property
    def allocation_authorized(self):return False
    @property
    def unresolved(self):return ('No native validation/test rows or evaluation adapter in this scope; train-only dispositions required',
        'Rfam homology/parent-alias/taxonomy heldout obligations remain unresolved; no global leakage-safe claim',
        'GOA exact current canonical sequence and observed sign supervision only; unknown/conflict masks and ontology nonpropagation preserved',
        'Full-owner snapshot, runtime qualification and finite-pass capacity remain required')

def checked_budgets(rfam_budgets,goa_budget):
    if type(rfam_budgets) is not dict or set(rfam_budgets)!=set(RFAM_KEYS):raise ValueError('Both exact Rfam direction budgets required')
    for b in rfam_budgets.values():
        if type(b) is not RfamBudget:raise ValueError('Exact typed Rfam budget required')
        b.__post_init__()
    if type(goa_budget) is not GOABudget:raise ValueError('Exact typed GOA budget required')
    goa_budget.__post_init__()

def load_original_rfam_goa_assets(root,*,goa_budget):
    if type(goa_budget) is not GOABudget:raise ValueError('Exact GOA resource budget required')
    goa_budget.__post_init__();root=Path(root).resolve()
    pins=dict(inverse_filter=FILTER,base_filter=CURRENT_VIEW_PINS['effective_source_view'],census=CURRENT_VIEW_PINS['effective_axis_census'],
        rfam=PINS['rfam_manifest'],axes=PINS['annotation_axes'],canonical=PINS['goa_canonical'],ontology=PINS['goa_ontology'])
    refs={k:_reference(dict(path=p,sha256=h)) for k,(p,h) in pins.items()};raws={k:_read(root,r) for k,r in refs.items()}
    census=json.loads(raws['census']);rfam=json.loads(raws['rfam'])
    selected=[r for r in census['task_schema_role_routes'] if r['source_family'] in ('Rfam15.1-source-family-prior','UniProt-GOA-assertion-packets')]
    expected={RFAM_KEYS[0]:18423,RFAM_KEYS[1]:18423,GOA_TASK:106892}
    if (len(selected)!=3 or {r['task_name']:r['retained_rows'] for r in selected}!=expected
            or any(r['split']!='train' or r['native_schema']!='mora3.CorpusTask/v1' for r in selected)
            or rfam['split_counts']!={'train':36846,'validation':0,'test':0}):raise ValueError('Exact three original train-only source axes required')
    rv=load_inverse_literal_filtered_native_view(FILTER[0],FILTER[1],root=root)
    axis=read_annotation_axes(root/refs['axes'].path,expected_sha256=refs['axes'].sha256)
    pool=StreamedGOASourcePool(root=root,native_filter_path=refs['base_filter'].path,native_filter_sha256=refs['base_filter'].sha256,
        registry=axis,canonical_reference_manifest=refs['canonical'].path,canonical_reference_sha256=refs['canonical'].sha256,
        ontology_manifest=refs['ontology'].path,ontology_sha256=refs['ontology'].sha256,
        max_batch_size=1,max_cached_kb_parts=goa_budget.max_cached_kb_parts,native_filter_policy='combined-study-overlay-v2')
    if finite_view_policy_sha256(rv)!=RFAM_VIEW or finite_view_policy_sha256(pool.native_view)!=GOA_VIEW:raise ValueError('Exact per-route native view policies required')
    availability=dict(distinct_native_source_axes=3,executable_keys=3,train_rows_by_task=expected,validation_rows=0,test_rows=0,
        rfam_heldout_obligations=rfam['homology_parent_aliases_taxonomy_heldout_unresolved'],rfam_global_leakage_safe=rfam['global_leakage_safe'])
    return rv,pool,availability,refs,raws

def _construct(root,rv,pool,snapshot,rfam_budgets,goa_budget):
    checked_budgets(rfam_budgets,goa_budget);pairs=[]
    for direction in ('generation','prediction'):
        key='Rfam_source_family_'+direction+'_prior';b=rfam_budgets[key]
        adapter=RfamCampaignAdapter(rv,direction,snapshot,b.batch_size)
        pairs.append((adapter.train_binding(),b.max_windows))
    pairs.append((GOACampaignAdapter(pool,snapshot).train_binding(),goa_budget.max_windows))
    bindings=[];plans=[]
    for binding,windows in sorted(pairs,key=lambda x:x[0].route.task_key):
        key=binding.route.task_key
        plans.append(NativeTrainRoutePlan(key,digest(binding.route.initial_cursor),executable_adapter_sha256(binding.prepare,root=root),windows))
        bindings.append((key,binding))
    for _,b in bindings:
        descriptors={d['source_shard']['path']:d for d in b.route._view.descriptors}
        if any(full9(root/p)!=tuple(descriptors[p]['source_full9']) for p in b.route._paths):raise ValueError('Original Rfam/GOA source changed')
    return tuple(bindings),tuple(plans)

def build_rfam_goa_registry(root,*,source_snapshot,rfam_budgets,goa_budget):
    checked_budgets(rfam_budgets,goa_budget)
    root=Path(root).resolve();snapshot=_reference(source_snapshot);before=_read(root,snapshot)
    rv,pool,availability,refs,raws=load_original_rfam_goa_assets(root,goa_budget=goa_budget)
    bindings,plans=_construct(root,rv,pool,snapshot.sha256,rfam_budgets,goa_budget)
    if _read(root,snapshot)!=before or any(_read(root,r)!=raws[k] for k,r in refs.items()):raise ValueError('Pinned Rfam/GOA assembly assets changed')
    from mora3.data.admet import canonical
    budget_sha=digest(canonical(dict(rfam={k:asdict(v) for k,v in rfam_budgets.items()},goa=asdict(goa_budget),fixed_goa_batch_size=1)))
    refs['source_snapshot']=snapshot
    return PartialRfamGOARegistry(bindings,plans,budget_sha,availability,tuple(sorted(refs.items())))
