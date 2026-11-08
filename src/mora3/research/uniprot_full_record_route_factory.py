"""Twelve full-record fields and guarded inverse sequence; no role inference."""
from dataclasses import dataclass
from pathlib import Path
import json
from mora3.data.admet import canonical,digest
from mora3.data.finite_union_inverse_literal_filter import load_inverse_literal_filtered_native_view
from mora3.data.finite_union_native_rows import full9
from mora3.data.uniprot_full_record import FullRecordUniProtSource,FAMILY,TASK
from mora3.data.uniprot_native_annotations import UNIPROT_NATIVE_INVERSE_TASK
from mora3.model.source_assembly_inputs import PINS
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.model.pubchem_decoder import PubChemByteCodec
from .primary_campaign_assembly import _reference,_read
from .primary_native_routes import finite_view_policy_sha256
from .uniprot_full_record_campaign_route import UniProtFullRecordCampaignAdapter
from .uniprot_campaign_routes import UniProtNativeCampaignAdapter
from .mup_campaign import NativeTrainRoutePlan,executable_adapter_sha256

CENSUS=('outputs/implementation/readiness-20261106-model-mup/uniprot-full-record-v1/source-census.json','e6e8add74b205252b8cd6d7fdad97480b0c253b8a518f0dc38e80886da8bfd8b')
FILTER=('outputs/data-intake/readiness-20261106-corpus-continuation/current32-effective-inverse-target-fragment-filter-v2/effective-filter-descriptor.json','2ac36e62d203f85a5db8f8a17d28bfabe0db2eba94f069f095262d3fa5ad5aba')
VIEW='11b7bb1121567ec3098c044c67a9254b954242f37e9f43a4002655c87c29c046'
FIELDS=('comments','extraAttributes','features','geneLocations','genes','keywords','organism','organismHosts','proteinDescription','proteinExistence','references','uniProtKBCrossReferences')
ROUTES=tuple(sorted(tuple(TASK+':'+f for f in FIELDS)+(UNIPROT_NATIVE_INVERSE_TASK,)))

@dataclass(frozen=True)
class FullRecordBudget:
    batch_size:int
    chunk_size:int
    max_windows:int
    def __post_init__(self):
        if any(type(x) is not int or x<1 for x in (self.batch_size,self.chunk_size,self.max_windows)):
            raise ValueError('Explicit positive full-record native route budgets required')

@dataclass(frozen=True)
class PartialFullRecordRegistry:
    bindings:tuple
    plans:tuple
    budgets:tuple
    availability:dict
    asset_pins:tuple
    @property
    def training_authorized(self):return False
    @property
    def allocation_authorized(self):return False
    @property
    def complete_campaign(self):return False
    @property
    def unresolved(self):return ('No native validation/test roles for this source scope; explicit train-only campaign disposition required',
        'Full owner snapshot, field-specific memory/runtime evidence and complete campaign remain required',
        'Missing source fields remain unavailable targets; max_windows and single-pass exhaustion need explicit campaign capacity policy')

def checked_budgets(budgets):
    if type(budgets) is not dict or set(budgets)!=set(ROUTES):raise ValueError('Exactly thirteen supported full-record routes required')
    for key,budget in budgets.items():
        if type(budget) is not FullRecordBudget:raise ValueError('Exact typed full-record budget required')
        budget.__post_init__()
        if key!=UNIPROT_NATIVE_INVERSE_TASK and budget.batch_size!=1:
            raise ValueError('Live full-record field adapter requires batch_size one')
    return tuple(sorted(budgets.items()))

def load_full_record_assets(root):
    root=Path(root).resolve()
    pins=dict(census=CENSUS,filter=FILTER,vocabulary=PINS['payload_vocabulary'],fit=PINS['payload_fit'])
    refs={k:_reference(dict(path=p,sha256=h)) for k,(p,h) in pins.items()};raws={k:_read(root,r) for k,r in refs.items()}
    fit=json.loads(raws['fit']);guard=json.loads(raws['filter']);census=json.loads(raws['census'])
    if (fit.get('status')!='complete-source-fit' or fit.get('train_views')!=602
            or fit.get('target_fields_used') is not False or fit.get('heldout_rows_used') is not False
            or fit.get('payload_vocabulary_artifact',{}).get('sha256')!=refs['vocabulary'].sha256):raise ValueError('Pinned source-only vocabulary proof required')
    if (tuple(census['complete_annotation_packet_fields'])!=FIELDS
            or census['retained_original_role_counts']!={'train':106675,'validation':0,'test':0}
            or guard['exact_inverse_before']!=76472 or guard['exact_inverse_after']!=76471
            or guard['additional_excluded_taskviews']!=1):raise ValueError('Exact original field axes and inverse guard counts required')
    view=load_inverse_literal_filtered_native_view(FILTER[0],FILTER[1],root=root)
    if finite_view_policy_sha256(view)!=VIEW:raise ValueError('Exact inverse-literal guard required; base view refused')
    source=FullRecordUniProtSource(view,CENSUS[0],CENSUS[1])
    inverse_paths=set()
    for d in view.descriptors:
        entries=[r for r in d['source_task_schema_counts'] if r['source_family']==FAMILY and r['task_name']==UNIPROT_NATIVE_INVERSE_TASK and r['eligible_view_disposition'] in (0,1,2)]
        if entries:
            if any(r['split']!='train' or r['eligible_view_disposition']!=0 for r in entries):raise ValueError('Unreviewed native inverse validation/test role')
            inverse_paths.add(d['source_shard']['path'])
    if not inverse_paths:raise ValueError('Missing original inverse sequence route')
    paths=set(source.source_paths)|inverse_paths;descriptors={d['source_shard']['path']:d for d in view.descriptors}
    if any(full9(root/p)!=tuple(descriptors[p]['source_full9']) for p in paths):raise ValueError('Original full-record source identity changed')
    availability=dict(full_record_prediction_train_rows=106675,inverse_train_rows=76471,
        validation_rows=0,test_rows=0,field_observed_train_rows=census['present_source_field_positions'],
        exact_original_source_shards=len(paths),guarded_inverse_rows_omitted=1)
    codec=PubChemByteCodec(AnnotationVocabulary(raws['vocabulary'].decode('utf8')).tokenizer())
    return view,source,codec,availability,refs,raws

def _construct(root,view,source,codec,snapshot,ordered):
    bindings=[];plans=[]
    for task,budget in ordered:
        if task==UNIPROT_NATIVE_INVERSE_TASK:
            adapter=UniProtNativeCampaignAdapter(view,snapshot,'inverse',codec,batch_size=budget.batch_size,chunk_size=budget.chunk_size)
        else:
            field=task[len(TASK)+1:]
            adapter=UniProtFullRecordCampaignAdapter(view,source,codec,field,snapshot,budget.chunk_size)
        binding=adapter.train_binding()
        if binding.route.task_key!=task or binding.route.native_view_policy_sha256!=VIEW:raise ValueError('Exact guarded requested task differs')
        plans.append(NativeTrainRoutePlan(task,digest(binding.route.initial_cursor),executable_adapter_sha256(binding.prepare,root=root),budget.max_windows))
        bindings.append((task,binding))
    return tuple(bindings),tuple(plans)

def build_full_record_registry(root,*,source_snapshot,expected_view_policy_sha256,budgets):
    ordered=checked_budgets(budgets)
    if expected_view_policy_sha256!=VIEW:raise ValueError('Only exact inverse-literal-filter view is supported')
    root=Path(root).resolve();snapshot=_reference(source_snapshot);before=_read(root,snapshot)
    view,source,codec,availability,refs,raws=load_full_record_assets(root)
    bindings,plans=_construct(root,view,source,codec,snapshot.sha256,ordered)
    descriptors={d['source_shard']['path']:d for d in view.descriptors}
    paths={p for _,b in bindings for p in b.route._paths}
    if any(full9(root/p)!=tuple(descriptors[p]['source_full9']) for p in paths):raise ValueError('Original full-record shards changed during assembly')
    if _read(root,snapshot)!=before or any(_read(root,r)!=raws[k] for k,r in refs.items()):raise ValueError('Pinned full-record assets changed')
    refs['source_snapshot']=snapshot
    return PartialFullRecordRegistry(bindings,plans,ordered,availability,tuple(sorted(refs.items())))
