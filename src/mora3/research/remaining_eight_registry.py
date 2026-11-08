"""Original eight finite task axes, eleven executable train keys; no admission."""
from dataclasses import dataclass
from pathlib import Path
import json
from mora3.data.admet import digest
from mora3.data.finite_union_native_rows import FiniteUnionNativeView,full9
from mora3.data.finite_union_annotation_study_filter import load_study_filtered_native_view
from mora3.data.finite_union_inverse_literal_filter import load_inverse_literal_filtered_native_view
from mora3.data.deepbacs_native import DeepBacsNativeSource
from mora3.data.uniprot_native_annotations import UNIPROT_FORWARD_TARGET_FIELDS,UNIPROT_NATIVE_FORWARD_TASK
from mora3.model.source_assembly_inputs import PINS,CURRENT_VIEW_PINS
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.model.pubchem_decoder import PubChemByteCodec
from mora3.model.source_scalar_objectives import SourceScalarAxis
from mora3.research.primary_campaign_assembly import _read,_reference
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.mup_campaign import NativeTrainRoutePlan,executable_adapter_sha256
from mora3.research.deepbacs_campaign_routes import DeepBacsNativeRouteAdapter
from mora3.research.deepbacs_forward_campaign_route import DeepBacsForwardCampaignAdapter
from mora3.research.gperturb_campaign_routes import GPerturbNativeCampaignAdapter
from mora3.research.native_scalar_campaign_routes import TrainOnlyScalarCampaignAdapter,_FAMILIES
from mora3.research.assay_campaign_routes import FreeSolvCampaignAdapter
from mora3.research.uniprot_campaign_routes import UniProtNativeCampaignAdapter

FILTER=('outputs/data-intake/readiness-20261106-corpus-continuation/current32-effective-inverse-target-fragment-filter-v2/effective-filter-descriptor.json','2ac36e62d203f85a5db8f8a17d28bfabe0db2eba94f069f095262d3fa5ad5aba')
BASE='c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'
INVERSE='11b7bb1121567ec3098c044c67a9254b954242f37e9f43a4002655c87c29c046'
SPECS={f'DeepBacs-Staph:{d}':('DeepBacs-Staph',d) for d in ('image_to_mask','mask_to_image')}
SPECS.update({f'GPerturb-sciPlex2:{d}':('GPerturb-sciPlex2',d) for d in ('intervention','response')})
SPECS.update({task:(family,task) for family,(task,_) in _FAMILIES.items()})
SPECS['experimental_hydration_free_energy']=('FreeSolv','experimental_hydration_free_energy')
SPECS.update({f'{UNIPROT_NATIVE_FORWARD_TASK}:{field}':('UniProt-function-annotations',UNIPROT_NATIVE_FORWARD_TASK) for field in UNIPROT_FORWARD_TARGET_FIELDS})
KEYS=tuple(sorted(SPECS))
@dataclass(frozen=True)
class RouteBudget:
    batch_size:int
    max_windows:int
    chunk_size:int
    def __post_init__(self):
        if any(type(x) is not int or x<1 for x in (self.batch_size,self.max_windows,self.chunk_size)):raise ValueError('Explicit positive resource budgets required')
@dataclass(frozen=True)
class PartialEightRegistry:
    bindings:tuple
    plans:tuple
    budgets:tuple
    availability:dict
    asset_pins:tuple
    @property
    def complete_campaign(self):return False
    @property
    def allocation_authorized(self):return False
    @property
    def training_authorized(self):return False

def checked_budgets(budgets):
    if type(budgets) is not dict or set(budgets)!=set(KEYS):raise ValueError('Exact eleven route budgets required')
    for k,b in budgets.items():
        if type(b) is not RouteBudget:raise ValueError('Exact typed budget required')
        b.__post_init__();family=SPECS[k][0]
        if family=='DeepBacs-Staph' and b.batch_size!=1:raise ValueError('Exact one-raster batch required')
        if family!='UniProt-function-annotations' and b.chunk_size!=1:raise ValueError('No chunk-size control for this native objective')
    return tuple(sorted(budgets.items()))

def load_assets(root):
    root=Path(root).resolve();pins=dict(view=CURRENT_VIEW_PINS['effective_source_view'],old=PINS['effective_source_view'],inverse=FILTER,
        census=CURRENT_VIEW_PINS['effective_axis_census'],image=CURRENT_VIEW_PINS['grayscale_envelope'],
        scalar=CURRENT_VIEW_PINS['scalar_source_projection'],guards=CURRENT_VIEW_PINS['scalar_original_guards'],
        vocabulary=PINS['payload_vocabulary'],fit=PINS['payload_fit'])
    refs={k:_reference(dict(path=p,sha256=h)) for k,(p,h) in pins.items()};raws={k:_read(root,r) for k,r in refs.items()}
    fit=json.loads(raws['fit'])
    if (fit.get('status')!='complete-source-fit' or fit.get('train_views')!=602 or fit.get('target_fields_used') is not False or fit.get('heldout_rows_used') is not False or fit.get('payload_vocabulary_artifact',{}).get('sha256')!=refs['vocabulary'].sha256):raise ValueError('Exact source-only vocabulary fit required')
    view=load_study_filtered_native_view(root/refs['view'].path,refs['view'].sha256,root=root)
    old=FiniteUnionNativeView.from_effective_filter(root/refs['old'].path,refs['old'].sha256,root=root)
    inverse=load_inverse_literal_filtered_native_view(root/refs['inverse'].path,refs['inverse'].sha256,root=root)
    if finite_view_policy_sha256(view)!=BASE or finite_view_policy_sha256(inverse)!=INVERSE:raise ValueError('Exact route policies required')
    image=DeepBacsNativeSource(view,refs['image'].path,refs['image'].sha256)
    scalar=SourceScalarAxis.from_verified_projection(refs['scalar'].path,refs['scalar'].sha256,root=root)
    rows=json.loads(raws['census'])['task_schema_role_routes'];availability={}
    for key,(family,task) in SPECS.items():
        selected=[r for r in rows if (r['source_family'],r['task_name'])==(family,task)]
        if len([r for r in selected if r['split']=='train' and r['retained_rows']>0])!=1:raise ValueError('Exact retained train source axis absent')
        availability[key]=dict(native_roles={r['split']:r['retained_rows'] for r in selected},validation_bound=False,
            evaluation_obligation='FreeSolv evaluator exists; exact coordinates, denominator and baseline remain unenrolled' if family=='FreeSolv' else 'No matched validation evaluator bound; native role rows are not campaign enrollment',
            source_limit='original test development-exposed' if family=='DeepBacs-Staph' else ('observational cells; source dose coordinate, no physical-dose or paired causal claim' if family=='GPerturb-sciPlex2' else 'original guarded source semantics only'))
    return view,old,inverse,image,scalar,PubChemByteCodec(AnnotationVocabulary(raws['vocabulary'].decode()).tokenizer()),availability,refs,raws

def _construct(root,assets,snapshot,budgets):
    view,old,inverse,image,scalar,codec=assets[:6];bindings=[];plans=[]
    for key,b in budgets:
        family,task=SPECS[key]
        if family=='DeepBacs-Staph':adapter=(DeepBacsForwardCampaignAdapter if task=='image_to_mask' else DeepBacsNativeRouteAdapter)(view,image,snapshot)
        elif family=='GPerturb-sciPlex2':adapter=GPerturbNativeCampaignAdapter(view,snapshot,task,b.batch_size)
        elif family=='FreeSolv':adapter=FreeSolvCampaignAdapter(old,view,scalar,snapshot)
        elif family in _FAMILIES:adapter=TrainOnlyScalarCampaignAdapter(old,view,scalar,snapshot,family)
        else:adapter=UniProtNativeCampaignAdapter(inverse,snapshot,'forward',codec,key.split(':',1)[1],b.batch_size,b.chunk_size)
        binding=adapter.train_binding(batch_size=b.batch_size) if family in _FAMILIES or family=='FreeSolv' else adapter.train_binding()
        if binding.route.task_key!=key:raise ValueError('Exact route key differs')
        # Original scalar projection may only be reused if per-source coordinates are unchanged.
        if family in _FAMILIES or family=='FreeSolv':
            paths=set(binding.route._paths)
            if (any(old._sealed_by_path[p]!=view._sealed_by_path[p] for p in paths) or
                {x for x in old._ordinal_exclusions if x[0] in paths}!={x for x in view._ordinal_exclusions if x[0] in paths}):raise ValueError('Scalar original guarded source coordinates changed')
        descriptors={d['source_shard']['path']:d for d in binding.route._view.descriptors}
        if any(full9(root/p)!=tuple(descriptors[p]['source_full9']) for p in binding.route._paths):raise ValueError('Native source fingerprint changed')
        bindings.append((key,binding));plans.append(NativeTrainRoutePlan(key,digest(binding.route.initial_cursor),executable_adapter_sha256(binding.prepare,root=root),b.max_windows))
    return tuple(bindings),tuple(plans)

def build_remaining_eight_registry(root,*,source_snapshot,budgets):
    ordered=checked_budgets(budgets);root=Path(root).resolve();sr=_reference(source_snapshot);before=_read(root,sr)
    assets=load_assets(root);bindings,plans=_construct(root,assets,sr.sha256,ordered)
    availability,refs,raws=assets[-3:]
    if _read(root,sr)!=before or any(_read(root,r)!=raws[k] for k,r in refs.items()):raise ValueError('Pinned assets changed')
    refs['source_snapshot']=sr
    return PartialEightRegistry(bindings,plans,ordered,availability,tuple(sorted(refs.items())))
