"""Pinned original ten Rosetta native train routes; no owner/eval admission."""
from dataclasses import dataclass
from pathlib import Path
import json
from mora3.data.admet import digest
from mora3.data.finite_union_native_rows import FiniteUnionNativeView,full9
from mora3.data.finite_union_annotation_study_filter import load_study_filtered_native_view
from mora3.model.source_assembly_inputs import PINS,CURRENT_VIEW_PINS
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.model.pubchem_decoder import PubChemByteCodec
from mora3.research.primary_campaign_assembly import _read,_reference
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.mup_campaign import NativeTrainRoutePlan,executable_adapter_sha256
from mora3.research.rosetta_cdrp_campaign_routes import RosettaCDRPCampaignAdapter
from mora3.research.lincs_pilot1_campaign_routes import LINCSCampaignAdapter
from mora3.research.rosetta_unconditional_campaign_routes import RosettaUnconditionalCampaignAdapter

VIEW='c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'
SPECS={}
for kind in ('CDRP','LINCS-Pilot1'):
    for modality in ('CellPainting','L1000'):
        for direction in ('molecule_to_profile','profile_to_molecule'):
            family=f'Rosetta-{kind}-molecule-profile-{modality}'
            SPECS[f'{family}:{direction}']=(kind,modality,family,direction)
for modality in ('CellPainting','L1000'):
    family='Rosetta-'+modality;task='unconditional_source_profile_modeling'
    SPECS[f'{family}:{task}']=('unconditional',modality,family,task)
KEYS=tuple(sorted(SPECS))

@dataclass(frozen=True)
class RosettaBudget:
    max_windows:int
    chunk_size:int
    def __post_init__(self):
        if any(type(x) is not int or x<1 for x in (self.max_windows,self.chunk_size)):
            raise ValueError('Explicit positive window/chunk budgets required; fixed batch size is one')

@dataclass(frozen=True)
class PartialRosettaRegistry:
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
    if type(budgets) is not dict or set(budgets)!=set(KEYS):raise ValueError('Exact ten Rosetta budgets required')
    for k,b in budgets.items():
        if type(b) is not RosettaBudget:raise ValueError('Typed Rosetta budget required')
        b.__post_init__()
        if SPECS[k][0]!='LINCS-Pilot1' and b.chunk_size!=1:raise ValueError('Non-LINCS routes require sentinel chunk_size=1; no chunked target semantics')
    return tuple(sorted(budgets.items()))

def load_assets(root):
    root=Path(root).resolve()
    pins=dict(view=CURRENT_VIEW_PINS['effective_source_view'],fit_view=PINS['effective_source_view'],
        census=CURRENT_VIEW_PINS['effective_axis_census'],groups=PINS['rosetta_groups'],codec=PINS['rosetta_codec'],
        arrays=CURRENT_VIEW_PINS['rosetta_unconditional_array_census'],lincs=PINS['lincs_manifest'],
        vocabulary=PINS['payload_vocabulary'],fit=PINS['payload_fit'])
    refs={k:_reference(dict(path=p,sha256=h)) for k,(p,h) in pins.items()};raws={k:_read(root,r) for k,r in refs.items()}
    fit=json.loads(raws['fit'])
    if (fit.get('status')!='complete-source-fit' or fit.get('train_views')!=602 or fit.get('target_fields_used') is not False
        or fit.get('heldout_rows_used') is not False or fit.get('payload_vocabulary_artifact',{}).get('sha256')!=refs['vocabulary'].sha256):
        raise ValueError('Exact source-only vocabulary fit required')
    view=load_study_filtered_native_view(root/refs['view'].path,refs['view'].sha256,root=root)
    old=FiniteUnionNativeView.from_effective_filter(root/refs['fit_view'].path,refs['fit_view'].sha256,root=root)
    if finite_view_policy_sha256(view)!=VIEW:raise ValueError('Exact original current source view required')
    rows=json.loads(raws['census'])['task_schema_role_routes'];availability={}
    for key,(_,_,family,task) in SPECS.items():
        selected=[r for r in rows if (r['source_family'],r['task_name'])==(family,task)]
        trains=[r for r in selected if r['split']=='train']
        if len(trains)!=1 or trains[0]['retained_rows']<1:raise ValueError('Exact retained original train axis absent')
        availability[key]=dict(native_roles={r['split']:r['retained_rows'] for r in selected},validation_evaluator_bound=False)
    return view,old,PubChemByteCodec(AnnotationVocabulary(raws['vocabulary'].decode()).tokenizer()),availability,refs,raws

def _construct(root,view,old,codec,snapshot,budgets):
    bindings=[];plans=[]
    for key,b in budgets:
        kind,modality,family,direction=SPECS[key]
        if kind=='CDRP':adapter=RosettaCDRPCampaignAdapter(view,old,modality,direction,snapshot)
        elif kind=='LINCS-Pilot1':adapter=LINCSCampaignAdapter(view,codec,modality,direction,snapshot,b.chunk_size)
        else:adapter=RosettaUnconditionalCampaignAdapter(view,view,family,snapshot)
        binding=adapter.train_binding()
        if binding.route.task_key!=key:raise ValueError('Exact native task differs')
        plans.append(NativeTrainRoutePlan(key,digest(binding.route.initial_cursor),executable_adapter_sha256(binding.prepare,root=root),b.max_windows))
        bindings.append((key,binding))
    descriptors={d['source_shard']['path']:d for d in view.descriptors}
    for path in {p for _,b in bindings for p in b.route._paths}:
        if full9(root/path)!=tuple(descriptors[path]['source_full9']):raise ValueError('Original Rosetta source fingerprint changed')
    return tuple(bindings),tuple(plans)

def build_rosetta_registry(root,*,source_snapshot,budgets):
    ordered=checked_budgets(budgets);root=Path(root).resolve();sr=_reference(source_snapshot);before=_read(root,sr)
    view,old,codec,availability,refs,raws=load_assets(root)
    bindings,plans=_construct(root,view,old,codec,sr.sha256,ordered)
    if _read(root,sr)!=before or any(_read(root,r)!=raws[k] for k,r in refs.items()):raise ValueError('Pinned assets changed')
    refs['source_snapshot']=sr
    return PartialRosettaRegistry(bindings,plans,ordered,availability,tuple(sorted(refs.items())))
