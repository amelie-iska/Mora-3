"""Ten original NCI/Tox21/ExpansionRx train bindings; no evaluation admission."""
from dataclasses import dataclass
from pathlib import Path
import json
from mora3.data.admet import canonical,digest
from mora3.data.finite_union_native_rows import full9
from mora3.data.finite_union_annotation_study_filter import load_study_filtered_native_view
from mora3.data.nci_current_assays import NciCurrentAssaySource
from mora3.data.nci_aids import FAMILIES
from mora3.data.tox21_current import Tox21CurrentSource
from mora3.data.tox21 import FORWARD,INVERSE
from mora3.data.expansionrx_current import ExpansionRxCurrentSource
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.model.pubchem_decoder import PubChemByteCodec
from mora3.model.source_assembly_inputs import PINS,CURRENT_VIEW_PINS
from .primary_campaign_assembly import _reference,_read
from .primary_native_routes import finite_view_policy_sha256
from .mup_campaign import NativeTrainRoutePlan,executable_adapter_sha256
from .assay_campaign_routes import NciCampaignAdapter
from .tox21_campaign_routes import Tox21CampaignAdapter
from .native_admet_reaction_campaign_routes import NativeAdmetReactionCampaignAdapter

BASE='outputs/implementation/readiness-20261106-model-mup/'
SPECS={
 'nci':('current-nci-native-assay-census-v1/manifest.json','75b9490c4e4a69107d266d4d31ed2a479b5de028031bf4cd85763d1b84a6ed3c','71f0074bc141b51b7444d14b3cb7e5572932813b54da6ec5e136b97f7262826a',NciCurrentAssaySource,'src/mora3/data/nci_current_assays.py','model_axes'),
 'tox21':('current-tox21-native-role-census-v1/manifest.json','befa198f67833b1f799112cbf200eb3943c6d709e781427d8059a292460578b1','b54d8555ea9094d3c01effa713230faa6411f990a61cc8e7b0c31f2b92428206',Tox21CurrentSource,'src/mora3/data/tox21_current.py','source_axes'),
 'expansionrx':('current-expansionrx-native-assay-census-v2/manifest.json','323e0da6f8be5717b3b82fae727e382435efe501fe7996cd452c8d3c5058b1ec','4a05948735c7289e693b99a6a47de47a10ef22c56e884056f99751744fddb6c4',ExpansionRxCurrentSource,'src/mora3/data/expansionrx_current.py','source_axes'),
}
ROUTES={FAMILIES[f]+d+'_prior':('nci',f,d) for f in FAMILIES for d in ('prediction','generation')}
ROUTES.update({d:('tox21',None,d) for d in (FORWARD,INVERSE)})
ROUTES.update({'source_native_ADMET_'+d:('expansionrx',None,d) for d in ('entity_to_measurement','measurement_to_entity')})
EXCLUDED=(
 ('NCI validation/test','Not enrolled here: requires exact matched cohort, live evaluator hashes and qualification of role/exposure; source-native role existence is not enrollment.'),
 ('Tox21 validation/test','Not enrolled here: original source roles preserved; exact campaign matched validation cohort and evaluator enrollment required.'),
 ('ExpansionRx validation/test','No validation factory in the existing adapter; original test is explicitly development-exposed and cannot become blind evaluation.'),
 ('inactive source expansions and any other task','Outside these ten pinned original train axes; unsupported axes refused, never silently skipped.'),
)

@dataclass(frozen=True)
class AssayBudget:
    batch_size:int
    chunk_size:int
    max_windows:int
    def __post_init__(self):
        if any(type(n) is not int or n<1 for n in (self.batch_size,self.chunk_size,self.max_windows)):
            raise ValueError('Explicit positive per-route budgets required')

@dataclass(frozen=True)
class PartialAssayRegistry:
    bindings:tuple
    plans:tuple
    budgets:tuple
    asset_pins:tuple
    excluded:tuple=EXCLUDED
    @property
    def complete_campaign(self):return False
    @property
    def allocation_authorized(self):return False
    @property
    def training_authorized(self):return False

def checked_budgets(budgets):
    if type(budgets) is not dict or set(budgets)!=set(ROUTES):
        raise ValueError('Exactly ten supported original assay routes and explicit budgets required')
    for b in budgets.values():
        if type(b) is not AssayBudget:raise ValueError('Exact typed assay budget required')
        b.__post_init__()
    return tuple(sorted(budgets.items()))

def load_original_assay_assets(root):
    root=Path(root).resolve()
    refs={k:_reference(dict(path=p,sha256=h)) for k,(p,h) in dict(view=CURRENT_VIEW_PINS['effective_source_view'],vocabulary=PINS['payload_vocabulary'],fit=PINS['payload_fit']).items()}
    for family,(path,census,receipt,*_) in SPECS.items():
        refs[family+'_census']=_reference(dict(path=BASE+path,sha256=census))
        refs[family+'_receipt']=_reference(dict(path=BASE+'current-'+family+'-native-adapter-v1/receipt.json',sha256=receipt))
    raws={k:_read(root,r) for k,r in refs.items()}
    fit=json.loads(raws['fit'])
    if (fit.get('status')!='complete-source-fit' or fit.get('train_views')!=602
            or fit.get('target_fields_used') is not False or fit.get('heldout_rows_used') is not False
            or fit.get('payload_vocabulary_artifact',{}).get('sha256')!=refs['vocabulary'].sha256):
        raise ValueError('Exact complete source-only vocabulary fit required')
    view=load_study_filtered_native_view(root/refs['view'].path,refs['view'].sha256,root=root)
    sources={}
    for family,(_,census,_,kind,code,axis_key) in SPECS.items():
        ref=refs[family+'_census'];source=kind(view,ref.path,ref.sha256)
        proof=json.loads(raws[family+'_receipt'])
        if (proof.get('source_census_sha256')!=census or proof.get('code_sha256')!=digest((root/code).read_bytes())
                or canonical(proof.get(axis_key))!=canonical(source.axes.binding())
                or proof.get('effective_filter')!=dict(path=refs['view'].path,sha256=refs['view'].sha256)):
            raise ValueError('Pinned actual original reader/census/axis binding differs: '+family)
        sources[family]=source
    _fingerprints(root,view,sources)
    vocabulary=AnnotationVocabulary(raws['vocabulary'].decode('utf8'))
    return view,sources,vocabulary,refs,raws

def _fingerprints(root,view,sources):
    descriptors={d['source_shard']['path']:d for d in view.descriptors}
    paths={p for source in sources.values() for p in source.source_paths}
    if any(p not in descriptors or full9(root/p)!=tuple(descriptors[p]['source_full9']) for p in paths):
        raise ValueError('Pinned original assay source fingerprint changed')

def _construct(root,view,sources,vocabulary,snapshot,ordered):
    bindings=[];plans=[];codec=PubChemByteCodec(vocabulary.tokenizer())
    for task,budget in ordered:
        kind,family,direction=ROUTES[task]
        if kind=='nci':adapter=NciCampaignAdapter(sources[kind],view,vocabulary,snapshot,family,direction,budget.chunk_size)
        elif kind=='tox21':adapter=Tox21CampaignAdapter(sources[kind],view,vocabulary,snapshot,direction,budget.chunk_size)
        else:adapter=NativeAdmetReactionCampaignAdapter(sources[kind],view,snapshot,direction,None if direction=='entity_to_measurement' else codec,budget.chunk_size)
        binding=adapter.train_binding(batch_size=budget.batch_size)
        if binding.route.task_key!=task:raise ValueError('Exact requested native task differs')
        plans.append(NativeTrainRoutePlan(task,digest(binding.route.initial_cursor),executable_adapter_sha256(binding.prepare,root=root),budget.max_windows))
        bindings.append((task,binding))
    return tuple(bindings),tuple(plans)

def build_original_assay_registry(root,*,source_snapshot,expected_view_policy_sha256,budgets):
    ordered=checked_budgets(budgets)
    root=Path(root).resolve();snapshot=_reference(source_snapshot);before=_read(root,snapshot)
    view,sources,vocabulary,refs,raws=load_original_assay_assets(root)
    if finite_view_policy_sha256(view)!=expected_view_policy_sha256:raise ValueError('Exact current assay source policy required')
    bindings,plans=_construct(root,view,sources,vocabulary,snapshot.sha256,ordered)
    _fingerprints(root,view,sources)
    if _read(root,snapshot)!=before or any(_read(root,r)!=raws[k] for k,r in refs.items()):
        raise ValueError('Pinned assay assets changed during construction')
    refs['source_snapshot']=snapshot
    return PartialAssayRegistry(bindings,plans,ordered,tuple(sorted(refs.items())))
