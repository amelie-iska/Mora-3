"""Original guarded ReactZyme train route only; no expansion or admission."""
from dataclasses import dataclass
from pathlib import Path
import json

from mora3.data.admet import canonical,digest
from mora3.data.finite_union_native_rows import full9
from mora3.data.finite_union_annotation_study_filter import load_study_filtered_native_view
from mora3.data.reactzyme_current import ReactZymeCurrentSource,TASK
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.model.pubchem_decoder import PubChemByteCodec
from mora3.model.source_assembly_inputs import PINS,CURRENT_VIEW_PINS
from .primary_campaign_assembly import _reference,_read
from .native_admet_reaction_campaign_routes import NativeAdmetReactionCampaignAdapter
from .mup_campaign import NativeTrainRoutePlan,executable_adapter_sha256

CENSUS=('outputs/implementation/readiness-20261106-model-mup/current-reactzyme-guarded-route-census-v1/manifest.json','a86e3af39c8fb0ff24eca23483eb382afa3586033679f11f9222ef440716cfd0')
READER=('outputs/implementation/readiness-20261106-model-mup/current-reactzyme-guarded-native-adapter-v1/receipt.json','66a1c2555b1edb2e64fb579fe45a41ff83c4f7e012af51790bee68166a5590b7')

@dataclass(frozen=True)
class ReactZymeBudget:
    batch_size: int
    chunk_size: int
    max_windows: int
    def __post_init__(self):
        if any(type(n) is not int or n<1 for n in (self.batch_size,self.chunk_size,self.max_windows)):
            raise ValueError('Explicit positive original ReactZyme route budgets required')

@dataclass(frozen=True)
class PartialReactZymeRegistry:
    binding: object
    plan: NativeTrainRoutePlan
    budget: ReactZymeBudget
    asset_pins: tuple
    validation_obligation: str='No original validation/test roles; campaign must explicitly admit a train-only disposition or supply separately qualified evaluation. No evaluator manufactured.'
    @property
    def complete_campaign(self):return False
    @property
    def training_authorized(self):return False
    @property
    def allocation_authorized(self):return False

def _receipt_check(receipt,source):
    if (receipt.get('schema')!='Mora3.CurrentReactZymeGuardClearNativeAdapter/v1'
            or receipt.get('all_guard_clear_original_condition_and_targets_verified') is not True
            or receipt.get('source_full9_unchanged') is not True
            or receipt.get('source_census_sha256')!=CENSUS[1]
            or receipt.get('verified_guard_clear_train_rows')!=40426
            or receipt.get('guard_clear_original_roles')!={'train':40426,'validation':0,'test':0}
            or canonical(receipt.get('source_axes'))!=canonical(source.axes.binding())):
        raise ValueError('Exact original guard-clear reader and source axes required')
    expected={(r['path'],r['ordinal']):r['native_payload_sha256'] for r in receipt['unresolved_exact_current_rows_refused']}
    if len(expected)!=16 or expected!=source._unresolved:
        raise ValueError('Exact sixteen original unresolved guards required')

def load_original_reactzyme_assets(root):
    """Pinned metadata/source identity verification, no native task iteration."""
    root=Path(root).resolve()
    pins=dict(census=CENSUS,reader=READER,view=CURRENT_VIEW_PINS['effective_source_view'],
        vocabulary=PINS['payload_vocabulary'],fit=PINS['payload_fit'])
    refs={k:_reference(dict(path=p,sha256=h)) for k,(p,h) in pins.items()}
    raws={k:_read(root,r) for k,r in refs.items()}
    fit=json.loads(raws['fit'])
    if (fit.get('status')!='complete-source-fit' or fit.get('train_views')!=602
            or fit.get('target_fields_used') is not False or fit.get('heldout_rows_used') is not False
            or fit.get('payload_vocabulary_artifact',{}).get('sha256')!=refs['vocabulary'].sha256):
        raise ValueError('Pinned source-only complete vocabulary fit required')
    view=load_study_filtered_native_view(root/refs['view'].path,refs['view'].sha256,root=root)
    source=ReactZymeCurrentSource(view,refs['census'].path,refs['census'].sha256)
    receipt=json.loads(raws['reader']);_receipt_check(receipt,source)
    if digest((root/'src/mora3/data/reactzyme_current.py').read_bytes())!=receipt['source_code_sha256']:
        raise ValueError('Prior actual native reader proof code changed')
    by_path={d['source_shard']['path']:d for d in view.descriptors}
    if any(p not in by_path or full9(root/p)!=tuple(by_path[p]['source_full9']) for p in source.source_paths):
        raise ValueError('Original ReactZyme source fingerprint changed')
    vocab=AnnotationVocabulary(raws['vocabulary'].decode('utf8'))
    return source,PubChemByteCodec(vocab.tokenizer()),refs,raws

def _construct(root,source,codec,snapshot_sha256,budget):
    adapter=NativeAdmetReactionCampaignAdapter(source,source._view,snapshot_sha256,'enzyme_product',codec,budget.chunk_size)
    binding=adapter.train_binding(batch_size=budget.batch_size)
    plan=NativeTrainRoutePlan(TASK,digest(binding.route.initial_cursor),
        executable_adapter_sha256(binding.prepare,root=root),budget.max_windows)
    return binding,plan

def build_original_reactzyme_registry(root,*,source_snapshot,expected_view_policy_sha256,budget):
    """External full-owner snapshot is pinned, never semantically admitted here."""
    if type(budget) is not ReactZymeBudget:raise ValueError('Exact typed ReactZyme budget required')
    budget.__post_init__()
    root=Path(root).resolve();snapshot=_reference(source_snapshot);before=_read(root,snapshot)
    source,codec,refs,raws=load_original_reactzyme_assets(root)
    if source.axes.native_view_policy_sha256!=expected_view_policy_sha256:
        raise ValueError('Exact current view policy required')
    binding,plan=_construct(root,source,codec,snapshot.sha256,budget)
    if _read(root,snapshot)!=before or any(_read(root,r)!=raws[k] for k,r in refs.items()):
        raise ValueError('Pinned assembly assets changed')
    by_path={d['source_shard']['path']:d for d in source._view.descriptors}
    if any(full9(root/p)!=tuple(by_path[p]['source_full9']) for p in source.source_paths):
        raise ValueError('Original source changed during route construction')
    refs['source_snapshot']=snapshot
    return PartialReactZymeRegistry(binding,plan,budget,tuple(sorted(refs.items())))
