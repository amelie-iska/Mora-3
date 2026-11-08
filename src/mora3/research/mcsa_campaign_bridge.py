"""Pinned scoped M-CSA train/evaluator bridge; no full owner admission."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
from mora3.data.admet import digest
from mora3.data.mcsa_weak_sites import WeakRoleVocabulary
from mora3.data.finite_union_inverse_literal_filter import load_inverse_literal_filtered_native_view
from mora3.research.primary_campaign_assembly import _reference,_read
from mora3.research.mcsa_campaign_route import MCSACampaignTrainRoute,MCSAFinitePassPolicy
from mora3.research.mup_campaign import NativeTrainRoutePlan,executable_adapter_sha256
from mora3.research.matched_validation import CampaignValidationSources,matched_validation_source_policy_sha256,MatchedValidationTask

REVIEW=('outputs/implementation/readiness-20261107-root-mcsa-scoped-review-37000-v1/accepted-scoped-source-review.json','6d73937c6dd60875f36d92a908cdd62df8b75d34d54b3f16f841dd9666b5f0ba')
ENROLLMENT=('outputs/implementation/readiness-20261106-model-mup/primary-campaign-driver-v1/actual-mcsa-mixed-validation-enrollment-v1.json','0c04cea7280f3093067b202e3d81ee1a701b38925a6f021e4dd4206cb75027d9')
FILTER=('outputs/data-intake/readiness-20261106-corpus-continuation/current32-effective-inverse-target-fragment-filter-v2/effective-filter-descriptor.json','2ac36e62d203f85a5db8f8a17d28bfabe0db2eba94f069f095262d3fa5ad5aba')
@dataclass(frozen=True)
class MCSABridge:
    binding:object
    plan:NativeTrainRoutePlan
    validation_sources:CampaignValidationSources
    evaluator:object
    evaluator_sha256:str
    mixed_validation_policy_sha256:str
    coordinates:tuple
    observed_targets:int
    availability:dict
    asset_pins:tuple
    @property
    def training_authorized(self):return False
    @property
    def allocation_authorized(self):return False
    @property
    def complete_campaign(self):return False
    def require_matched_task(self,task,*,read_batch_size,max_selected_bytes):
        if type(task) is not MatchedValidationTask:raise ValueError('Explicit matched task and independently supplied reference required')
        return self.binding.route.matched_validation_batches(task,read_batch_size=read_batch_size,max_selected_bytes=max_selected_bytes)

def _source_reference(root,ref):
    path=Path(ref['path'])
    if path.is_absolute():path=path.relative_to(root)
    return _reference(dict(path=str(path),sha256=ref['sha256']))

def build_mcsa_bridge(root,*,source_snapshot,pass_policy,max_windows,task_cycle,total_updates,max_validation_bytes):
    root=Path(root).resolve()
    if root!=Path.cwd().resolve():raise ValueError('Existing MCSA readers require exact repository working directory')
    if type(pass_policy) is not MCSAFinitePassPolicy:raise ValueError('Explicit reviewed finite pass policy required')
    pass_policy.__post_init__()
    if any(type(x) is not int or x<1 for x in (max_windows,total_updates,max_validation_bytes)):raise ValueError('Explicit positive update/window/validation budgets required')
    refs={k:_reference(dict(path=p,sha256=h)) for k,(p,h) in dict(review=REVIEW,enrollment=ENROLLMENT,filter=FILTER).items()}
    refs['snapshot']=_reference(source_snapshot);raws={k:_read(root,r) for k,r in refs.items()}
    packet=json.loads(raws['review']);enrollment=json.loads(raws['enrollment'])
    ref=packet['component_manifest'];refs['component']=_source_reference(root,ref);raws['component']=_read(root,refs['component']);component=json.loads(raws['component'])
    for name,key in [('vocabulary','source_vocabulary'),('candidate','source_candidate_file')]:
        ref=component[key];refs[name]=_source_reference(root,ref)
    # Candidate content is validated by the existing admission compiler, which owns its size policy.
    raws['vocabulary']=_read(root,refs['vocabulary']);vocab=WeakRoleVocabulary.from_document(json.loads(raws['vocabulary']))
    if packet['approved_model_binding']['payload_max_tokens']!=37000:raise ValueError('Exact literal reviewed capacity required')
    route=MCSACampaignTrainRoute(root/REVIEW[0],REVIEW[1],vocabulary=vocab,source_snapshot_sha256=refs['snapshot'].sha256,pass_policy=pass_policy,reviewed_payload_max_tokens=37000)
    route.require_task_cycle_capacity(task_cycle,total_updates)
    coordinates,count=route.validation_enrollment(max_selected_bytes=max_validation_bytes)
    if ([asdict(r) for r in coordinates]!=enrollment['literal_original_validation_coordinates'] or len(coordinates)!=68 or count!=2043):raise ValueError('Exact reviewed original validation coordinates/denominator differ')
    view=load_inverse_literal_filtered_native_view(root/FILTER[0],FILTER[1],root=root)
    mixed=CampaignValidationSources(view,route);binding=route.train_binding()
    plan=NativeTrainRoutePlan(route.task_key,digest(route.initial_cursor),executable_adapter_sha256(binding.prepare,root=root),max_windows)
    evaluator=route.evaluate_validation;esha=executable_adapter_sha256(evaluator,root=root)
    for k,raw in raws.items():
        if _read(root,refs[k])!=raw:raise ValueError('Pinned MCSA source metadata changed')
    return MCSABridge(binding,plan,mixed,evaluator,esha,matched_validation_source_policy_sha256(mixed),coordinates,count,
        dict(source_splits=component['source_splits'],role_EOS_symbols_by_split=component['role_EOS_symbols_by_split'],records_per_pass=route.records_per_pass,maximum_updates=route.maximum_updates,reviewed_payload_capacity=37000,matched_reference_loss=None,test_evaluator_bound=False),tuple(sorted(refs.items())))
