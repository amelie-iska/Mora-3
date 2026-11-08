"""Exact pending Tox21 inverse validation source enrollment, not a full cohort."""
from dataclasses import dataclass
from pathlib import Path
import json
from mora3.data.admet import canonical,digest
from mora3.data.finite_union_native_rows import SelectedNativeSourceBatch
from mora3.data.finite_union_annotation_study_filter import load_study_filtered_native_view
from mora3.data.finite_union_inverse_literal_filter import load_inverse_literal_filtered_native_view
from mora3.data.tox21_current import Tox21CurrentSource
from mora3.data.tox21 import INVERSE,SOURCE
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from .primary_campaign_assembly import _reference,_read
from .primary_native_routes import finite_view_policy_sha256
from .tox21_campaign_routes import Tox21CampaignAdapter
from .mup_campaign import executable_adapter_sha256
from .matched_validation import ValidationSourceRow

PROOF=('outputs/implementation/readiness-20261106-model-mup/primary-campaign-driver-v1/actual-tox21-inverse-campaign-validation-enrollment-v1.json','d5a8e6c2f6e011f88348668effe31157df4a889cad102b32b364b429025b7f42')

@dataclass(frozen=True)
class PendingInverseValidation:
    adapter:object
    evaluator_sha256:str
    train_initial_cursor:bytes
    coordinates:tuple
    batches:tuple
    selection_sha256:str
    observed_targets:int
    @property
    def evaluator(self):return self.adapter.evaluate_validation
    @property
    def complete_cohort(self):return False
    @property
    def training_authorized(self):return False
    @property
    def unresolved(self):return ('complete_owner_snapshot','initial_reference_loss','normalization_floor','full_campaign_cohort')
    def cursor(self,index):
        if type(index) is not int or not 0<=index<=len(self.batches):raise ValueError('Exact bounded validation batch index required')
        return canonical(dict(schema='Mora3.PendingTox21InverseValidationCursor/v1',role='validation',
            task=INVERSE,selection_sha256=self.selection_sha256,next_batch_index=index))
    @property
    def initial_cursor(self):return self.cursor(0)
    def iter_from(self,cursor):
        try:index=json.loads(cursor)['next_batch_index']
        except (ValueError,TypeError,KeyError) as e:raise ValueError('Exact validation cursor required') from e
        if type(cursor) is not bytes or cursor!=self.cursor(index):raise ValueError('Validation cursor role or selection differs')
        for i in range(index,len(self.batches)):
            batch=self.batches[i]
            self.adapter.current_view.require_native_rows(batch,split='validation')
            self.adapter.validation_observed_targets(batch)
            yield batch,self.cursor(i+1)

def _scope_check(proof):
    if (proof.get('schema')!='Mora3.PendingTox21InverseCampaignValidationEnrollment/v1'
            or proof.get('complete_owner_snapshot_sha256') is not None or proof.get('initial_reference_loss') is not None
            or proof.get('runnable_complete_campaign_created') is not False
            or proof.get('full_project_training_authorized') is not False
            or proof.get('inverse_original_validation_rows')!=114 or proof.get('exact_payload_tokens_and_EOS')!=5225
            or proof.get('source_scope',{}).get('complete_model_source_snapshot') is not False
            or digest(canonical(proof['source_scope']))!=proof.get('source_scope_sha256')):
        raise ValueError('Exact pending original validation-only scope required')
    coordinates=tuple(ValidationSourceRow(**r['coordinate']) for r in proof['inverse_validation_source_coordinates'])
    keys=tuple((r.source_path,r.original_row_ordinal) for r in coordinates)
    if len(coordinates)!=114 or keys!=tuple(sorted(set(keys))):raise ValueError('Exact unique ordered validation coordinates required')
    return coordinates

def bind_pending_inverse_validation(root,*,read_batch_size,max_selected_arrow_bytes):
    if any(type(n) is not int or n<1 for n in (read_batch_size,max_selected_arrow_bytes)):
        raise ValueError('Explicit positive validation read and selected-byte budgets required')
    import pyarrow as pa
    root=Path(root).resolve();proof_ref=_reference(dict(path=PROOF[0],sha256=PROOF[1]))
    proof_raw=_read(root,proof_ref);proof=json.loads(proof_raw);coordinates=_scope_check(proof)
    refs={k:_reference(dict(path=v['path'],sha256=v['sha256'])) for k,v in proof['source_scope']['source_evidence'].items()}
    raws={k:_read(root,r) for k,r in refs.items()}
    base=refs['base_filter'];current=refs['current_filter'];census=refs['source_census']
    original=load_study_filtered_native_view(root/base.path,base.sha256,root=root)
    view=load_inverse_literal_filtered_native_view(current.path,current.sha256,root=root)
    if finite_view_policy_sha256(view)!=proof['source_scope']['current_native_view_policy_sha256']:
        raise ValueError('Exact enrolled inverse-target guard policy required')
    source=Tox21CurrentSource(original,census.path,census.sha256)
    vocabulary=AnnotationVocabulary(raws['payload_vocabulary'].decode('utf8'))
    witness,=tuple(w for w in proof['native_route_binding_witnesses'] if w['direction']==INVERSE)
    adapter=Tox21CampaignAdapter(source,view,vocabulary,proof['source_scope_sha256'],INVERSE,witness['adapter_source_binding']['chunk_size'])
    train=adapter.train_binding(batch_size=witness['train_route_batch_size'])
    evaluator_sha=executable_adapter_sha256(adapter.evaluate_validation,root=root)
    if (canonical(adapter.executable_binding())!=canonical(witness['adapter_source_binding'])
            or evaluator_sha!=witness['validation_adapter_sha256']
            or executable_adapter_sha256(adapter.prepare,root=root)!=witness['train_adapter_sha256']
            or digest(train.route.initial_cursor)!=witness['train_route_cursor_sha256']):
        raise ValueError('Exact reviewed live evaluator/adapter/cursor binding changed')
    remaining={(c.source_path,c.original_row_ordinal):c for c in coordinates}
    batches=[];seen=[];selected_bytes=targets=0
    for batch in view.iter_native_task_batches(split='validation',source_paths=tuple(dict.fromkeys(c.source_path for c in coordinates)),batch_size=read_batch_size):
        selected=[(i,o,remaining[(batch.source_shard['path'],o)]) for i,o in enumerate(batch.source_row_ordinals) if (batch.source_shard['path'],o) in remaining]
        if not selected:continue
        positions,ordinals,cs=zip(*selected,strict=True)
        exact=SelectedNativeSourceBatch(batch.source_shard,ordinals,'validation',batch.native_rows.take(pa.array(positions,type=pa.int64())))
        view.require_native_rows(exact,split='validation')
        fields=('source_family','task_name','native_schema','split','task_id','native_payload_sha256','source_revision')
        for row,c in zip(zip(*(exact.native_rows[k].to_pylist() for k in fields),strict=True),cs,strict=True):
            if (batch.source_shard['sha256']!=c.shard_sha256 or row!=(SOURCE,INVERSE,'mora3.CorpusTask/v1','validation',c.task_id,c.payload_sha256,c.source_revision)):
                raise ValueError('Original validation role/direction/coordinate differs')
            del remaining[(c.source_path,c.original_row_ordinal)];seen.append(c)
        selected_bytes+=exact.native_rows.nbytes
        if selected_bytes>max_selected_arrow_bytes:raise ValueError('Complete validation selection exceeds declared byte budget')
        targets+=adapter.validation_observed_targets(exact);batches.append(exact)
    if remaining or tuple(seen)!=coordinates or targets!=5225:raise ValueError('Complete exact original validation denominator differs')
    if _read(root,proof_ref)!=proof_raw or any(_read(root,r)!=raws[k] for k,r in refs.items()):raise ValueError('Pinned enrollment evidence changed')
    selection=digest(canonical(dict(enrollment_sha256=PROOF[1],scope_sha256=proof['source_scope_sha256'],
        evaluator_sha256=evaluator_sha,role='validation',read_batch_size=read_batch_size,max_selected_arrow_bytes=max_selected_arrow_bytes)))
    return PendingInverseValidation(adapter,evaluator_sha,train.route.initial_cursor,coordinates,tuple(batches),selection,targets)
