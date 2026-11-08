"""Scoped reviewed INV-081/GSM8K native train cursors and read-only validation.

Does not enroll FrozenPrimaryProxyCampaign validation or authorize a model run.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
from decimal import Decimal
import json
from mora3.data.admet import canonical,digest
from mora3.data.source_text_roles import NativeTextRoleView
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.model.pubchem_decoder import PubChemByteCodec
from mora3.model.source_assembly_inputs import PINS

REVIEW=('outputs/implementation/readiness-20261106-root-text-role-review-v1/accepted-source-role-review.json','3120a0ffa8c56a96647e35352f3459540e01c2a33215889227536f41dcbd43fe')
ENROLLMENT=('outputs/implementation/readiness-20261108-native-text-scoped-validation-v2/receipt.json','4516219d83e9397a2b6696645c18c3f34176148a2bba3b92fe8dbe789a85531c')
SPECS={'INV-081':('source_language_prior',4601,546,4982433),'INV-084/GSM8K':('gsm8k_answer',13392,1554,570624)}

@dataclass(frozen=True)
class TextBudget:
    batch_size:int
    max_batch_target_bytes:int
    max_materialized_target_bytes:int
    max_materialized_rows:int
    max_passes:int
    chunk_size:int
    recompute:bool
    def __post_init__(self):
        if any(type(x) is not int or x<1 for x in (self.batch_size,self.max_batch_target_bytes,self.max_materialized_target_bytes,self.max_materialized_rows,self.max_passes,self.chunk_size)) or type(self.recompute) is not bool:raise ValueError('Explicit positive text source/compute/pass bounds and recompute policy required')

@dataclass(frozen=True)
class TextWindow:
    rows:tuple
    role:str
    cursor_after:bytes
    evidence_sha256:str
    @property
    def original_source_rows(self):return len(self.rows)

class ReviewedTextCampaignRoute:
    loss_unit='complete native byte-token and EOS cross entropy in nats'
    target_unit='native fitted byte tokens and EOS'
    def __init__(self,view,codec,family,snapshot,budget,train,validation,validation_coordinates):
        self.view=view;self.codec=codec;self.family=family;self.source_snapshot_sha256=snapshot;self.budget=budget
        self.task_key='reviewed-native-text:'+family+':'+SPECS[family][0]
        self._train=tuple(train);self._validation=tuple(validation);self.validation_coordinates=tuple(validation_coordinates)
        self._binding=canonical(dict(schema='Mora3.ReviewedTextCampaignRoute/v1',snapshot=snapshot,family=family,task=SPECS[family][0],task_key=self.task_key,
            review_sha256=REVIEW[1],validation_enrollment_sha256=ENROLLMENT[1],vocabulary_sha256=PINS['payload_vocabulary'][1],codec=codec.configuration(),budget=asdict(budget),train_rows=len(train),validation_rows=len(validation),
            train_order='original-reviewed-reader-order;finite-explicit-identical-passes',validation_order='exact-pinned-coordinate-digest-ledger',original_roles_preserved=True))
        self.native_view_policy_sha256=digest(self._binding)
    @property
    def training_authorized(self):return False
    def executable_binding(self):
        self.view._require_unchanged()
        doc=json.loads(self._binding)
        if doc['budget']!=asdict(self.budget) or doc['snapshot']!=self.source_snapshot_sha256 or canonical(doc['codec'])!=canonical(self.codec.configuration()) or doc['family']!=self.family:raise ValueError('Text route binding changed')
        return doc
    @property
    def maximum_windows(self):return self.budget.max_passes*((len(self._train)+self.budget.batch_size-1)//self.budget.batch_size)
    def _cursor(self,pass_index,offset):return canonical(dict(schema='Mora3.ReviewedTextTrainCursor/v1',route_sha256=self.native_view_policy_sha256,role='train',pass_index=pass_index,next_offset=offset))
    @property
    def initial_cursor(self):return self._cursor(0,0)
    def _window(self,rows,role,after):
        for row in rows:self.view.require_row(row,role=role)
        if sum(len(r.native_row.full_target_utf8) for r in rows)>self.budget.max_batch_target_bytes:raise ValueError('Complete native text batch exceeds explicit target-byte budget; no truncation')
        return TextWindow(tuple(rows),role,after,digest(canonical([self.native_view_policy_sha256,role,[(digest(r.native_row.source_coordinate_json),digest(r.native_row.full_target_utf8)) for r in rows]])))
    def iter_from(self,cursor):
        self.executable_binding();d=json.loads(cursor)
        p,o=d.get('pass_index'),d.get('next_offset')
        if type(cursor) is not bytes or type(p) is not int or type(o) is not int or not 0<=p<=self.budget.max_passes or not 0<=o<=len(self._train) or (p==self.budget.max_passes and o!=0) or (o%self.budget.batch_size!=0 and o!=len(self._train)) or cursor!=self._cursor(p,o):raise ValueError('Exact finite original train cursor required')
        for pi in range(p,self.budget.max_passes):
            start=o if pi==p else 0
            for i in range(start,len(self._train),self.budget.batch_size):
                stop=min(i+self.budget.batch_size,len(self._train));after=self._cursor(pi+1,0) if stop==len(self._train) else self._cursor(pi,stop)
                yield self._window(self._train[i:stop],'train',after)
    def validation_batches(self):
        self.executable_binding()
        for i in range(0,len(self._validation),self.budget.batch_size):yield self._window(self._validation[i:i+self.budget.batch_size],'validation',b'')
    def observed_targets(self,window):
        if type(window) is not TextWindow or window.role not in ('train','validation') or not window.rows or len(window.rows)>self.budget.batch_size:raise ValueError('Exact scoped text window required')
        for issued in window.rows:
            self.view.require_row(issued,role=window.role)
            if issued.native_row.source_family!=self.family or issued.native_row.condition_inputs()['task']!=SPECS[self.family][0] or issued.native_row.evaluation_only:raise ValueError('Exact family/task and non-evaluation-only rows required')
        expected=self._window(window.rows,window.role,window.cursor_after)
        if window.evidence_sha256!=expected.evidence_sha256:raise ValueError('Text window evidence changed')
        return sum(len(self.codec.tokenizer.encode(r.native_row.full_target_utf8))+1 for r in window.rows)
    def _objective(self,owner,window,*,validation):
        from mora3.model.full_owner import SourceBoundOwner
        from mora3.model.source_text import SharedSourceTextModel
        if type(owner) is not SourceBoundOwner:raise ValueError('Exact source-bound full owner required')
        self.executable_binding()
        if not validation:owner.require_training_admission()
        if owner.get_extra_state()['source_snapshot_sha256']!=self.source_snapshot_sha256 or type(owner.text) is not SharedSourceTextModel or owner.text.decoder is not owner.pubchem:raise ValueError('Exact native shared text owner/snapshot required')
        branch=owner.text
        if self.budget.batch_size>branch.max_batch_size or self.budget.max_batch_target_bytes>branch.max_total_target_bytes or canonical(branch.decoder.codec.configuration())!=canonical(self.codec.configuration()):raise ValueError('Exact owner text codec/resource bounds required')
        expected='validation' if validation else 'train'
        if window.role!=expected:raise ValueError('No train/validation role substitution')
        self.observed_targets(window)
        conditions=tuple(r.native_row.condition_inputs() for r in window.rows);targets=tuple(r.native_row.full_target_utf8 for r in window.rows)
        method=branch.evaluation_objective if validation else branch.objective
        return method(conditions,targets,chunk_size=self.budget.chunk_size,recompute=self.budget.recompute if not validation else False)
    def prepare(self,window):
        from .primary_native_routes import PreparedNativeTargets
        if window.role!='train':raise ValueError('Validation rows cannot prepare train objectives')
        count=self.observed_targets(window)
        return PreparedNativeTargets(count,None,lambda owner:self._objective(owner,window,validation=False))
    def train_binding(self):
        from .primary_native_routes import NativeRouteBinding
        return NativeRouteBinding(self,self.prepare,self.loss_unit,self.target_unit)
    def evaluate_validation(self,owner,window):
        import torch
        from .mup_campaign import ValidationBatchReduction
        if torch.is_grad_enabled() or getattr(owner,'training',True):raise ValueError('Evaluation-only no-grad owner required')
        count=self.observed_targets(window);loss=self._objective(owner,window,validation=True)
        return ValidationBatchReduction(Decimal(str(float(loss.detach().cpu())))*count,count,None)

def build_reviewed_text_routes(root,*,source_snapshot,budgets):
    from .primary_campaign_assembly import _read,_reference
    root=Path(root).resolve()
    if type(budgets) is not dict or set(budgets)!=set(SPECS) or any(type(b) is not TextBudget for b in budgets.values()):raise ValueError('Explicit budgets for both exactly reviewed validation families required')
    for b in budgets.values():b.__post_init__()
    sr=_reference(source_snapshot);before=_read(root,sr)
    refs={k:_reference(dict(path=p,sha256=h)) for k,(p,h) in dict(review=REVIEW,enrollment=ENROLLMENT,vocabulary=PINS['payload_vocabulary'],fit=PINS['payload_fit']).items()};raws={k:_read(root,r) for k,r in refs.items()}
    proof=json.loads(raws['enrollment']);fit=json.loads(raws['fit'])
    if proof['production_evaluation_enrollment'] is not False or proof['global_exposure_accepted'] is not False or proof['exact_validation_rows']!=2100 or proof['exact_token_and_eos_targets']!=5553057 or proof['scoped_review']['sha256']!=REVIEW[1] or proof['vocabulary_sha256']!=refs['vocabulary'].sha256:raise ValueError('Exact scoped pending validation receipt required')
    if fit.get('status')!='complete-source-fit' or fit.get('target_fields_used') is not False or fit.get('heldout_rows_used') is not False or fit['payload_vocabulary_artifact']['sha256']!=refs['vocabulary'].sha256:raise ValueError('Source-only fitted codec required')
    cref=_reference({k:proof['coordinate_digest_ledger'][k] for k in ('path','sha256')});ledger=json.loads(_read(root,cref))
    view=NativeTextRoleView(root/REVIEW[0],REVIEW[1],root=root);codec=PubChemByteCodec(AnnotationVocabulary(raws['vocabulary'].decode()).tokenizer())
    material={f:{'train':[],'validation':[]} for f in SPECS};used=dict.fromkeys(SPECS,0)
    for issued in view.iter_rows(roles=('train','validation'),families=tuple(SPECS)):
        row=issued.native_row;f=row.source_family;b=budgets[f];role=issued.prospective_role
        view.require_row(issued,role=role)
        if row.original_role in ('test','evaluation_only') or row.evaluation_only:raise ValueError('Original test/evaluation-only row cannot enter scoped bridge')
        used[f]+=len(row.full_target_utf8);material[f][role].append(issued)
        if used[f]>b.max_materialized_target_bytes or sum(map(len,material[f].values()))>b.max_materialized_rows:raise ValueError('Explicit one-time native role materialization budget exceeded')
    routes=[]
    for f,(_,train_n,val_n,target_n) in SPECS.items():
        train=material[f]['train'];val=material[f]['validation'];expected=[r for r in ledger if r['source_family']==f];by_coord={digest(r.native_row.source_coordinate_json):r for r in val};observed=[]
        if len(train)!=train_n or len(val)!=val_n or len(by_coord)!=val_n:raise ValueError('Exact reviewed per-role accounting differs')
        ordered=[]
        for coordinate in expected:
            issued=by_coord[coordinate['coordinate_sha256']];row=issued.native_row;tokens=len(codec.tokenizer.encode(row.full_target_utf8))+1
            actual=dict(source_family=f,task=row.condition_inputs()['task'],coordinate_sha256=digest(row.source_coordinate_json),condition_sha256=digest(row.conditions_json),native_target_sha256=digest(row.full_target_utf8),native_group_key_digests=sorted(digest(x.encode()) for x in row.group_keys),observed_tokens_and_eos=tokens,original_role=row.original_role,prospective_role='validation')
            if actual!=coordinate:raise ValueError('Exact source validation coordinate/content/role/token count changed')
            ordered.append(issued);observed.append(tokens)
        if sum(observed)!=target_n:raise ValueError('Exact validation denominator differs')
        routes.append(ReviewedTextCampaignRoute(view,codec,f,sr.sha256,budgets[f],train,ordered,expected))
    if _read(root,sr)!=before or any(_read(root,r)!=raws[k] for k,r in refs.items()):raise ValueError('Pinned text assets changed')
    return tuple(routes)
