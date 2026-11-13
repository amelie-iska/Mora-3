"""Flow proposals -> non-destructive FoT corrections -> multiple verified outputs.

FoT (Bi et al., ICML2025, sections3.1-3.3) supplies sparse activation and
score-triggered correction. Latent flows, subtree replacement and multi-output
selection are Mora-3 adaptations. This is inference, not a trained head or an
on-policy GFlowNet sampler. Verifiers/decoder/velocity field are caller-bound;
no default biological-validity oracle or density is invented.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from types import MappingProxyType
import torch
from mora3.model.flow_matching import euler_flow


def _sha(value):
    if type(value)is not str or re.fullmatch('[0-9a-f]{64}',value)is None:
        raise ValueError('Exact revision/evidence SHA256 required')


def _number(value):
    if type(value)not in(int,float)or not math.isfinite(value):
        raise ValueError('Finite literal score/distance required')


def _wire(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _tensor_bytes(value):
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _tree(parents,statuses):
    if type(parents)is not tuple or type(statuses)is not tuple or not parents or len(parents)!=len(statuses):
        raise ValueError('Exact nonempty ordered forest skeleton required')
    children=set()
    for i,parent in enumerate(parents):
        if type(parent)is not int or not -1<=parent<i:
            raise ValueError('Forest parent must precede child; no cycles or future parents')
        if parent>=0:children.add(parent)
        if statuses[i]not in('open','terminal','censored','invalid'):
            raise ValueError('Explicit forest node disposition required')
    if any((status=='open')!=(i in children)for i,status in enumerate(statuses)):
        raise ValueError('Internal nodes must be open; leaves need explicit completion/censoring')


@dataclass(frozen=True)
class ForestState:
    node_ids: tuple[str,...]
    parents: tuple[int,...]
    statuses: tuple[str,...]
    states: torch.Tensor
    clamped: tuple[int,...]
    directions: tuple[str,...]
    context_sha256: str
    coordinate_sha256: str
    skeleton_origin: str='declared'

    def __post_init__(self):
        _tree(self.parents,self.statuses)
        n=len(self.parents)
        if (type(self.node_ids)is not tuple or len(self.node_ids)!=n or len(set(self.node_ids))!=n
                or any(type(k)is not str or not k or len(k)>4096 for k in self.node_ids)):
            raise ValueError('Exact unique bounded node identities required')
        if (not isinstance(self.states,torch.Tensor)or self.states.ndim!=2 or self.states.shape[0]!=n
                or not 1<=n<=512 or not 1<=self.states.shape[1]<=256
                or not self.states.is_floating_point()or not torch.isfinite(self.states).all()):
            raise ValueError('Finite bounded complete forest embeddings [nodes,dimensions] required')
        if (type(self.clamped)is not tuple or self.clamped!=tuple(sorted(set(self.clamped)))
                or any(type(i)is not int or not 0<=i<n for i in self.clamped)
                or not {i for i,p in enumerate(self.parents)if p==-1}<=set(self.clamped)):
            raise ValueError('Known condition roots must be explicitly clamped')
        if (type(self.directions)is not tuple or len(self.directions)!=n
                or any(d not in('root','computational','temporal','observational','interventional')for d in self.directions)
                or any((d=='root')!=(p==-1)for d,p in zip(self.directions,self.parents))):
            raise ValueError('Exact directed evidence categories required')
        for value in(self.context_sha256,self.coordinate_sha256):_sha(value)
        if self.skeleton_origin not in('declared','generated'):
            raise ValueError('Hidden target topology cannot be a sampler condition')

    def copy(self):
        return ForestState(self.node_ids,self.parents,self.statuses,self.states.detach().clone(),
            self.clamped,self.directions,self.context_sha256,self.coordinate_sha256,self.skeleton_origin)

    @property
    def is_complete(self):
        return all(s in('open','terminal')for s in self.statuses)


def forest_fingerprint(forest):
    if type(forest)is not ForestState:raise ValueError('Exact complete forest required')
    forest.__post_init__()
    meta=dict(nodes=forest.node_ids,parents=forest.parents,statuses=forest.statuses,
        clamped=forest.clamped,directions=forest.directions,context=forest.context_sha256,
        coordinates=forest.coordinate_sha256,origin=forest.skeleton_origin,
        shape=tuple(forest.states.shape),dtype=str(forest.states.dtype))
    return sha256(_wire(meta)+b'\0'+_tensor_bytes(forest.states)).hexdigest()


@dataclass(frozen=True)
class ForestAssessment:
    candidate_sha256: str
    context_sha256: str
    evaluator_sha256: str
    status: str
    score: float
    node_scores: tuple[float,...]
    evidence_sha256: str
    repairable: bool=False

    def __post_init__(self):
        for x in(self.candidate_sha256,self.context_sha256,self.evaluator_sha256,self.evidence_sha256):_sha(x)
        _number(self.score)
        if self.status not in('pass','refused','unknown')or type(self.repairable)is not bool:
            raise ValueError('Explicit verdict and repairability required')
        if type(self.node_scores)is not tuple:raise ValueError('Ordered node scores required')
        for x in self.node_scores:_number(x)


@dataclass(frozen=True)
class DecodeRefusal:
    """An incomplete decoder result is not a completed output or a zero score."""
    candidate_sha256: str
    context_sha256: str
    decoder_sha256: str
    reason: str
    evidence_sha256: str
    repairable: bool=False

    def __post_init__(self):
        for value in(self.candidate_sha256,self.context_sha256,self.decoder_sha256,self.evidence_sha256):_sha(value)
        if type(self.reason)is not str or not 0<len(self.reason)<=1024 or type(self.repairable)is not bool:
            raise ValueError('Explicit bounded decoder failure and repairability required')


@dataclass(frozen=True)
class DecodedForest:
    tokens: tuple[int,...]
    semantic_key: str
    candidate_sha256: str
    context_sha256: str
    decoder_sha256: str

    def __post_init__(self):
        if (type(self.tokens)is not tuple or not 1<=len(self.tokens)<=262144
                or any(type(i)is not int or i<0 for i in self.tokens)
                or type(self.semantic_key)is not str or not self.semantic_key or len(self.semantic_key)>131072):
            raise ValueError('Complete token output and canonical output identity required')
        for x in(self.candidate_sha256,self.context_sha256,self.decoder_sha256):_sha(x)


def decoded_fingerprint(output):
    if type(output)is not DecodedForest:raise ValueError('Exact decoded forest required')
    output.__post_init__()
    return sha256(_wire(dict(tokens=output.tokens,identity=output.semantic_key,
        candidate=output.candidate_sha256,context=output.context_sha256,decoder=output.decoder_sha256))).hexdigest()


@dataclass(frozen=True)
class DecodedAssessment:
    decoded_sha256: str
    context_sha256: str
    evaluator_sha256: str
    status: str
    score: float
    evidence_sha256: str
    repairable: bool=False

    def __post_init__(self):
        for x in(self.decoded_sha256,self.context_sha256,self.evaluator_sha256,self.evidence_sha256):_sha(x)
        _number(self.score)
        if self.status not in('pass','refused','unknown')or type(self.repairable)is not bool:
            raise ValueError('Explicit decoded verdict and repairability required')


@dataclass(frozen=True)
class ForestRevision:
    forest: ForestState
    operation: str
    cut_node_id: str


@dataclass(frozen=True)
class RefinementPolicy:
    initial_candidates: int
    active_candidates: int
    return_outputs: int
    edit_rounds: int
    edits_per_candidate: int
    total_candidates: int
    minimum_score: float
    minimum_decoded_score: float
    correction_threshold: float
    minimum_distance: float
    post_decode_rounds: int=0

    def __post_init__(self):
        integers=(self.initial_candidates,self.active_candidates,self.return_outputs,
            self.edit_rounds,self.edits_per_candidate,self.total_candidates,self.post_decode_rounds)
        if (any(type(x)is not int for x in integers)
                or not 2<=self.return_outputs<=self.active_candidates<=self.initial_candidates<=self.total_candidates<=1024
                or not 0<=self.edit_rounds<=64 or not 0<=self.post_decode_rounds<=64
                or not 1<=self.edits_per_candidate<=64):
            raise ValueError('Explicit bounded multi-candidate/multi-output policy required')
        for x in(self.minimum_score,self.minimum_decoded_score,self.correction_threshold,self.minimum_distance):_number(x)
        if self.minimum_distance<0:raise ValueError('Nonnegative minimum distance required')


@torch.no_grad()
def sample_flow_forests(template,field,*,count,steps,generator,field_sha256):
    """Batched whole-skeleton generation using the native Euler solver.

    Field receives (states[B,N,D], time[B], condition-only template). Unknown
    template entries are erased before any callback. Independent initial draws
    are explicit, not guarantees of high quality or statistical output diversity.
    """
    forest_fingerprint(template);_sha(field_sha256)
    if (not callable(field)or type(count)is not int or not 1<=count<=256
            or type(steps)is not int or not 1<=steps<=4096
            or type(generator)is not torch.Generator
            or torch.device(generator.device)!=template.states.device
            or count*template.states.numel()>16_777_216):
        raise ValueError('Bounded field sampling and matching explicit generator required')
    condition=template.copy();mask=torch.zeros_like(condition.states,dtype=torch.bool)
    mask[list(condition.clamped)]=True
    condition.states[~mask]=0
    before=generator.get_state().clone()
    initial=torch.randn((count,*condition.states.shape),generator=generator,
        dtype=condition.states.dtype,device=condition.states.device)
    initial=torch.where(mask[None],condition.states[None],initial)
    n,d=condition.states.shape
    def velocity(flat,t):
        x=flat.reshape(count,n,d).clone();x=torch.where(mask[None],condition.states[None],x)
        local=condition.copy();pin=forest_fingerprint(local);unchanged=x.clone();tt=t.clone()
        v=field(x,tt,local)
        if forest_fingerprint(local)!=pin or not torch.equal(x,unchanged)or not torch.equal(tt,t):
            raise ValueError('Velocity callback mutated its conditions/state/time')
        if (not isinstance(v,torch.Tensor)or v.shape!=x.shape or v.dtype!=x.dtype or v.device!=x.device
                or not torch.isfinite(v).all()):raise ValueError('Complete finite forest velocity required')
        return torch.where(mask[None],torch.zeros_like(v),v).reshape(count,-1)
    values=euler_flow(velocity,initial.reshape(count,-1),steps=steps).reshape(count,n,d)
    forests=[]
    for value in values:
        f=condition.copy();f.states.copy_(torch.where(mask,condition.states,value));forests.append(f)
    return dict(forests=tuple(forests),generator_before=before,generator_after=generator.get_state().clone(),
        field_sha256=field_sha256,velocity_evaluations=steps,candidate_equivalent_evaluations=count*steps,
        gfn_on_policy=False,reward_proportionality_verified=False,training_admission=False)


def _descendants(forest,key):
    if key not in forest.node_ids:raise ValueError('Unknown edit root')
    cut=forest.node_ids.index(key);removed={cut}
    for i,p in enumerate(forest.parents):
        if p in removed:removed.add(i)
    return removed


@torch.no_grad()
def replace_subtree(forest,cut_node_id,new_states,new_parents,new_statuses):
    """Replace ALL descendants in a fresh forest; never splice stale downstream states.

    New internal edges are computational proposals, not observed causal links.
    The original forest and unrelated branches remain intact. Whole-forest
    scoring must be repeated after replacement; no old certificate transfers.
    """
    pin=forest_fingerprint(forest);removed=_descendants(forest,cut_node_id)
    if removed.intersection(forest.clamped):raise ValueError('Subtree edit would remove a clamped condition')
    _tree(new_parents,new_statuses)
    if new_parents[0]!=-1 or any(p==-1 for p in new_parents[1:]):raise ValueError('One replacement subtree root required')
    if (not isinstance(new_states,torch.Tensor)or new_states.shape!=(len(new_parents),forest.states.shape[1])
            or new_states.dtype!=forest.states.dtype or new_states.device!=forest.states.device
            or not torch.isfinite(new_states).all()):raise ValueError('Exact complete replacement embeddings required')
    cut=forest.node_ids.index(cut_node_id)
    before=[i for i in range(cut)if i not in removed]
    after=[i for i in range(cut,len(forest.node_ids))if i not in removed]
    prefix=len(before);patch_size=len(new_parents)
    lookup={old:i for i,old in enumerate(before)}
    lookup.update({old:prefix+patch_size+i for i,old in enumerate(after)})
    patch=sha256(pin.encode()+_wire([new_parents,new_statuses])+_tensor_bytes(new_states)).hexdigest()
    names=tuple('revision:'+patch+':'+str(i)for i in range(patch_size))
    ids=tuple(forest.node_ids[i]for i in before)+names+tuple(forest.node_ids[i]for i in after)
    def old_parents(indices):return tuple(-1 if forest.parents[i]<0 else lookup[forest.parents[i]]for i in indices)
    parents=old_parents(before)+(lookup[forest.parents[cut]],)+tuple(prefix+p for p in new_parents[1:])+old_parents(after)
    statuses=tuple(forest.statuses[i]for i in before)+new_statuses+tuple(forest.statuses[i]for i in after)
    directions=tuple(forest.directions[i]for i in before)+(forest.directions[cut],)+('computational',)*(patch_size-1)+tuple(forest.directions[i]for i in after)
    return ForestState(ids,parents,statuses,torch.cat((forest.states[before],new_states.detach().clone(),forest.states[after])),
        tuple(sorted(lookup[i]for i in forest.clamped)),directions,
        forest.context_sha256,forest.coordinate_sha256,forest.skeleton_origin)



def _same_conditions(original,other):
    forest_fingerprint(other)
    if (original.context_sha256!=other.context_sha256 or original.coordinate_sha256!=other.coordinate_sha256
            or original.states.dtype!=other.states.dtype or original.states.device!=other.states.device
            or original.states.shape[1]!=other.states.shape[1]or original.skeleton_origin!=other.skeleton_origin):
        raise ValueError('Changed context/coordinate system or source condition')
    expected={original.node_ids[i]:original.states[i]for i in original.clamped}
    actual={other.node_ids[i]:other.states[i]for i in other.clamped}
    if actual.keys()!=expected.keys()or any(not torch.equal(expected[k],actual[k])for k in expected):
        raise ValueError('Edit changed clamped source conditions')


def _checked_revision(parent,proposal):
    if (type(proposal)is not ForestRevision or proposal.operation not in
            ('rule_correction','subtree_regeneration','branch_expansion','post_decode_correction')):
        raise ValueError('Explicit supported forest revision required')
    child=proposal.forest;_same_conditions(parent,child)
    removed=_descendants(parent,proposal.cut_node_id)
    if removed.intersection(parent.clamped):raise ValueError('Edit crosses a clamped source node')
    keep={parent.node_ids[i]:i for i in range(len(parent.node_ids))if i not in removed}
    child_index={k:i for i,k in enumerate(child.node_ids)}
    if not keep.keys()<=child_index.keys():raise ValueError('Edit removed an unrelated branch')
    if set(parent.node_ids[i]for i in removed)&child_index.keys():
        raise ValueError('Replacement must invalidate the complete original descendant subtree')
    for key,i in keep.items():
        j=child_index[key];pp=None if parent.parents[i]<0 else parent.node_ids[parent.parents[i]]
        cp=None if child.parents[j]<0 else child.node_ids[child.parents[j]]
        if pp!=cp or parent.statuses[i]!=child.statuses[j]or parent.directions[i]!=child.directions[j]or not torch.equal(parent.states[i],child.states[j]):
            raise ValueError('Edit changed an unrelated ancestor or sibling branch')
    new=set(child_index)-set(keep);cut=parent.node_ids.index(proposal.cut_node_id)
    attach=parent.node_ids[parent.parents[cut]];roots=[]
    for key in new:
        j=child_index[key];p=child.parents[j];pk=None if p<0 else child.node_ids[p]
        if pk not in new:roots.append((key,pk))
        elif child.directions[j]!='computational':raise ValueError('Generated edits cannot invent intervention evidence')
    if len(roots)!=1 or roots[0][1]!=attach:raise ValueError('One fully connected replacement subtree required')
    if child.directions[child_index[roots[0][0]]]!=parent.directions[cut]:
        raise ValueError('Replacement changed its boundary direction evidence')


def _distance(left,right,custom):
    if custom is not None:
        a,b=left.copy(),right.copy();ap,bp=forest_fingerprint(a),forest_fingerprint(b)
        value=custom(a,b);_number(value)
        if value<0 or forest_fingerprint(a)!=ap or forest_fingerprint(b)!=bp:
            raise ValueError('Invalid or mutating diversity comparison')
        return value
    if (left.parents,left.clamped,left.statuses,left.directions)!=(right.parents,right.clamped,right.statuses,right.directions):
        return None  # Different declared structural strata, not asserted isometric.
    value=torch.linalg.vector_norm(left.states.double()-right.states.double()).item()
    _number(value);return value


@torch.no_grad()
def run_flow_seeded_refinement(template,field,*,steps,generator,policy,assess,edit,decode,
        assess_decoded,field_sha256,evaluator_sha256,decoder_sha256,decoded_evaluator_sha256,distance=None):
    """Bounded inference retaining elites, edit lineage, refusals and raw proposals.

    Each callback sees a detached copy; mutation is refused. assess returns a
    full-forest certificate, not an endpoint-only check. Decoding receives only a
    complete latent forest, then assess_decoded independently checks that output.
    Missing K qualified distinct outputs yields an explicit shortfall.
    """
    if type(policy)is not RefinementPolicy:raise ValueError('Exact refinement policy required')
    policy.__post_init__();forest_fingerprint(template)
    if any(not callable(c)for c in(field,assess,edit,decode,assess_decoded))or(distance is not None and not callable(distance)):
        raise ValueError('Explicit field, verifier, editor, decoder and decoded verifier required')
    for pin in(field_sha256,evaluator_sha256,decoder_sha256,decoded_evaluator_sha256):_sha(pin)
    samples=sample_flow_forests(template,field,count=policy.initial_candidates,steps=steps,generator=generator,field_sha256=field_sha256)
    records=[];events=[];seen={};attempted=0;decode_cache={}
    def add(forest,parent,operation,round_index):
        nonlocal attempted
        if attempted>=policy.total_candidates:raise ValueError('Candidate budget exceeded')
        attempted+=1;_same_conditions(template,forest);candidate=forest.copy();pin=forest_fingerprint(candidate)
        if pin in seen:
            events.append(dict(kind='duplicate_proposal',budget_slot=attempted,parent_id=parent,original_id=seen[pin]));return
        local=candidate.copy();decision=assess(local)
        if forest_fingerprint(local)!=pin:raise ValueError('Scorer mutated the full forest')
        if type(decision)is not ForestAssessment:raise ValueError('Exact forest assessment required')
        decision.__post_init__()
        if (decision.candidate_sha256!=pin or decision.context_sha256!=template.context_sha256
                or decision.evaluator_sha256!=evaluator_sha256 or len(decision.node_scores)!=len(candidate.node_ids)):
            raise ValueError('Assessment does not bind the current full forest/context/evaluator')
        identity=len(records);seen[pin]=identity
        records.append(dict(candidate_id=identity,forest=candidate,assessment=decision,parent_id=parent,
            operation=operation,round=round_index,candidate_sha256=pin))
        events.append(dict(kind='proposal',candidate_id=identity,parent_id=parent,operation=operation,
            round=round_index,status=decision.status,score=decision.score,budget_slot=attempted))
    for f in samples['forests']:add(f,None,'flow_proposal',0)
    initial_ids=tuple(r['candidate_id']for r in records)
    def active():
        order=sorted((r for r in records if r['assessment'].status=='pass'or r['assessment'].repairable),
            key=lambda r:(r['assessment'].status!='pass',-r['assessment'].score,r['candidate_id']))
        chosen=[]
        for row in order:
            if all((d:=_distance(row['forest'],p['forest'],distance))is None or d>=policy.minimum_distance for p in chosen):chosen.append(row)
            if len(chosen)==policy.active_candidates:break
        return chosen
    def revise(parents,round_index,post=False):
        before=len(records)
        for row in parents:
            count=min(policy.edits_per_candidate,policy.total_candidates-attempted)
            if not count:break
            f=row['forest'].copy();pin=forest_fingerprint(f);s=row['assessment']
            reason='post_decode'if post else('correct'if s.score<policy.correction_threshold or any(x<policy.correction_threshold for x in s.node_scores)else'improve')
            feedback=MappingProxyType(dict(decode_cache[row['candidate_id']]))if post else None
            proposals=edit(f,s,count=count,reason=reason,feedback=feedback)
            if forest_fingerprint(f)!=pin:raise ValueError('Editor mutated its parent forest')
            if type(proposals)not in(tuple,list)or len(proposals)>count:raise ValueError('Editor exceeded its explicit proposal budget')
            events.append(dict(kind='edit_call',parent_id=row['candidate_id'],reason=reason,round=round_index,proposals=len(proposals)))
            for proposal in proposals:
                _checked_revision(row['forest'],proposal);add(proposal.forest,row['candidate_id'],proposal.operation,round_index)
        return len(records)>before
    rounds=0
    for round_index in range(1,policy.edit_rounds+1):
        if attempted>=policy.total_candidates:break
        rounds=round_index
        if not revise(active(),round_index):break
    def decode_pool():
        for row in records:
            i=row['candidate_id'];s=row['assessment'];f=row['forest']
            if i in decode_cache or s.status!='pass' or s.score<policy.minimum_score or not f.is_complete:continue
            local=f.copy();pin=forest_fingerprint(local);out=decode(local)
            if forest_fingerprint(local)!=pin:raise ValueError('Decoder mutated the latent forest')
            if type(out)is DecodeRefusal:
                out.__post_init__()
                if out.candidate_sha256!=pin or out.context_sha256!=template.context_sha256 or out.decoder_sha256!=decoder_sha256:
                    raise ValueError('Decode refusal does not bind current forest/context/decoder')
                decode_cache[i]=dict(candidate_id=i,decoded=None,assessment=None,failure=out)
                events.append(dict(kind='decode_refused',candidate_id=i,reason=out.reason,repairable=out.repairable))
                continue
            if type(out)is not DecodedForest:raise ValueError('Exact complete decoded token forest required')
            out.__post_init__()
            if out.candidate_sha256!=pin or out.context_sha256!=template.context_sha256 or out.decoder_sha256!=decoder_sha256:
                raise ValueError('Output does not bind the current forest/context/decoder')
            local=f.copy();decision=assess_decoded(local,out)
            if forest_fingerprint(local)!=pin:raise ValueError('Decoded verifier mutated the forest')
            if type(decision)is not DecodedAssessment:raise ValueError('Exact decoded assessment required')
            decision.__post_init__()
            if (decision.decoded_sha256!=decoded_fingerprint(out)or decision.context_sha256!=template.context_sha256
                    or decision.evaluator_sha256!=decoded_evaluator_sha256):raise ValueError('Stale decoded assessment binding')
            decode_cache[i]=dict(candidate_id=i,decoded=out,assessment=decision)
            events.append(dict(kind='decoded',candidate_id=i,status=decision.status,score=decision.score))
        ranked=sorted((d for d in decode_cache.values()if d['assessment']is not None and d['assessment'].status=='pass'and d['assessment'].score>=policy.minimum_decoded_score),
            key=lambda d:(-d['assessment'].score,-records[d['candidate_id']]['assessment'].score,d['candidate_id']))
        outputs=[];keys=set();token_sets=set()
        for item in ranked:
            output=item['decoded'];row=records[item['candidate_id']]
            if output.semantic_key in keys or output.tokens in token_sets:continue
            if any((d:=_distance(row['forest'],records[p['candidate_id']]['forest'],distance))is not None and d<policy.minimum_distance for p in outputs):continue
            outputs.append(item);keys.add(output.semantic_key);token_sets.add(output.tokens)
            if len(outputs)==policy.return_outputs:break
        return tuple(outputs)
    outputs=decode_pool();post_rounds=0
    for number in range(1,policy.post_decode_rounds+1):
        if len(outputs)>=policy.return_outputs or attempted>=policy.total_candidates:break
        failed=[records[i]for i,d in decode_cache.items()if
            (d['assessment']is None and d['failure'].repairable)or
            (d['assessment']is not None and d['assessment'].repairable and
             (d['assessment'].status!='pass'or d['assessment'].score<policy.minimum_decoded_score))]
        failed=sorted(failed,key=lambda r:(-r['assessment'].score,r['candidate_id']))[:policy.active_candidates]
        if not failed:break
        post_rounds=number
        if not revise(failed,rounds+number,True):break
        outputs=decode_pool()
    return dict(candidates=tuple(records),initial_candidate_ids=initial_ids,outputs=outputs,
        decoded_candidates=tuple(decode_cache.values()),events=tuple(events),candidate_attempts=attempted,
        edit_rounds_completed=rounds,post_decode_rounds_completed=post_rounds,requested_outputs=policy.return_outputs,
        status='qualified-multiple-outputs'if len(outputs)==policy.return_outputs else'insufficient-qualified-outputs',
        velocity_evaluations=samples['velocity_evaluations'],candidate_equivalent_evaluations=samples['candidate_equivalent_evaluations'],
        generator_before=samples['generator_before'],generator_after=samples['generator_after'],
        field_sha256=field_sha256,evaluator_sha256=evaluator_sha256,decoder_sha256=decoder_sha256,
        decoded_evaluator_sha256=decoded_evaluator_sha256,context_sha256=template.context_sha256,
        inference_counts=dict(initial_batched_velocity_calls=samples['velocity_evaluations'],
            edit_callback_calls=sum(e['kind']=='edit_call'for e in events),
            full_forest_assessments=len(records),decoder_calls=len(decode_cache),
            decoded_assessments=sum(d['assessment']is not None for d in decode_cache.values()),total_internal_model_calls=None,
            internal_callback_costs='caller must measure; not inferred from solver steps'),
        coordinate_sha256=template.coordinate_sha256,gfn_on_policy=False,reward_proportionality_verified=False,
        training_admission=False,biological_validity_inferred=False)
