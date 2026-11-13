"""Learned context-to-complete-forest keys and a bounded train-only L2 index.

Keys are approximate semantic summaries, not directed-analogy proofs. The
existing complete-path categorical/topological verifier remains the reranker.
Unknown pairs are excluded, not fabricated negatives. Group IDs must come from
the separately qualified source policy; declaring a digest does not prove closure.
"""
from dataclasses import dataclass,asdict,replace
from hashlib import sha256
import copy,math
import torch
from torch import nn
from torch.nn import functional as F
from mora3.data.schema import GraphRecord
from mora3.data.tasks import ConditionalTask
from mora3.reasoning.flow_forest_refinement import ForestState,forest_fingerprint
from .forest_flow import (NativeForestFlowHead,_sha,_wire,_linear,_parameter_versions,
                          forest_flow_parameter_roles)
from .mixed_payload_projection import exact_wire


@dataclass(frozen=True)
class ForestAnalogySpec:
    dimension: int
    group_policy_sha256: str
    temperature: float=.1
    max_memory: int=2048
    max_queries: int=256
    max_centers: int=128

    def __post_init__(self):
        _sha(self.group_policy_sha256)
        for n,limit in((self.dimension,1024),(self.max_memory,65536),(self.max_queries,1024),(self.max_centers,2048)):
            if type(n)is not int or not 1<=n<=limit:raise ValueError('Exact bounded analogy dimensions/resources required')
        if type(self.temperature)not in(int,float)or not math.isfinite(self.temperature)or self.temperature<=0:
            raise ValueError('Positive finite contrastive temperature required')


@dataclass(frozen=True)
class ForestAnalogyItem:
    identity: str
    record: GraphRecord
    task: ConditionalTask
    forest: ForestState | None
    groups: tuple[str,...]
    split: str
    group_policy_sha256: str
    evidence_sha256: str

    def __post_init__(self):
        if type(self.identity)is not str or not 0<len(self.identity)<=4096:
            raise ValueError('Bounded native item identity required')
        if type(self.record)is not GraphRecord or type(self.task)is not ConditionalTask:
            raise ValueError('Exact native problem/task required')
        if self.forest is not None and type(self.forest)is not ForestState:
            raise ValueError('Complete native forest or absent query target required')
        if(type(self.groups)is not tuple or not self.groups or len(set(self.groups))!=len(self.groups)
            or any(type(x)is not str or not 0<len(x)<=4096 for x in self.groups)):
            raise ValueError('Explicit nonempty unique biological leakage groups required')
        if self.split not in('train','validation','test'):raise ValueError('Original split required')
        for value in(self.group_policy_sha256,self.evidence_sha256):_sha(value)


class NativeForestAnalogyHead(nn.Module):
    def __init__(self,field,spec):
        super().__init__()
        if type(field)is not NativeForestFlowHead or type(spec)is not ForestAnalogySpec:
            raise ValueError('Actual registered forest field and analogy specification required')
        field.get_extra_state();spec.__post_init__()
        self.field=field;self.spec=spec;self._field_reference=(field,)
        self.query_readout=nn.Linear(field.width,spec.dimension)
        self.memory_readout=nn.Linear(field.width,spec.dimension)
        with torch.no_grad():
            for layer in(self.query_readout,self.memory_readout):
                layer.weight.mul_(math.sqrt(field.width/field.base_width))
                layer.bias.mul_(math.sqrt(field.width/field.base_width))
        for layer in(self.query_readout,self.memory_readout):
            layer.to(dtype=field.readout.weight.dtype,device=field.readout.weight.device)
        self._configuration=self._current_configuration()
        self._topology=tuple((n,type(m))for n,m in self.named_modules(remove_duplicate=False))

    def _current_configuration(self):
        return dict(schema='Mora3.NativeForestAnalogy/v1',spec=asdict(self.spec),field=self.field.get_extra_state(),
            query='condition-only-native-geometry-graph-token',memory='mean-of-all-complete-forest-node-tokens-at-t1',
            shared_prefix_count='each-node-once',metric='L2-unit-normalized-learned-keys',
            unknown_pairs='masked-out-not-negatives',directed_semantic_mapping_verified=False,training_admission=False)

    def get_extra_state(self):
        self.spec.__post_init__()
        if(self.field is not self._field_reference[0]
            or not _linear(self.query_readout,self.field.width,self.spec.dimension)
            or not _linear(self.memory_readout,self.field.width,self.spec.dimension)
            or self._topology!=tuple((n,type(m))for n,m in self.named_modules(remove_duplicate=False))
            or not exact_wire(self._configuration,self._current_configuration())):
            raise ValueError('Native analogy field/source/axis/operator drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('Analogy checkpoint specification mismatch')

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if strict is not True or assign is not False:raise ValueError('Strict non-assignment analogy recovery required')
        return super().load_state_dict(state_dict,strict=True,assign=False)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        self.get_extra_state();expected=self.state_dict()
        if(local_metadata.get('assign_to_params_buffers',False)is not False
            or {k for k in state_dict if k.startswith(prefix)}!={prefix+k for k in expected}):
            raise ValueError('Exact complete analogy checkpoint keys and original parameter identities required')
        for name,live in expected.items():
            incoming=state_dict[prefix+name]
            if isinstance(live,torch.Tensor):
                if(not isinstance(incoming,torch.Tensor)or incoming.is_meta or incoming.shape!=live.shape
                    or incoming.dtype!=live.dtype or incoming.layout!=live.layout):
                    raise ValueError('Analogy checkpoint tensor specification mismatch')
            elif not exact_wire(live,incoming):raise ValueError('Analogy checkpoint source/operator sidecar mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _key(self,layer,hidden):
        value=layer(F.silu(hidden)*(self.field.readout_mult/(self.field.width/self.field.base_width)))
        if not torch.isfinite(value).all():raise ValueError('Nonfinite analogy key')
        return F.normalize(value,p=2,dim=-1,eps=1e-12)

    def query(self,record,task):
        self.get_extra_state()
        return self._key(self.query_readout,self.field.encode_context(record,task))

    def _check_item(self,item,*,memory=False,training=False):
        if type(item)is not ForestAnalogyItem:raise ValueError('Exact source-bound analogy item required')
        item.__post_init__()
        if item.group_policy_sha256!=self.spec.group_policy_sha256:
            raise ValueError('Analogy grouping policy mismatch')
        if (training or memory)and item.split!='train':raise ValueError('Learning and retrieval memory must be train-only')
        if memory and(type(item.forest)is not ForestState or not item.forest.is_complete):
            raise ValueError('Complete forest memory required')

    def memory_key(self,item):
        self.get_extra_state();self._check_item(item,memory=True)
        forest=item.forest
        x=forest.states.detach()[None]
        h=self.field.encode_forest(item.record,item.task,x,x.new_ones(1),forest)
        return self._key(self.memory_readout,h.mean(dim=1))


def forest_analogy_loss(head,queries,memories,positive,known):
    """Multi-positive contrastive objective with EXPLICIT supported negatives.

    All examples are train-only and positive pairs must be disjoint under the
    declared grouping policy. Unknown pairs cannot change the denominator.
    Source admission and scientific correspondence labels are external evidence.
    """
    if type(head)is not NativeForestAnalogyHead:raise ValueError('Exact native analogy head required')
    head.get_extra_state()
    if(type(queries)is not tuple or type(memories)is not tuple
        or not 1<=len(queries)<=head.spec.max_queries or not 1<=len(memories)<=head.spec.max_memory):
        raise ValueError('Bounded explicit query and whole-forest memory tuples required')
    device=head.query_readout.weight.device
    if(any(not isinstance(x,torch.Tensor)or x.dtype!=torch.bool or x.shape!=(len(queries),len(memories))or x.device!=device for x in(positive,known))):
        raise ValueError('Exact same-device positive and known-pair masks required')
    for q in queries:head._check_item(q,training=True)
    for m in memories:head._check_item(m,memory=True)
    disjoint=torch.tensor([[not(set(q.groups)&set(m.groups))for m in memories]for q in queries],dtype=torch.bool,device=device)
    eligible=known&disjoint
    if (positive&~eligible).any()or not positive.any(dim=1).all()or not(eligible&~positive).any(dim=1).all():
        raise ValueError('Each query needs explicit disjoint positives and supported negatives; no unknown labels')
    q=torch.cat([head.query(x.record,x.task)for x in queries])
    m=torch.cat([head.memory_key(x)for x in memories])
    logits=q@m.T/head.spec.temperature
    loss=(torch.logsumexp(logits.masked_fill(~eligible,float('-inf')),1)
         -torch.logsumexp(logits.masked_fill(~positive,float('-inf')),1)).mean()
    return dict(loss=loss,known_pair_count=int(eligible.sum()),positive_pair_count=int(positive.sum()),
                group_completeness_verified=False,training_admission=False)


def _tensor_sha(value):
    return sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class ForestAnalogyIndex:
    """Deterministic farthest-pivot partition, no extra learned index optimizer.

    Probing fewer cells is approximate; probing all cells or exact=True evaluates
    every eligible key. Distances within a shortlist are actual Euclidean L2.
    The index owns cloned forests and binds weights/coordinates/group policy.
    It returns candidates for full directed/physical checking, not certified
    analogies. No query target forest is accessed or required.
    """
    @torch.no_grad()
    def __init__(self,head,items,*,centers=None):
        if type(head)is not NativeForestAnalogyHead:raise ValueError('Exact native analogy owner required')
        head.get_extra_state()
        if type(items)is not tuple or not 1<=len(items)<=head.spec.max_memory:
            raise ValueError('Bounded train-only memory tuple required')
        for item in items:head._check_item(item,memory=True)
        if len({x.identity for x in items})!=len(items):raise ValueError('Duplicate memory identity')
        if sum(x.forest.states.numel()for x in items)>16_777_216:raise ValueError('Complete memory exceeds retained-coordinate budget')
        centers=min(head.spec.max_centers,len(items))if centers is None else centers
        if type(centers)is not int or not 1<=centers<=min(head.spec.max_centers,len(items)):
            raise ValueError('Exact bounded center count required')
        self.head=head;self._configuration=_wire(head.get_extra_state());self._versions=_parameter_versions(head)
        self._items=tuple(replace(x,forest=x.forest.copy())for x in sorted(items,key=lambda x:x.identity))
        self._keys=torch.cat([head.memory_key(x)for x in self._items]).detach().clone()
        self._key_sha=_tensor_sha(self._keys)
        selected=[0];nearest=(self._keys-self._keys[0]).square().sum(1)
        while len(selected)<centers:
            next_index=int(nearest.argmax())
            if float(nearest[next_index])==0:break
            selected.append(next_index)
            nearest=torch.minimum(nearest,(self._keys-self._keys[next_index]).square().sum(1))
        self._pivots=self._keys[selected].clone()
        self._assignments=torch.cdist(self._keys,self._pivots,p=2).argmin(1)
        self._partition_sha=(_tensor_sha(self._pivots),_tensor_sha(self._assignments))
        weight_sha=sha256(self._configuration)
        for name,p in head.named_parameters():weight_sha.update(_wire([name,list(p.shape),str(p.dtype),_tensor_sha(p)]))
        self._item_sha=tuple(forest_fingerprint(x.forest)for x in self._items)
        self.sha256=sha256(_wire(dict(schema='Mora3.NativeForestAnalogyIndex/v1',weights=weight_sha.hexdigest(),
            memory=[dict(identity=x.identity,groups=x.groups,evidence=x.evidence_sha256,forest=h)for x,h in zip(self._items,self._item_sha)],
            keys=self._key_sha,partition=self._partition_sha))).hexdigest()
        if self._versions!=_parameter_versions(head):raise ValueError('Model changed while building analogy index')

    def _current(self):
        if(self._versions!=_parameter_versions(self.head)or self._configuration!=_wire(self.head.get_extra_state())
            or self._key_sha!=_tensor_sha(self._keys)
            or self._partition_sha!=(_tensor_sha(self._pivots),_tensor_sha(self._assignments))
            or self._item_sha!=tuple(forest_fingerprint(x.forest)for x in self._items)):
            raise ValueError('Analogy index is stale or changed; rebuild explicitly from current train-only memory')

    @torch.no_grad()
    def search(self,query,*,k,probes=1,exact=False):
        self._current();self.head._check_item(query)
        if(type(k)is not int or not 1<=k<=self.head.spec.max_memory
            or type(probes)is not int or probes<1 or type(exact)is not bool):
            raise ValueError('Exact bounded search budgets required')
        key=self.head.query(query.record,query.task)
        eligible=torch.tensor([not(set(x.groups)&set(query.groups))for x in self._items],device=key.device,dtype=torch.bool)
        searched_centers=len(self._pivots)if exact else min(probes,len(self._pivots))
        if not exact:
            order=torch.argsort(torch.linalg.vector_norm(self._pivots-key,dim=1),stable=True)[:searched_centers]
            eligible&=(self._assignments[:,None]==order[None]).any(1)
        positions=eligible.nonzero().flatten()
        distance=torch.linalg.vector_norm(self._keys[positions]-key,dim=1)
        chosen=torch.argsort(distance,stable=True)[:k]
        identities=tuple(self._items[int(positions[i])].identity for i in chosen)
        return dict(identities=identities,distances=tuple(float(distance[i])for i in chosen),shortfall=k-len(chosen),
            examined_candidates=len(positions),searched_centers=searched_centers,
            exact_global_retrieval=exact or searched_centers==len(self._pivots),index_sha256=self.sha256,
            metric='L2-normalized-learned-keys',directed_semantic_mapping_verified=False,
            group_completeness_verified=False,training_admission=False)

    def exemplar(self,identity):
        self._current()
        for item in self._items:
            if item.identity==identity:return replace(item,forest=item.forest.copy())
        raise KeyError(identity)


def native_forest_analogy_objective(owner,*args,**kwargs):
    from .full_owner import SourceBoundOwner
    if type(owner)is not SourceBoundOwner:raise ValueError('Exact registered complete owner required')
    owner.require_training_admission()
    if type(getattr(owner,'forest_analogy',None))is not NativeForestAnalogyHead:
        raise ValueError('Factory-registered native analogy head required')
    return forest_analogy_loss(owner.forest_analogy,*args,**kwargs)


def forest_analogy_parameter_roles(model,*,identifier_axis):
    from .parameterization import Axis,ParameterRole,parameter_ledger
    from .mu_optimizer import _storage_signature
    if type(model)is not NativeForestAnalogyHead:raise ValueError('Exact native analogy head required')
    model.get_extra_state()
    roles={'field.'+n:r for n,r in forest_flow_parameter_roles(model.field,identifier_axis=identifier_axis).items()}
    hidden=Axis('width','hidden');out=Axis('finite',sha256(_wire(['forest-analogy-key-v1',model.spec.dimension])).hexdigest())
    for name in('query_readout','memory_readout'):
        roles[name+'.weight']=ParameterRole('output',(out,hidden));roles[name+'.bias']=ParameterRole('vector',(out,))
    _storage_signature(model);parameter_ledger(model,roles)
    return roles
