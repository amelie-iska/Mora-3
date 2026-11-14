"""Normalized native prefix policy for bounded, variable-size ordered forests.

Unlike the joint noisy-state flow, this policy sees only given task context and
already generated topology. Source teacher forcing supplies next-action labels,
not the completed target graph as predictor input. The finite grammar is a
Mora-3 construction; learned quality and a reward-proportional teacher are separate.
"""
from dataclasses import asdict,dataclass
from hashlib import sha256
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from mora3.reasoning.forest_topology import (TopologyBounds,prefix_state,
    canonical_forest_actions,maximum_forest_nodes,CLOSE,ADD_CHILD)
from mora3.reasoning.flow_forest_refinement import ForestState
from .forest_flow import (NativeForestFlowHead,forest_context_sha256,
    _condition_wire,_wire,_sha,_linear,_parameter_versions,forest_flow_parameter_roles)
from .mixed_payload_projection import exact_wire

_FEATURES=('depth/max_nodes','children/max_nodes','closed','on_stack','current',
           'condition_root','completed_roots/max_nodes')


@dataclass(frozen=True)
class ForestTopologySpec:
    bounds: TopologyBounds
    construction_policy_sha256: str
    max_batch_size: int=32
    max_total_decisions: int=8192

    def __post_init__(self):
        if type(self.bounds) is not TopologyBounds:
            raise ValueError('Exact finite ordered-forest grammar required')
        self.bounds.__post_init__();_sha(self.construction_policy_sha256)
        if (type(self.max_batch_size) is not int or not 1 <= self.max_batch_size <= 256
                or type(self.max_total_decisions) is not int or not 1 <= self.max_total_decisions <= 1_048_576):
            raise ValueError('Explicit bounded topology batch and decision budget required')


class NativeForestTopologyHead(nn.Module):
    def __init__(self,field,spec):
        super().__init__()
        if type(field) is not NativeForestFlowHead or type(spec) is not ForestTopologySpec:
            raise ValueError('Exact native shared field and topology specification required')
        spec.__post_init__();field.get_extra_state()
        if spec.bounds.max_nodes > field.spec.max_nodes:
            raise ValueError('Generated topology support must fit the registered field capacity')
        if field.parameterization != 'mup-forest-flow-v1':
            raise ValueError('Native topology width family currently requires applied ordinary-attention muP')
        self.field=field;self.spec=spec;self._field_reference=(field,)
        self.structure_projection=nn.Linear(len(_FEATURES),field.width)
        self.action_readout=nn.Linear(field.width,2)
        with torch.no_grad():
            scale=math.sqrt(field.width/field.base_width)
            self.action_readout.weight.mul_(scale);self.action_readout.bias.mul_(scale)
        ref=field.readout.weight
        for layer in(self.structure_projection,self.action_readout):
            layer.to(device=ref.device,dtype=ref.dtype)
        self._configuration=self._current_configuration()
        self._topology=tuple((n,type(m)) for n,m in self.named_modules(remove_duplicate=False))

    def _current_configuration(self):
        return dict(schema='Mora3.NativeForestTopology/v2',spec=asdict(self.spec),
            resource_policy='complete-reachable-grammar-attention-v1',
            field=self.field.get_extra_state(),features=_FEATURES,actions=('CLOSE','ADD_CHILD'),
            input='available-task-context-and-generated-DFS-prefix-only',
            output='normalized-masked-action-law-on-bounded-ordered-forests',
            root_contract='given-ordered-condition-roots',generated_evidence='computational',
            unlabelled_graph_quotient=False,reward_proportionality_verified=False,training_admission=False)

    def get_extra_state(self):
        self.spec.__post_init__()
        if (self.field is not self._field_reference[0]
                or not _linear(self.structure_projection,len(_FEATURES),self.field.width)
                or not _linear(self.action_readout,self.field.width,2)
                or self._topology != tuple((n,type(m))for n,m in self.named_modules(remove_duplicate=False))
                or not exact_wire(self._current_configuration(),self._configuration)):
            raise ValueError('Native topology source/operator/shape/configuration drift')
        ref=self.field.readout.weight
        if any(p.device!=ref.device or p.dtype!=ref.dtype for p in self.parameters()):
            raise ValueError('Exact shared native topology dtype/device required')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):
            raise ValueError('Topology checkpoint configuration mismatch')

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if strict is not True or assign is not False:
            raise ValueError('Strict non-assignment topology recovery required')
        return super().load_state_dict(state_dict,strict=True,assign=False)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        self.get_extra_state();expected=self.state_dict()
        if (local_metadata.get('assign_to_params_buffers',False)is not False
                or {k for k in state_dict if k.startswith(prefix)}!={prefix+k for k in expected}):
            raise ValueError('Complete non-assignment native topology checkpoint required')
        for name,live in expected.items():
            value=state_dict[prefix+name]
            if isinstance(live,torch.Tensor):
                if (type(value)is not torch.Tensor or value.is_meta or value.shape!=live.shape
                        or value.dtype!=live.dtype or value.layout!=live.layout
                        or value.is_floating_point()and not torch.isfinite(value).all()):
                    raise ValueError('Finite exact topology checkpoint tensors required')
            elif not exact_wire(value,live):
                raise ValueError('Topology checkpoint source/operator configuration mismatch')
        aliases={}
        for name,p in self.named_parameters(remove_duplicate=False):
            value=state_dict[prefix+name]
            if id(p)in aliases and not torch.equal(value,aliases[id(p)]):
                raise ValueError('Topology checkpoint tied alias mismatch')
            aliases[id(p)]=value
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def preflight(self,record,task,*,root_count,actions):
        self.get_extra_state();state=prefix_state(self.spec.bounds,root_count,actions)
        if state.complete:
            raise ValueError('A completed topology has no next action')
        raw=_condition_wire(record,task,self.field.spec.source_snapshot_sha256)
        g=self.field.geometry;spec=self.field.spec
        heads=g._source_backbone_reference[0].configuration()['heads']
        supported_nodes=maximum_forest_nodes(self.spec.bounds,root_count)
        supported_tokens=1+2*supported_nodes-root_count
        if (supported_nodes>g.config['id_width']
                or heads*supported_tokens*supported_tokens>spec.max_attention_elements):
            raise ValueError('Complete reachable topology support exceeds native resource capacity')
        n=len(state.parents);tokens=1+2*n-root_count
        if (len(raw)>spec.max_condition_bytes or len(record.nodes)>g.config['id_width']
                or n>g.config['id_width'] or heads*tokens*tokens>spec.max_attention_elements
                or heads*(1+len(record.nodes)+len(record.edges))**2>spec.max_attention_elements):
            raise ValueError('Complete topology/context exceeds native resource capacity')
        return state

    def forward(self,record,task,*,root_count,actions):
        state=self.preflight(record,task,root_count=root_count,actions=actions)
        ref=self.action_readout.weight;n=len(state.parents);limit=self.spec.bounds.max_nodes
        closed=set(state.closed);stack=set(state.stack);current=state.stack[-1]
        rows=[(state.depths[i]/limit,state.children[i]/limit,float(i in closed),
               float(i in stack),float(i==current),float(i<root_count),state.completed_roots/limit)
              for i in range(n)]
        features=F.silu((self.structure_projection(ref.new_tensor(rows))
                         +self.field.encode_context(record,task))/math.sqrt(2.))
        children=[i for i,p in enumerate(state.parents)if p>=0]
        parents=[state.parents[i]for i in children]
        edges=(features[parents]+features[children])/2.
        graph=dict(features=torch.cat((features,edges),0),
            endpoints=torch.tensor([(i,i)for i in range(n)]+list(zip(parents,children)),device=ref.device,dtype=torch.long),
            roles=torch.tensor([0]*n+[1]*len(children),device=ref.device,dtype=torch.long),
            identifiers=torch.eye(n,self.field.geometry.config['id_width'],dtype=ref.dtype,device=ref.device))
        hidden=self.field.geometry.graph([graph])[0][0,1+current]
        logits=self.action_readout(F.silu(hidden)*(self.field.readout_mult/(self.field.width/self.field.base_width)))
        if not torch.isfinite(logits).all():
            raise ValueError('Finite native topology logits required')
        mask=torch.tensor([i in state.allowed_actions for i in(CLOSE,ADD_CHILD)],device=ref.device)
        return torch.where(mask,logits,torch.full_like(logits,-torch.inf))


def topology_log_probability(head,record,task,parents,*,root_count):
    """Exact teacher-forced probability of the canonical ordered topology."""
    if type(head)is not NativeForestTopologyHead:
        raise ValueError('Exact native topology policy required')
    head.get_extra_state();source=canonical_forest_actions(head.spec.bounds,parents,root_count=root_count)
    if len(source.actions)>head.spec.max_total_decisions:
        raise ValueError('Complete topology trace exceeds declared decision budget')
    # Validate every structural/context prefix before the first neural call.
    for i in range(len(source.actions)):
        head.preflight(record,task,root_count=root_count,actions=source.actions[:i])
    terms=[]
    for i,action in enumerate(source.actions):
        logits=head(record,task,root_count=root_count,actions=source.actions[:i])
        terms.append(F.log_softmax(logits,dim=-1)[action])
    return dict(log_probability=torch.stack(terms).sum(),decisions=len(terms),
        canonical_parents=source.parents,canonical_to_original=source.canonical_to_original,
        actions=source.actions,training_admission=False)


def forest_topology_loss(head,record,task,forests,*,root_count,split):
    """Mean complete-trace NLL, not an average that reweights by trajectory length."""
    if type(head)is not NativeForestTopologyHead or split!='train':
        raise ValueError('Exact native head and original train-only topology supervision required')
    head.get_extra_state()
    if type(forests)is not tuple or not 1<=len(forests)<=head.spec.max_batch_size:
        raise ValueError('Explicit bounded complete source-forest batch required')
    sources=tuple(canonical_forest_actions(head.spec.bounds,p,root_count=root_count)for p in forests)
    if sum(len(x.actions)for x in sources)>head.spec.max_total_decisions:
        raise ValueError('Complete teacher batch exceeds declared decision budget')
    for source in sources:
        for i in range(len(source.actions)):
            head.preflight(record,task,root_count=root_count,actions=source.actions[:i])
    results=[topology_log_probability(head,record,task,p,root_count=root_count)for p in forests]
    loss=-torch.stack([x['log_probability']for x in results]).mean()
    if not torch.isfinite(loss):raise ValueError('Finite complete topology objective required')
    return dict(loss=loss,source_forests=len(forests),decisions=sum(x['decisions']for x in results),
        construction_policy_sha256=head.spec.construction_policy_sha256,training_admission=False)


def topology_policy_sha256(head):
    head.get_extra_state();versions=_parameter_versions(head);h=sha256(_wire(head.get_extra_state()))
    for name,p in head.named_parameters():
        h.update(_wire([name,list(p.shape),str(p.dtype)]))
        h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    if versions!=_parameter_versions(head):raise ValueError('Topology policy changed during binding')
    return h.hexdigest()


@torch.no_grad()
def sample_forest_topologies(head,record,task,*,root_count,count,generator):
    if type(head)is not NativeForestTopologyHead:
        raise ValueError('Exact native topology head required')
    head.preflight(record,task,root_count=root_count,actions=())
    ref=head.action_readout.weight
    if (type(count)is not int or not 1<=count<=head.spec.max_batch_size
            or type(generator)is not torch.Generator or torch.device(generator.device)!=ref.device
            or any(m.training for m in head.modules())
            or count*(2*head.spec.bounds.max_nodes-root_count)>head.spec.max_total_decisions):
        raise ValueError('Bounded evaluation-mode sampling and explicit same-device generator required')
    pin=topology_policy_sha256(head);versions=_parameter_versions(head)
    context=forest_context_sha256(record,task,source_snapshot_sha256=head.field.spec.source_snapshot_sha256)
    before=generator.get_state().clone();samples=[]
    try:
        for _ in range(count):
            actions=();terms=[]
            while not prefix_state(head.spec.bounds,root_count,actions).complete:
                logp=F.log_softmax(head(record,task,root_count=root_count,actions=actions),dim=-1)
                selected=int(torch.multinomial(logp.exp(),1,generator=generator))
                terms.append(logp[selected]);actions+=(selected,)
            state=prefix_state(head.spec.bounds,root_count,actions)
            samples.append(dict(actions=list(actions),parents=list(state.parents),root_count=root_count,
                log_probability=float(torch.stack(terms).sum()),policy_sha256=pin,context_sha256=context,
                source_snapshot_sha256=head.field.spec.source_snapshot_sha256,
                coordinate_sha256=head.field.spec.coordinate_sha256,
                construction_policy_sha256=head.spec.construction_policy_sha256))
        if versions!=_parameter_versions(head)or pin!=topology_policy_sha256(head):
            raise ValueError('Topology policy changed while sampling')
        if context!=forest_context_sha256(record,task,source_snapshot_sha256=head.field.spec.source_snapshot_sha256):
            raise ValueError('Topology conditioning changed while sampling')
    except BaseException:
        generator.set_state(before)
        raise
    return dict(samples=tuple(samples),generator_before=before,generator_after=generator.get_state().clone(),
        policy_sha256=pin,context_sha256=context,on_policy_for_bound_topology_policy=True,
        gfn_reward_teacher=False,reward_proportionality_verified=False,training_admission=False)


def sampled_topology_forest(head,record,task,sample,*,root_ids,root_states):
    """Bind a sampled topology to known roots; every free latent starts unknown/zero."""
    if type(head)is not NativeForestTopologyHead or type(sample)is not dict:
        raise ValueError('Exact native sampled topology required')
    expected={'actions','parents','root_count','log_probability','policy_sha256','context_sha256',
        'source_snapshot_sha256','coordinate_sha256','construction_policy_sha256'}
    if (set(sample)!=expected or type(sample['actions'])is not list or type(sample['parents'])is not list
            or type(sample['log_probability'])is not float or not math.isfinite(sample['log_probability'])
            or sample['log_probability']>1e-6):
        raise ValueError('Exact complete sampled-topology wire required')
    root_count=sample['root_count'];state=prefix_state(head.spec.bounds,root_count,tuple(sample['actions']))
    if not state.complete or list(state.parents)!=sample['parents']:
        raise ValueError('Complete sampled construction trace and parents must agree')
    context=forest_context_sha256(record,task,source_snapshot_sha256=head.field.spec.source_snapshot_sha256)
    for key,value in dict(policy_sha256=topology_policy_sha256(head),context_sha256=context,
        source_snapshot_sha256=head.field.spec.source_snapshot_sha256,coordinate_sha256=head.field.spec.coordinate_sha256,
        construction_policy_sha256=head.spec.construction_policy_sha256).items():
        if sample[key]!=value:raise ValueError('Sampled topology belongs to another source/context/policy snapshot')
    ref=head.action_readout.weight
    if (type(root_ids)is not tuple or len(root_ids)!=root_count or len(set(root_ids))!=root_count
            or any(type(x)is not str or not x or len(x)>4096 for x in root_ids)
            or type(root_states)is not torch.Tensor or root_states.requires_grad
            or root_states.shape!=(root_count,head.field.spec.dimension)
            or root_states.dtype!=ref.dtype or root_states.device!=ref.device or not torch.isfinite(root_states).all()):
        raise ValueError('Exact detached original root identities and coordinate values required')
    n=len(state.parents);values=ref.new_zeros((n,head.field.spec.dimension));values[:root_count]=root_states
    ids=root_ids+tuple('mora-generated-node:'+str(i)for i in range(root_count,n))
    statuses=tuple('open'if state.children[i]else'terminal'for i in range(n))
    return ForestState(ids,state.parents,statuses,values,tuple(range(root_count)),
        ('root',)*root_count+('computational',)*(n-root_count),context,head.field.spec.coordinate_sha256,'generated')


def native_forest_topology_objective(owner,*args,**kwargs):
    from .full_owner import SourceBoundOwner
    if type(owner)is not SourceBoundOwner:raise ValueError('Exact native owner required')
    owner.require_training_admission()
    if type(getattr(owner,'forest_topology',None))is not NativeForestTopologyHead:
        raise ValueError('Factory-registered native topology head required')
    return forest_topology_loss(owner.forest_topology,*args,**kwargs)


def forest_topology_parameter_roles(model,*,identifier_axis):
    from .parameterization import Axis,ParameterRole,parameter_ledger
    from .mu_optimizer import _storage_signature
    if type(model)is not NativeForestTopologyHead:raise ValueError('Exact native topology owner required')
    model.get_extra_state()
    roles={'field.'+n:r for n,r in forest_flow_parameter_roles(model.field,identifier_axis=identifier_axis).items()}
    hidden=Axis('width','hidden');features=Axis('finite',sha256(_wire(['forest-topology-features-v1',_FEATURES])).hexdigest())
    actions=Axis('finite',sha256(_wire(['forest-topology-actions-v1',asdict(model.spec)])).hexdigest())
    roles.update({'structure_projection.weight':ParameterRole('input',(hidden,features)),
        'structure_projection.bias':ParameterRole('vector',(hidden,)),
        'action_readout.weight':ParameterRole('output',(actions,hidden)),
        'action_readout.bias':ParameterRole('vector',(actions,))})
    _storage_signature(model);parameter_ledger(model,roles)
    return roles
