"""Learned, source-bound joint denoising of complete latent reasoning forests.

Conditional Gaussian-path FM follows Lipman et al.2210.02747v2. Forest graph
features, native context sharing and the controller binding are Mora-3 adaptations.
No biological validity, trained quality, corpus admission or tropical muP theorem
is implied. All noisy branches are legitimate joint-denoising inputs; this is NOT
an autoregressive clean-prefix policy and is not used to assign GFN densities.
"""
from dataclasses import asdict,dataclass
from hashlib import sha256
import copy
import json
import math
import re
import torch
from torch import nn
from torch.nn import functional as F
from mora3.data.schema import GraphRecord
from mora3.data.tasks import ConditionalTask
from mora3.reasoning.flow_forest_refinement import ForestState
from .geometry_conditioning import GeometryConditioner
from .flow_matching import velocity_loss
from .mixed_payload_projection import exact_wire


_STATUSES=('open','terminal','censored','invalid')
_DIRECTIONS=('root','computational','temporal','observational','interventional')
_STRUCTURE_FIELDS=('depth/max_nodes','clamped')+tuple('status:'+x for x in _STATUSES)+tuple('direction:'+x for x in _DIRECTIONS)


def _wire(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def _sha(value):
    if type(value)is not str or re.fullmatch('[0-9a-f]{64}',value)is None:
        raise ValueError('Exact source/coordinate SHA256 required')


@dataclass(frozen=True)
class ForestFlowSpec:
    dimension: int
    source_snapshot_sha256: str
    coordinate_sha256: str
    max_nodes: int=128
    max_batch_size: int=32
    max_attention_elements: int=8_000_000
    max_condition_bytes: int=2_000_000
    sigma_min: float=.01

    def __post_init__(self):
        for n,limit in((self.dimension,256),(self.max_nodes,512),(self.max_batch_size,256),
                      (self.max_attention_elements,2**31),(self.max_condition_bytes,64*1024**2)):
            if type(n)is not int or not 1<=n<=limit:
                raise ValueError('Exact positive bounded forest dimensions/resources required')
        for s in(self.source_snapshot_sha256,self.coordinate_sha256):_sha(s)
        if type(self.sigma_min)not in(int,float)or not math.isfinite(self.sigma_min)or not 0<=self.sigma_min<1:
            raise ValueError('Gaussian path sigma_min must be finite in [0,1)')


def _condition_wire(record,task,source_snapshot_sha256):
    _sha(source_snapshot_sha256)
    if type(record)is not GraphRecord or type(task)is not ConditionalTask:
        raise ValueError('Exact native graph and condition/target contract required')
    chosen=task.input_fields(record);roles=dict(task.roles)
    entities={}
    for name,values in(('node',record.nodes),('edge',record.edges)):
        rows=[]
        for i,value in enumerate(values):
            row=dict(kind=value.kind,measurements=[dict(**asdict(m),role=roles[(name,i,m.name)])
                for m in sorted(value.measurements,key=lambda m:m.name)if(name,i,m.name)in chosen])
            if name=='edge':row.update(source=value.source,target=value.target,directed=value.directed)
            rows.append(row)
        entities[name]=rows
    # Record IDs, held-out values, target availability, target labels and source
    # metadata not encoded by the biological context cannot become conditioning.
    return _wire(dict(schema='Mora3.ForestFlowCondition/v1',source_snapshot_sha256=source_snapshot_sha256,
        topology_known=task.topology_known,context=task.context,entities=entities))


def forest_context_sha256(record,task,*,source_snapshot_sha256):
    return sha256(_condition_wire(record,task,source_snapshot_sha256)).hexdigest()


def _linear(layer,inp,out):
    return(type(layer)is nn.Linear and not layer._modules and not layer._buffers
        and set(layer._parameters)=={'weight','bias'}and layer.in_features==inp and layer.out_features==out
        and tuple(layer.weight.shape)==(out,inp)and layer.bias is not None and tuple(layer.bias.shape)==(out,))


class NativeForestFlowHead(nn.Module):
    """One biological problem, B complete candidate/target forests per call.

    Continuous state -> finite-to-hidden projection -> shared native graph layers
    -> per-node hidden-to-finite velocity. Both context and forest invocations
    reuse the EXACT native transformer. All node and directed edge tokens remain.
    A mixed-problem training schedule can accumulate explicit per-problem losses.
    """
    def __init__(self,geometry,spec,*,readout_mult=1.):
        super().__init__()
        if type(geometry)is not GeometryConditioner or type(spec)is not ForestFlowSpec:
            raise ValueError('Exact registered native geometry owner and forest specification required')
        spec.__post_init__();geometry.get_extra_state()
        source=geometry._source_backbone_reference[0];native=source.configuration()
        width=geometry.config['width'];rank=geometry.config['id_width']
        if spec.max_nodes>rank:raise ValueError('Forest maximum exceeds exact identifier rank')
        if geometry.config['feature_width']!=width or native['width']!=width:
            raise ValueError('Native forest feature axis must match shared hidden width')
        if type(readout_mult)not in(int,float)or not math.isfinite(readout_mult)or readout_mult<=0:
            raise ValueError('Positive finite readout multiplier required')
        if any(getattr(geometry.graph,n)is not getattr(source,n)for n in('feature','role','graph_token','layers','final_norm')):
            raise ValueError('Exact native graph transformer sharing required')
        applied=geometry.parameterization_policy=='mup-geometry-v1'
        if applied and(native['schema']!='tokengt-native-configuration-v3'or native['attention']!='vanilla'):
            raise ValueError('Current native applied muP family does not qualify tropical attention')
        self.geometry=geometry;self.spec=spec;self.width=width
        self.base_width=native['parameterization']['base_width']if applied else width
        self.readout_mult=float(readout_mult)
        self.parameterization='mup-forest-flow-v1'if applied else'legacy-no-mup-qualification'
        self._shared_reference=(geometry,source)
        self.state_projection=nn.Linear(spec.dimension,width)
        self.time_projection=nn.Linear(1,width)
        self.structure_projection=nn.Linear(len(_STRUCTURE_FIELDS),width)
        self.readout=nn.Linear(width,spec.dimension)
        with torch.no_grad():
            scale=math.sqrt(width/self.base_width)
            self.readout.weight.mul_(scale);self.readout.bias.mul_(scale)
        ref=geometry.graph.feature.weight
        for layer in(self.state_projection,self.time_projection,self.structure_projection,self.readout):
            layer.to(device=ref.device,dtype=ref.dtype)
        self._configuration=self._current_configuration()
        self._topology=tuple((n,type(m))for n,m in self.named_modules(remove_duplicate=False))

    def _current_configuration(self):
        source=self.geometry._source_backbone_reference[0]
        return dict(schema='Mora3.NativeCompleteForestFlow/v1',spec=asdict(self.spec),
            width=self.width,base_width=self.base_width,readout_mult=self.readout_mult,
            parameterization=self.parameterization,geometry=self.geometry.get_extra_state(),
            shared_native_operator=source.configuration(),structural_features=_STRUCTURE_FIELDS,
            computation='silu((state+time+structure+biological_context)/2);node-and-directed-edge-TokenGT;per-node-MuReadout',
            edge_features='mean-of-parent-child-feature-v1',path='Gaussian-CFM-on-available-free-coordinates',
            unavailable_noisy_state='independent-noise-no-supervised-loss',
            context_batch='one-known-biological-context-many-full-forests',
            attention_semantics='joint-noisy-forest-not-clean-prefix-causal-policy',
            output='velocity-in-fixed-source-bound-latent-axis',training_admission=False)

    def get_extra_state(self):
        self.spec.__post_init__()
        g,source=self._shared_reference
        if(self.geometry is not g or g._source_backbone_reference[0]is not source
            or any(getattr(g.graph,n)is not getattr(source,n)for n in('feature','role','graph_token','layers','final_norm'))
            or not _linear(self.state_projection,self.spec.dimension,self.width)
            or not _linear(self.time_projection,1,self.width)
            or not _linear(self.structure_projection,len(_STRUCTURE_FIELDS),self.width)
            or not _linear(self.readout,self.width,self.spec.dimension)
            or self._topology!=tuple((n,type(m))for n,m in self.named_modules(remove_duplicate=False))
            or not exact_wire(self._configuration,self._current_configuration())):
            raise ValueError('Native forest operator/source/axis/parameter configuration drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('Forest checkpoint configuration mismatch')

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if strict is not True or assign is not False:raise ValueError('Exact forest recovery requires strict non-assignment load')
        return super().load_state_dict(state_dict,strict=True,assign=False)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        self.get_extra_state()
        expected=self.state_dict()
        if local_metadata.get('assign_to_params_buffers',False)is not False:
            raise ValueError('Forest recovery cannot replace shared parameter identities')
        if {k for k in state_dict if k.startswith(prefix)}!={prefix+k for k in expected}:
            raise ValueError('Complete exact forest checkpoint keys required')
        for k,live in expected.items():
            item=state_dict[prefix+k]
            if isinstance(live,torch.Tensor):
                if(not isinstance(item,torch.Tensor)or item.is_meta or item.shape!=live.shape
                    or item.dtype!=live.dtype or item.layout!=live.layout):
                    raise ValueError('Forest checkpoint tensor shape/dtype/layout mismatch')
            elif not exact_wire(item,live):raise ValueError('Forest checkpoint source/operator sidecar mismatch')
        aliases={}
        for name,p in self.named_parameters(remove_duplicate=False):
            item=state_dict[prefix+name]
            if id(p)in aliases and not torch.equal(item,aliases[id(p)]):
                raise ValueError('Forest checkpoint tied alias values differ')
            aliases[id(p)]=item
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _validate(self,record,task,state,time,template):
        self.get_extra_state();s=self.spec
        if type(template)is not ForestState:raise ValueError('Exact complete forest skeleton required')
        template.__post_init__()
        if(not template.is_complete or template.coordinate_sha256!=s.coordinate_sha256
            or template.states.shape[1]!=s.dimension or len(template.parents)>s.max_nodes):
            raise ValueError('Complete forest/source coordinate axis or node capacity mismatch')
        raw=_condition_wire(record,task,s.source_snapshot_sha256)
        if len(raw)>s.max_condition_bytes:raise ValueError('Complete known condition exceeds byte budget; no truncation')
        if template.context_sha256!=sha256(raw).hexdigest():raise ValueError('Forest context identity mismatch')
        ref=self.readout.weight;n=len(template.parents)
        if(not isinstance(state,torch.Tensor)or state.ndim!=3 or not 1<=len(state)<=s.max_batch_size
            or state.shape[1:]!=(n,s.dimension)or state.dtype!=ref.dtype or state.device!=ref.device
            or template.states.dtype!=ref.dtype or template.states.device!=ref.device
            or not isinstance(time,torch.Tensor)or time.shape!=(len(state),)
            or time.dtype!=ref.dtype or time.device!=ref.device
            or not torch.isfinite(state).all()or not torch.isfinite(time).all()or((time<0)|(time>1)).any()):
            raise ValueError('Finite complete noisy forests and matching batch times in [0,1] required')
        heads=self._shared_reference[1].configuration()['heads']
        forest_tokens=1+n+sum(p!=-1 for p in template.parents)
        context_tokens=1+len(record.nodes)+len(record.edges)
        if(len(record.nodes)>self.geometry.config['id_width']
            or len(state)*heads*forest_tokens**2>s.max_attention_elements
            or heads*context_tokens**2>s.max_attention_elements
            or state.numel()>16_777_216):
            raise ValueError('Complete context/forest exceeds declared identifier or attention capacity')
        return raw

    def encode_context(self,record,task):
        self.get_extra_state()
        raw=_condition_wire(record,task,self.spec.source_snapshot_sha256)
        heads=self._shared_reference[1].configuration()['heads']
        if (len(raw)>self.spec.max_condition_bytes or len(record.nodes)>self.geometry.config['id_width']
                or heads*(1+len(record.nodes)+len(record.edges))**2>self.spec.max_attention_elements):
            raise ValueError('Complete native context exceeds declared resource envelope')
        return self.geometry(record,task)

    def encode_forest(self,record,task,state,time,template):
        """Differentiable per-node features for the shared field and analogy head."""
        self._validate(record,task,state,time,template)
        b,n,d=state.shape;mask=torch.zeros(n,dtype=torch.bool,device=state.device)
        mask[list(template.clamped)]=True
        x=torch.where(mask[None,:,None],template.states.detach()[None],state)
        rows=[];depth=[]
        for i,parent in enumerate(template.parents):
            level=0 if parent==-1 else depth[parent]+1;depth.append(level)
            rows.append((level/self.spec.max_nodes,float(i in template.clamped))
                +tuple(float(template.statuses[i]==v)for v in _STATUSES)
                +tuple(float(template.directions[i]==v)for v in _DIRECTIONS))
        structure=state.new_tensor(rows)
        context=self.geometry(record,task)
        features=F.silu((self.state_projection(x)+self.time_projection(time[:,None])[:,None,:]
            +self.structure_projection(structure)[None]+context[:,None,:])/2.)
        child=[i for i,p in enumerate(template.parents)if p!=-1]
        parent=[template.parents[i]for i in child]
        edge=(features[:,parent]+features[:,child])/2.
        all_features=torch.cat((features,edge),1)
        endpoints=torch.tensor([(i,i)for i in range(n)]+list(zip(parent,child)),device=state.device,dtype=torch.long)
        roles=torch.tensor([0]*n+[1]*len(child),device=state.device,dtype=torch.long)
        ids=torch.eye(n,self.geometry.config['id_width'],device=state.device,dtype=state.dtype)
        graphs=[dict(features=row,endpoints=endpoints,roles=roles,identifiers=ids)for row in all_features]
        return self.geometry.graph(graphs)[0][:,1:n+1,:]

    def forward(self,record,task,state,time,template):
        hidden=self.encode_forest(record,task,state,time,template)
        mask=torch.zeros(state.shape[1],device=state.device,dtype=torch.bool)
        mask[list(template.clamped)]=True
        velocity=self.readout(F.silu(hidden)*(self.readout_mult/(self.width/self.base_width)))
        if not torch.isfinite(velocity).all():raise ValueError('Nonfinite native forest velocity')
        return torch.where(mask[None,:,None],torch.zeros_like(velocity),velocity)

    def bind_context(self,record,task):
        return BoundForestVelocity(self,record,task)


def _parameter_versions(model):
    return(tuple((n,id(p),p._version,str(p.dtype),str(p.device))for n,p in model.named_parameters()),
           tuple((n,m.training)for n,m in model.named_modules()))


class BoundForestVelocity:
    """One snapshot/condition-bound callback for the existing native FoT sampler.

    The digest binds actual weights, not just a configuration string. It is not
    a model-quality certificate. Each call recomputes the native condition; there
    is no stale differentiable-context cache across sampler/training invocations.
    """
    def __init__(self,head,record,task):
        if type(head)is not NativeForestFlowHead:raise ValueError('Exact native learned forest head required')
        self.head=head;self.record=record;self.task=task
        self.context_sha256=forest_context_sha256(record,task,source_snapshot_sha256=head.spec.source_snapshot_sha256)
        self._configuration=_wire(head.get_extra_state());self._versions=_parameter_versions(head)
        h=sha256(self._configuration)
        for name,p in head.named_parameters():
            h.update(_wire([name,list(p.shape),str(p.dtype)]))
            h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        self.sha256=sha256(_wire(dict(schema='Mora3.BoundForestVelocity/v1',weights_sha256=h.hexdigest(),
                                    context_sha256=self.context_sha256))).hexdigest()
        if self._versions!=_parameter_versions(head):raise ValueError('Model changed during field binding')

    def validate(self):
        if(self._versions!=_parameter_versions(self.head)or self._configuration!=_wire(self.head.get_extra_state())
            or self.context_sha256!=forest_context_sha256(self.record,self.task,source_snapshot_sha256=self.head.spec.source_snapshot_sha256)):
            raise ValueError('Bound native velocity model/context changed; bind a fresh snapshot')

    def __call__(self,state,time,template):
        self.validate()
        return self.head(self.record,self.task,state,time,template)


def forest_flow_matching_loss(head,record,task,template,target,noise,time,available):
    """Conditional FM on explicitly supervised non-condition coordinates.

    The target is detached and masked BEFORE arithmetic. Missing coordinates use
    independent supplied noise and carry no loss. This partial-label objective
    does not by itself establish a normalized model over unobserved coordinates.
    Noise/time are caller-owned so the training runner can checkpoint their RNG.
    """
    if type(head)is not NativeForestFlowHead:raise ValueError('Exact native forest head required')
    head._validate(record,task,noise,time,template)
    if(not isinstance(target,torch.Tensor)or target.shape!=noise.shape
        or target.dtype!=noise.dtype or target.device!=noise.device
        or not isinstance(available,torch.Tensor)or available.dtype!=torch.bool
        or available.shape!=noise.shape or available.device!=noise.device):
        raise ValueError('Exact complete target/noise/availability tensors required')
    if available[:,list(template.clamped)].any():raise ValueError('Known clamped conditions cannot be supervised as generated targets')
    count=int(available.sum())
    if count==0:return dict(loss=None,supervised_coordinates=0,training_admission=False)
    if not torch.isfinite(target.detach()[available]).all():raise ValueError('Available forest targets must be finite')
    y=torch.where(available,target.detach(),torch.zeros_like(target));rate=1-head.spec.sigma_min
    state=torch.where(available,(1-rate*time[:,None,None])*noise+time[:,None,None]*y,noise)
    clamp=torch.zeros(noise.shape[1],device=noise.device,dtype=torch.bool);clamp[list(template.clamped)]=True
    state=torch.where(clamp[None,:,None],template.states.detach()[None],state)
    prediction=head(record,task,state,time,template)
    loss=velocity_loss(prediction.reshape(len(noise),-1),(y-rate*noise).reshape(len(noise),-1),available.reshape(len(noise),-1))
    return dict(loss=loss,supervised_coordinates=count,training_admission=False)


def native_forest_flow_objective(owner,*args,**kwargs):
    """Production entry: never bypass the existing complete-owner release gate."""
    from .full_owner import SourceBoundOwner
    if type(owner)is not SourceBoundOwner:raise ValueError('Exact registered source owner required')
    owner.require_training_admission()
    if type(getattr(owner,'forest_flow',None))is not NativeForestFlowHead:
        raise ValueError('Actual native forest head must be factory-registered')
    return forest_flow_matching_loss(owner.forest_flow,*args,**kwargs)


def forest_flow_parameter_roles(model,*,identifier_axis):
    from .morphology_mup import _geometry_roles
    from .parameterization import Axis,ParameterRole,parameter_ledger
    from .mu_optimizer import _storage_signature
    if type(model)is not NativeForestFlowHead:raise ValueError('Exact native forest head required')
    model.get_extra_state()
    if model.parameterization!='mup-forest-flow-v1':raise ValueError('No muP qualification for legacy/tropical forest owner')
    roles={'geometry.'+n:r for n,r in _geometry_roles(model.geometry,identifier_axis=identifier_axis).items()}
    hidden=Axis('width','hidden');latent=Axis('finite',model.spec.coordinate_sha256)
    time_axis=Axis('finite',sha256(_wire(['forest-flow-time-v1',['t']])).hexdigest())
    structure=Axis('finite',sha256(_wire(['forest-flow-structure-v1',_STRUCTURE_FIELDS])).hexdigest())
    for name,axis in(('state_projection',latent),('time_projection',time_axis),('structure_projection',structure)):
        roles[name+'.weight']=ParameterRole('input',(hidden,axis))
        roles[name+'.bias']=ParameterRole('vector',(hidden,))
    roles['readout.weight']=ParameterRole('output',(latent,hidden))
    roles['readout.bias']=ParameterRole('vector',(latent,))
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


class NativeForestSubtreeEditor:
    """Score-directed regeneration with the real native conditional velocity field.

    Nonedited generated nodes are fixed proposal context, NOT newly observed
    facts. Every revision must pass the controller's fresh whole-forest checks.
    This is conditional inference refinement, not an on-policy GFN transition.
    """
    def __init__(self,field,*,steps,generator):
        if(type(field)is not BoundForestVelocity or type(steps)is not int or not 1<=steps<=4096
                or type(generator)is not torch.Generator):
            raise ValueError('Exact native field, bounded solver steps and explicit generator required')
        field.validate()
        if torch.device(generator.device)!=field.head.readout.weight.device:
            raise ValueError('Editor generator and native field device must match')
        self.field=field;self.steps=steps;self.generator=generator;self.last_receipt=None

    @torch.no_grad()
    def __call__(self,forest,assessment,*,count,reason,feedback):
        from mora3.reasoning.flow_forest_refinement import (
            ForestAssessment,ForestRevision,forest_fingerprint,replace_subtree,_descendants)
        from .flow_matching import euler_flow
        self.field.validate();pin=forest_fingerprint(forest)
        if(type(assessment)is not ForestAssessment or assessment.candidate_sha256!=pin
                or assessment.context_sha256!=forest.context_sha256
                or len(assessment.node_scores)!=len(forest.parents)
                or type(count)is not int or not 1<=count<=self.field.head.spec.max_batch_size
                or reason not in('correct','improve','post_decode')):
            raise ValueError('Current complete-forest assessment and bounded native edit request required')
        assessment.__post_init__()
        if not forest.is_complete:raise ValueError('Native regeneration requires a complete candidate skeleton')
        eligible=[]
        for i in range(len(forest.parents)):
            affected=_descendants(forest,forest.node_ids[i])
            if not affected.intersection(forest.clamped):eligible.append((assessment.node_scores[i],i,affected))
        if not eligible:
            self.last_receipt=dict(status='no-editable-subtree',velocity_evaluations=0)
            return ()
        _,cut,affected=min(eligible,key=lambda row:(row[0],row[1]))
        indices=tuple(sorted(affected));index={old:i for i,old in enumerate(indices)}
        parents=tuple(-1 if old==cut else index[forest.parents[old]]for old in indices)
        statuses=tuple(forest.statuses[i]for i in indices)
        n,d=forest.states.shape
        # Validate the real shape/resource/context BEFORE advancing caller RNG.
        probe=forest.states[None].expand(count,n,d)
        self.field.head._validate(self.field.record,self.field.task,probe,probe.new_zeros(count),forest)
        before=self.generator.get_state().clone()
        x=probe.clone();x[:,indices]=torch.randn((count,len(indices),d),
            generator=self.generator,dtype=x.dtype,device=x.device)
        mask=torch.zeros(n,dtype=torch.bool,device=x.device);mask[list(indices)]=True
        def velocity(flat,time):
            local=flat.reshape(count,n,d)
            local=torch.where(mask[None,:,None],local,forest.states[None])
            value=self.field(local,time,forest)
            return torch.where(mask[None,:,None],value,torch.zeros_like(value)).reshape(count,-1)
        values=euler_flow(velocity,x.reshape(count,-1),steps=self.steps).reshape(count,n,d)
        revisions=tuple(ForestRevision(replace_subtree(forest,forest.node_ids[cut],row[list(indices)],parents,statuses),
            'post_decode_correction'if reason=='post_decode'else'subtree_regeneration',forest.node_ids[cut])for row in values)
        self.last_receipt=dict(schema='Mora3.NativeForestSubtreeRegeneration/v1',status='proposed-unverified',
            parent_sha256=pin,field_sha256=self.field.sha256,cut_node_id=forest.node_ids[cut],
            affected_nodes=len(indices),proposals=count,velocity_evaluations=self.steps,
            candidate_equivalent_evaluations=count*self.steps,generator_before=before,
            generator_after=self.generator.get_state().clone(),generated_preserved_nodes_are_observations=False,
            improvement_verified=False,gfn_on_policy=False,training_admission=False)
        return revisions
