"""Learned full-forest and node criterion scores with source-bound masked labels.

These predictions rank candidates; they are not hard constraint decisions or
calibrated scientific evidence. Actual criterion meanings, admissible labels,
context and qualification belong to an immutable source label-policy artifact.
"""
from dataclasses import dataclass,asdict
from hashlib import sha256
import copy,math
import torch
from torch import nn
from torch.nn import functional as F
from .forest_flow import NativeForestFlowHead,_sha,_wire,_linear,forest_flow_parameter_roles
from .mixed_payload_projection import exact_wire


@dataclass(frozen=True)
class ForestQualitySpec:
    criteria: tuple[str,...]
    label_policy_sha256: str
    ranking_weights: tuple[float,...]

    def __post_init__(self):
        _sha(self.label_policy_sha256)
        if(type(self.criteria)is not tuple or not 1<=len(self.criteria)<=32
            or len(set(self.criteria))!=len(self.criteria)
            or any(type(x)is not str or not 1<=len(x)<=256 for x in self.criteria)
            or type(self.ranking_weights)is not tuple or len(self.ranking_weights)!=len(self.criteria)
            or any(type(x)not in(int,float)or not math.isfinite(x)or x<0 for x in self.ranking_weights)
            or not 0<sum(self.ranking_weights)<float('inf')):
            raise ValueError('Explicit finite criterion axis and nonnegative ranking weights required')


@dataclass(frozen=True)
class ForestQualityLabels:
    whole_values: torch.Tensor
    node_values: torch.Tensor
    whole_available: torch.Tensor
    node_available: torch.Tensor
    source_snapshot_sha256: str
    label_policy_sha256: str
    evidence_sha256: str
    split: str

    def __post_init__(self):
        for x in(self.source_snapshot_sha256,self.label_policy_sha256,self.evidence_sha256):_sha(x)
        if self.split not in('train','validation','test'):raise ValueError('Explicit original label split required')


class NativeForestQualityHead(nn.Module):
    def __init__(self,field,spec):
        super().__init__()
        if type(field)is not NativeForestFlowHead or type(spec)is not ForestQualitySpec:
            raise ValueError('Exact native field and criterion specification required')
        field.get_extra_state();spec.__post_init__()
        self.field=field;self._field_reference=(field,);self.spec=spec
        self.node_readout=nn.Linear(field.width,len(spec.criteria))
        self.forest_readout=nn.Linear(field.width,len(spec.criteria))
        with torch.no_grad():
            for layer in(self.node_readout,self.forest_readout):
                layer.weight.mul_(math.sqrt(field.width/field.base_width))
                layer.bias.mul_(math.sqrt(field.width/field.base_width))
        for layer in(self.node_readout,self.forest_readout):
            layer.to(dtype=field.readout.weight.dtype,device=field.readout.weight.device)
        self._configuration=self._current_configuration()
        self._topology=tuple((name,type(module))for name,module in self.named_modules(remove_duplicate=False))

    def _current_configuration(self):
        return dict(schema='Mora3.NativeForestQuality/v1',spec=asdict(self.spec),field=self.field.get_extra_state(),
            representation='all-node-complete-forest-tokens-and-once-per-node-mean',
            labels='criterion-specific-support-in-unit-interval-with-explicit-unknown-mask',
            ranking='declared-weighted-mean-not-hard-constraint-admission',calibrated=False,training_admission=False)

    def get_extra_state(self):
        self.spec.__post_init__()
        if(self.field is not self._field_reference[0]
            or any(not _linear(layer,self.field.width,len(self.spec.criteria))for layer in(self.node_readout,self.forest_readout))
            or self._topology!=tuple((n,type(m))for n,m in self.named_modules(remove_duplicate=False))
            or not exact_wire(self._configuration,self._current_configuration())):
            raise ValueError('Quality head source/axis/shared-operator drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('Quality checkpoint specification mismatch')

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if strict is not True or assign is not False:raise ValueError('Strict original-identity quality recovery required')
        return super().load_state_dict(state_dict,strict=True,assign=False)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        self.get_extra_state();expected=self.state_dict()
        if(local_metadata.get('assign_to_params_buffers',False)is not False
            or {k for k in state_dict if k.startswith(prefix)}!={prefix+k for k in expected}):
            raise ValueError('Exact quality checkpoint keys required')
        for name,live in expected.items():
            value=state_dict[prefix+name]
            if isinstance(live,torch.Tensor):
                if(not isinstance(value,torch.Tensor)or value.is_meta or value.shape!=live.shape
                    or value.dtype!=live.dtype or value.layout!=live.layout):
                    raise ValueError('Quality checkpoint tensor shape/dtype/layout mismatch')
            elif not exact_wire(value,live):raise ValueError('Quality checkpoint source/policy mismatch')
        aliases={}
        for name,p in self.named_parameters(remove_duplicate=False):
            value=state_dict[prefix+name]
            if id(p)in aliases and not torch.equal(value,aliases[id(p)]):raise ValueError('Quality tied checkpoint values differ')
            aliases[id(p)]=value
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def forward(self,record,task,state,template):
        self.get_extra_state()
        if not isinstance(state,torch.Tensor)or state.ndim!=3:raise ValueError('Complete forest state tensor required')
        hidden=self.field.encode_forest(record,task,state,state.new_ones(len(state)),template)
        scale=self.field.readout_mult/(self.field.width/self.field.base_width)
        node=self.node_readout(F.silu(hidden)*scale)
        whole=self.forest_readout(F.silu(hidden.mean(1))*scale)
        if not torch.isfinite(node).all()or not torch.isfinite(whole).all():raise ValueError('Nonfinite learned quality score')
        total=sum(self.spec.ranking_weights)
        weights=whole.new_tensor(tuple(w/total for w in self.spec.ranking_weights))
        return dict(criterion_logits=whole,node_logits=node,criteria=self.spec.criteria,
            ranking_score=(whole.sigmoid()*weights).sum(-1),node_ranking_score=(node.sigmoid()*weights).sum(-1),
            calibrated=False,scientific_validity_verified=False,training_admission=False)


    def bind_context(self,record,task,certify,*,constraint_policy_sha256,max_assessments=1024):
        return BoundForestQuality(self,record,task,certify,constraint_policy_sha256,max_assessments)


def forest_quality_loss(head,record,task,state,template,labels):
    if type(head)is not NativeForestQualityHead or type(labels)is not ForestQualityLabels:
        raise ValueError('Exact native learned quality head and source label contract required')
    head.get_extra_state();labels.__post_init__()
    if(labels.split!='train'or labels.source_snapshot_sha256!=head.field.spec.source_snapshot_sha256
        or labels.label_policy_sha256!=head.spec.label_policy_sha256):
        raise ValueError('Train-only same-source exact quality-label policy required')
    if not isinstance(state,torch.Tensor)or state.ndim!=3:raise ValueError('Complete forest state tensor required')
    head.field._validate(record,task,state,state.new_ones(len(state)),template)
    q=len(head.spec.criteria);b,n=state.shape[:2]
    for target,mask,shape in((labels.whole_values,labels.whole_available,(b,q)),(labels.node_values,labels.node_available,(b,n,q))):
        if(not isinstance(target,torch.Tensor)or target.shape!=shape or target.dtype!=state.dtype or target.device!=state.device
            or not isinstance(mask,torch.Tensor)or mask.shape!=shape or mask.dtype!=torch.bool or mask.device!=state.device):
            raise ValueError('Exact criterion target axes, dtype and availability required')
        known=target.detach()[mask]
        if not torch.isfinite(known).all()or((known<0)|(known>1)).any():
            raise ValueError('Known quality labels must be finite values in [0,1]')
    if labels.node_available[:,list(template.clamped)].any():raise ValueError('Clamped conditions cannot be scored as generated targets')
    count=int(labels.whole_available.sum()+labels.node_available.sum())
    if not count:return dict(loss=None,supervised_labels=0,training_admission=False)
    output=head(record,task,state,template);parts=[]
    for logits,target,mask in((output['criterion_logits'],labels.whole_values,labels.whole_available),
                              (output['node_logits'],labels.node_values,labels.node_available)):
        if mask.any():parts.append(F.binary_cross_entropy_with_logits(logits[mask],target.detach()[mask],reduction='sum'))
    return dict(loss=sum(parts)/count,supervised_labels=count,criterion_names=head.spec.criteria,
        calibration_verified=False,training_admission=False)


def forest_quality_parameter_roles(model,*,identifier_axis):
    from .parameterization import Axis,ParameterRole,parameter_ledger
    from .mu_optimizer import _storage_signature
    if type(model)is not NativeForestQualityHead:raise ValueError('Exact native quality owner required')
    model.get_extra_state()
    roles={'field.'+n:r for n,r in forest_flow_parameter_roles(model.field,identifier_axis=identifier_axis).items()}
    hidden=Axis('width','hidden');axis=Axis('finite',sha256(_wire(['forest-quality-axis-v1',asdict(model.spec)])).hexdigest())
    for name in('node_readout','forest_readout'):
        roles[name+'.weight']=ParameterRole('output',(axis,hidden));roles[name+'.bias']=ParameterRole('vector',(axis,))
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def native_forest_quality_objective(owner,*args,**kwargs):
    from .full_owner import SourceBoundOwner
    if type(owner)is not SourceBoundOwner:raise ValueError('Exact registered full source owner required')
    owner.require_training_admission()
    if type(getattr(owner,'forest_quality',None))is not NativeForestQualityHead:
        raise ValueError('Factory-registered complete-forest quality head required')
    return forest_quality_loss(owner.forest_quality,*args,**kwargs)


@dataclass(frozen=True)
class ForestConstraintDecision:
    candidate_sha256: str
    context_sha256: str
    policy_sha256: str
    status: str
    evidence_sha256: str
    repairable: bool=False

    def __post_init__(self):
        for pin in(self.candidate_sha256,self.context_sha256,self.policy_sha256,self.evidence_sha256):_sha(pin)
        if self.status not in('pass','refused','unknown')or type(self.repairable)is not bool:
            raise ValueError('Explicit independent constraint verdict required')


class BoundForestQuality:
    """Native learned priority plus a separately bound hard-constraint decision."""
    def __init__(self,head,record,task,certify,constraint_policy_sha256,max_assessments):
        from .forest_flow import _parameter_versions,forest_context_sha256
        from .forest_token_decoder import _tensor_sha
        if type(head)is not NativeForestQualityHead or not callable(certify):
            raise ValueError('Exact native scorer and explicit independent verifier required')
        _sha(constraint_policy_sha256)
        if type(max_assessments)is not int or not 1<=max_assessments<=65536:raise ValueError('Bounded evidence-record budget required')
        self.head=head;self.record=record;self.task=task;self.certify=certify
        self.constraint_policy_sha256=constraint_policy_sha256;self.max_assessments=max_assessments
        self.context_sha256=forest_context_sha256(record,task,source_snapshot_sha256=head.field.spec.source_snapshot_sha256)
        self._configuration=_wire(head.get_extra_state());self._versions=_parameter_versions(head);self._evidence={}
        h=sha256(self._configuration)
        for name,value in head.named_parameters():h.update(_wire([name,_tensor_sha(value)]))
        self.sha256=sha256(_wire(dict(schema='Mora3.BoundForestQuality/v1',model=h.hexdigest(),
            context=self.context_sha256,constraints=constraint_policy_sha256,max_assessments=max_assessments))).hexdigest()

    @torch.no_grad()
    def __call__(self,forest):
        from .forest_flow import _parameter_versions,forest_context_sha256
        from mora3.reasoning.flow_forest_refinement import forest_fingerprint,ForestAssessment
        pin=forest_fingerprint(forest)
        if(self._versions!=_parameter_versions(self.head)or self._configuration!=_wire(self.head.get_extra_state())
            or self.context_sha256!=forest_context_sha256(self.record,self.task,source_snapshot_sha256=self.head.field.spec.source_snapshot_sha256)
            or forest.context_sha256!=self.context_sha256):
            raise ValueError('Bound quality model or context changed')
        if len(self._evidence)>=self.max_assessments:raise ValueError('Complete assessment-evidence budget exhausted')
        candidate=forest.copy();decision=self.certify(candidate)
        if(type(decision)is not ForestConstraintDecision or forest_fingerprint(candidate)!=pin):
            raise ValueError('Unmodified candidate and exact independent constraint decision required')
        decision.__post_init__()
        if decision.candidate_sha256!=pin or decision.context_sha256!=self.context_sha256 or decision.policy_sha256!=self.constraint_policy_sha256:
            raise ValueError('Constraint decision does not bind the current forest/context/policy')
        out=self.head(self.record,self.task,forest.states[None],forest)
        values=out['criterion_logits'][0].sigmoid().tolist();node_values=out['node_logits'][0].sigmoid().tolist()
        evidence=dict(schema='Mora3.NativeForestQualityEvidence/v1',candidate_sha256=pin,
            evaluator_sha256=self.sha256,context_sha256=self.context_sha256,
            constraint_decision=asdict(decision),criteria=self.head.spec.criteria,criterion_scores=values,
            node_criterion_scores=node_values,calibrated=False,training_admission=False)
        digest=sha256(_wire(evidence)).hexdigest();self._evidence[digest]=evidence
        return ForestAssessment(pin,self.context_sha256,self.sha256,decision.status,
            float(out['ranking_score'][0]),tuple(out['node_ranking_score'][0].tolist()),digest,decision.repairable)

    def evidence(self,digest):
        _sha(digest)
        if digest not in self._evidence:raise ValueError('Unknown assessment evidence digest')
        return copy.deepcopy(self._evidence[digest])
