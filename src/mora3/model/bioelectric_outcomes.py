"""Shared native perturbation outcome reference heads; no admission or causal inference."""
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import re

import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.bioelectric_evidence import NativeBioelectricEndpoint,NativeBioelectricObservation
from mora3.data.provenance import Provenance
from mora3.data.schema import GraphRecord,Measurement,Node
from mora3.data.tasks import ConditionalTask
from .annotation_multihead import SharedAnnotationModel
from .mixed_payload_projection import exact_wire
from .source_scalar_objectives import native_scalar_likelihood


def _wire(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
def _sha(x):return hashlib.sha256(_wire(x)).hexdigest()


@dataclass(frozen=True)
class BioelectricAssayAxis:
    measurement: str
    assay: str
    literal_unit: str | None
    coordinate: str
    category_axis: tuple = ()

    @classmethod
    def from_endpoint(cls,endpoint):
        if type(endpoint) is not NativeBioelectricEndpoint:raise ValueError('exact native measurement endpoint required')
        endpoint.binding()
        return cls(endpoint.measurement,endpoint.native_assay,endpoint.literal_unit,endpoint.coordinate,endpoint.category_axis)

    def binding(self):
        return NativeBioelectricEndpoint(self.measurement,self.assay,self.literal_unit,self.coordinate,None,self.category_axis).binding()

    @property
    def key(self):return _sha(self.binding())

    @property
    def outputs(self):return ('mean','raw_scale') if self.coordinate=='native_scalar' else self.category_axis


def _graph(condition):
    record=GraphRecord('input',(Node('native-bioelectric-assay',(
        Measurement('known_query',_wire(condition).decode('utf8'),None,'observed',True),
        Measurement('requested_outcome',None,None,'missing',False))),),(),
        (Provenance('native-model-input','bioelectric-outcome-v1','input','0'*64,'source-admission-external'),))
    task=ConditionalTask(((("node",0,"known_query"),'condition'),(("node",0,"requested_outcome"),'target')),topology_known=True)
    return record,task


class SharedBioelectricOutcomeModel(nn.Module):
    """Separate native assay coordinates on the exact registered TokenGT owner."""
    def __init__(self,shared,axes,*,source_contract_manifest_sha256,max_batch_size=32):
        super().__init__()
        if (not isinstance(shared,SharedAnnotationModel) or type(axes) is not tuple or not axes
                or any(type(a) is not BioelectricAssayAxis for a in axes)
                or len({a.key for a in axes})!=len(axes)
                or type(source_contract_manifest_sha256) is not str or not re.fullmatch('[0-9a-f]{64}',source_contract_manifest_sha256)
                or type(max_batch_size) is not int or max_batch_size<1):
            raise ValueError('complete shared encoder, exact native assay axes and source/batch binding required')
        shared.validate_parameterization_configuration();c=shared.get_extra_state();p=shared.payload.embedding.weight
        self.shared=shared;self._shared_owner=(shared,);self.axes=axes;self.source_contract_manifest_sha256=source_contract_manifest_sha256
        self.width=c['width'];self.base_width=c['mup']['base_width'];self.readout_mult=c['mup']['output_mult'];self.max_batch_size=max_batch_size
        self.heads=nn.ModuleDict({a.key:nn.Linear(self.width,len(a.outputs)) for a in axes})
        with torch.no_grad():
            for head in self.heads.values():
                factor=math.sqrt(self.width/self.base_width);head.weight.mul_(factor);head.bias.mul_(factor)
        self.heads.to(device=p.device,dtype=p.dtype);self._configuration=self._local_state()

    def _local_state(self):
        return dict(schema='Mora3.SharedBioelectricOutcomeModel/v1',width=self.width,base_width=self.base_width,
            readout_mult=self.readout_mult,source_contract_manifest_sha256=self.source_contract_manifest_sha256,
            max_batch_size=self.max_batch_size,axes=tuple(a.binding()for a in self.axes),
            conditions='known baseline/ordered intervention/native assay query only; outcome, source IDs and audit excluded',
            loss='native Float64 Gaussian point/censor mass or literal finite categorical CE; missing excluded',training_admission=False)

    def get_extra_state(self):
        self.shared.validate_parameterization_configuration();c=self.shared.get_extra_state()
        if (self.shared is not self._shared_owner[0] or set(self._modules)!={'shared','heads'}
                or not exact_wire(self._local_state(),self._configuration) or tuple(self.heads)!=tuple(a.key for a in self.axes)
                or (self.width,self.base_width,self.readout_mult)!=(c['width'],c['mup']['base_width'],c['mup']['output_mult'])
                or any(type(self.heads[a.key]) is not nn.Linear or self.heads[a.key]._modules or self.heads[a.key]._buffers
                    or tuple(self.heads[a.key].weight.shape)!=(len(a.outputs),self.width)
                    or tuple(self.heads[a.key].bias.shape)!=(len(a.outputs),) for a in self.axes)):
            raise ValueError('native bioelectric assay/shared/readout configuration drift')
        return dict(**copy.deepcopy(self._configuration),shared=c)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('native bioelectric checkpoint source/assay mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        for name,value in self.state_dict().items():
            if not isinstance(value,torch.Tensor) and not exact_wire(value,state_dict.get(prefix+name)):
                raise ValueError('native bioelectric checkpoint source/assay mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _readout(self,axis,conditions):
        records=tuple(_graph(c)for c in conditions)
        shared=self.shared
        graphs=[shared.payload.forward_task(r,task=t,identifiers=shared.sequence_identifiers()) for r,t in records]
        state=shared.backbone(graphs)[0][:,0]
        return self.heads[axis.key](state*(self.readout_mult/(self.width/self.base_width))).double()

    def reference_objective(self,rows,*,root):
        """Authenticate original evidence before running one homogeneous assay head."""
        self.get_extra_state();rows=tuple(rows)
        if not 0<len(rows)<=self.max_batch_size or any(type(r) is not NativeBioelectricObservation for r in rows):
            raise ValueError('bounded original native bioelectric observation batch required')
        axis=BioelectricAssayAxis.from_endpoint(rows[0].endpoint)
        if axis not in self.axes or any(BioelectricAssayAxis.from_endpoint(r.endpoint)!=axis for r in rows):
            raise ValueError('one explicitly bound native assay/coordinate/unit partition per objective required')
        values=tuple(r.verify_evidence(root)for r in rows)
        conditions=tuple(r.condition_inputs()for r in rows)
        if axis.coordinate=='native_scalar' and any(r.numeric_target.kind not in (
                'reported_point','censored_left','censored_right','interval','missing')for r in rows):
            raise ValueError('unsupported native bioelectric numeric observation kind')
        raw=self._readout(axis,conditions)
        if not torch.isfinite(raw).all():raise ValueError('finite native bioelectric head coordinates required')
        if axis.coordinate=='native_scalar':
            mean=raw[:,0];scale=F.softplus(raw[:,1])+torch.finfo(torch.float64).eps
            return dict(loss=native_scalar_likelihood(mean,scale,tuple(r.numeric_target for r in rows)),
                mean=mean,scale=scale,observed=sum(x is not None for x in values))
        active=[i for i,x in enumerate(values)if x is not None]
        labels=torch.tensor([axis.category_axis.index(values[i])for i in active],device=raw.device,dtype=torch.long)
        picked=raw[active]
        return dict(loss=F.cross_entropy(picked,labels)if active else picked.sum(),logits=raw,observed=len(active))


def bioelectric_outcome_parameter_roles(model,*,identifier_axis):
    from .annotation_mup import annotation_mup_parameter_roles
    from .parameterization import Axis,ParameterRole,parameter_ledger
    from .mu_optimizer import _storage_signature
    if type(model) is not SharedBioelectricOutcomeModel:raise ValueError('exact native bioelectric owner required')
    model.get_extra_state()
    roles={'shared.'+name:r for name,r in annotation_mup_parameter_roles(model.shared,identifier_axis=identifier_axis).items()}
    for axis in model.axes:
        name='heads.'+axis.key;output=Axis('finite',_sha(['native-bioelectric-assay-readout-v1',axis.binding(),axis.outputs]))
        roles[name+'.weight']=ParameterRole('output',(output,Axis('width','hidden')))
        roles[name+'.bias']=ParameterRole('vector',(output,))
    _storage_signature(model);parameter_ledger(model,roles)
    return roles
