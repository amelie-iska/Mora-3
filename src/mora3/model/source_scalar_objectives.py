"""Source-native scalar heads on the same annotation/TokenGT encoder.

One density or interval-mass likelihood per native partition. The source
reporting coordinate of AZ logD is not a physically asserted equality, and
FreeSolv's possibly default error bar is not a measured observation scale.
"""
from dataclasses import dataclass
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.admet import NativeEntity, canonical
from mora3.data.provenance import Provenance
from mora3.data.schema import GraphRecord,Measurement,Node
from mora3.data.tasks import ConditionalTask
from mora3.data.source_scalar_native_rows import SCOPES, ScalarTarget, require_native_scalar_batch
from .admet_objectives import gaussian_censored_nll
from .annotation_multihead import SharedAnnotationModel
from .mixed_payload_projection import exact_wire


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _is_sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def _valid_key(key):
    return (type(key) is tuple and len(key)==7
        and all(type(key[i]) is str and key[i] for i in (0,1,2,6))
        and SCOPES.get(key[0])==key[1]
        and all(key[i] is None or type(key[i]) is str and key[i] for i in (3,4,5)))


@dataclass(frozen=True)
class SourceScalarAxis:
    proof_sha256: str
    filter_payload_sha256: str
    partitions: tuple

    def __post_init__(self):
        if (not _is_sha(self.proof_sha256) or not _is_sha(self.filter_payload_sha256)
                or type(self.partitions) is not tuple or not self.partitions
                or any(type(row) is not tuple or len(row)!=2 or not _valid_key(row[0])
                       or type(row[1]) is not int or row[1]<1 for row in self.partitions)
                or len({key for key,_ in self.partitions})!=len(self.partitions)):
            raise ValueError('complete unique verified scalar source partitions required')

    @classmethod
    def from_verified_projection(cls,path,expected_sha256,*,root):
        root=Path(root).resolve()
        def trusted(ref):
            target=(root/ref['path']).resolve();target.relative_to(root)
            raw=target.read_bytes()
            if ('bytes' in ref and len(raw)!=ref['bytes']) or _sha(raw)!=ref['sha256']:
                raise ValueError('verified scalar source reference changed')
            return raw
        p=(root/path).resolve();p.relative_to(root)
        raw=p.read_bytes()
        if _sha(raw)!=expected_sha256:raise ValueError('exact verified scalar projection SHA required')
        v=json.loads(raw)
        if (v.get('schema')!='Mora3.FullNativeScalarProjectionVerification/v1'
                or v.get('exact_native_target_float64_and_json_number_tokens') is not True
                or v.get('original_roles_flags_and_artifacts_unchanged') is not True
                or v.get('optimizer_or_task_training_authorized') is not False
                or v.get('explicit_unqualified_source_rows')!=0):
            raise ValueError('complete scoped source scalar readback required')
        trusted(v['reader'])
        partitions=tuple((tuple(row['partition_key']),row['rows']) for row in v['partitions'])
        if sum(count for _,count in partitions)!=v['all_native_source_rows']:
            raise ValueError('native scalar partition counts do not close')
        filt=json.loads(trusted(v['source_filter']))
        return cls(expected_sha256,_sha(canonical(filt)),partitions)


def scalar_condition_graph(key,inputs):
    if not _valid_key(key) or type(inputs) is not dict or set(inputs)!={'task','native_entity','endpoint','known_context'} or inputs['task']!='source_scalar_prediction':
        raise ValueError('strict native target-free scalar condition required')
    family,_,quantity,unit,method,_,coordinate=key
    entity=inputs['native_entity'];q=inputs['endpoint'];context=inputs['known_context']
    molecular=family!='UniProt-kinetics'
    if (type(entity) is not dict or set(entity)!={'molecule_class','representation_type','representation'}
            or entity['molecule_class']!=('small_molecule' if molecular else 'protein')
            or entity['representation_type']!=('SMILES' if molecular else 'amino_acid_sequence')
            or type(q) is not dict or q.get('readout')!=quantity or q.get('semantic_name')!=quantity
            or q.get('unit')!=unit or q.get('scale')!=coordinate or type(context) is not dict):
        raise ValueError('native condition entity/endpoint/coordinate differs from fixed partition')
    if family=='FreeSolv' and context!={}:raise ValueError('FreeSolv assay context was not supplied')
    if family=='ChEMBL37-AZ-experimental-logD7_4' and not exact_wire(context,dict(
            pH=7.4,solvent_pair=['octan-1-ol','water'],method=method)):
        raise ValueError('known AZ logD context differs from source contract')
    if family=='UniProt-kinetics':
        alias={'Km':'substrate_alias','Vmax':'enzyme_alias'}.get(quantity)
        if (alias is None or set(context)!={alias,'measurement_type','literal_unit'}
                or type(context[alias]) is not str or not context[alias]
                or context['measurement_type']!=quantity or context['literal_unit']!=unit):
            raise ValueError('exact native kinetic alias/type/unit required')
    expected=dict(semantic_name=quantity,description='Reported native source coordinate; physical and assay missingness remains explicit',
        readout=quantity,unit=unit,unit_basis='unresolved' if unit is None else 'source_literal',
        scale=coordinate,measurement_type=quantity,binary_readout=None,
        unit_applicability='unknown' if unit is None else 'declared_native_unit')
    if not exact_wire(q,expected):raise ValueError('exact known scalar endpoint query semantics required')
    NativeEntity.from_mapping(entity)
    record=GraphRecord('input',(Node('source-scalar',(
        Measurement('condition',canonical(inputs).decode('utf8'),None,'observed',True),
        Measurement('requested_output',None,None,'missing',False))),),(),
        (Provenance('native-model-input','source-scalar-v1','input','0'*64,'source-admission-external'),))
    task=ConditionalTask(((("node",0,"condition"),'condition'),(("node",0,"requested_output"),'target')),topology_known=True)
    return record,task


_RELATIONS={'reported_point':'equal','reported_nominal_coordinate':'equal',
    'censored_left':'left','censored_right':'right','interval':'interval','missing':'unobserved'}


def _validate_targets(targets,count,*,nominal):
    targets=tuple(targets)
    if len(targets)!=count or any(type(t) is not ScalarTarget or t.kind not in _RELATIONS
            or t.kind!='missing' and (t.kind=='reported_nominal_coordinate')!=nominal for t in targets):
        raise ValueError('one qualified observation in the native partition coordinate per input required')
    for t in targets:t.__post_init__()
    return targets


def native_scalar_likelihood(mean,scale,targets,*,nominal=False):
    """Float64 likelihood on reported numbers/bounds, not on source error bars."""
    if (not isinstance(mean,torch.Tensor) or not isinstance(scale,torch.Tensor)
            or mean.ndim!=1 or not mean.numel() or scale.shape!=mean.shape or scale.device!=mean.device
            or mean.dtype!=torch.float64 or scale.dtype!=torch.float64 or type(nominal) is not bool):
        raise ValueError('matched nonempty Float64 Gaussian coordinates required')
    targets=_validate_targets(targets,mean.numel(),nominal=nominal)
    observed=torch.tensor([t.numeric_coordinate_available for t in targets],device=mean.device,dtype=torch.bool)
    applicable=torch.ones_like(observed)
    values=lambda field: mean.new_tensor([getattr(t,field) if getattr(t,field) is not None else math.nan for t in targets])
    # The reused density branch scores an observed nominal number as a number.
    # Native raw_relation stays None; this does not assert physical equality.
    return gaussian_censored_nll(mean,scale,relation=tuple(_RELATIONS[t.kind] for t in targets),
        value=values('reported_value'),lower=values('lower'),upper=values('upper'),
        observed=observed,applicable=applicable,applicability_known=applicable)


@dataclass(frozen=True)
class ScalarPartitionObjective:
    partition_key: tuple
    original_batch_indices: tuple
    loss: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor
    observation_kinds: tuple


class SharedNativeScalarModel(nn.Module):
    """Finite native statistics per partition; no duplicate encoder weights."""
    def __init__(self,shared,axis,*,max_batch_size=128):
        super().__init__()
        if not isinstance(shared,SharedAnnotationModel) or type(axis) is not SourceScalarAxis or type(max_batch_size) is not int or max_batch_size<1:
            raise ValueError('exact shared annotation owner, source axis, and positive batch bound required')
        shared.validate_parameterization_configuration();axis.__post_init__()
        cfg=shared.get_extra_state();p=shared.payload.embedding.weight
        self.shared=shared;self._owner=(shared,);self.axis=axis;self.max_batch_size=max_batch_size
        self.width=p.shape[1];self.base_width=cfg['mup']['base_width'];self.output_mult=cfg['mup']['output_mult']
        self.heads=nn.ModuleDict({_sha(canonical(key)):nn.Linear(self.width,2) for key,_ in axis.partitions})
        with torch.no_grad():
            for head in self.heads.values():
                factor=math.sqrt(self.width/self.base_width);head.weight.mul_(factor);head.bias.mul_(factor)
        self.heads.to(device=p.device,dtype=p.dtype)
        self._configuration=self._local_state()

    def _local_state(self):
        return dict(schema='Mora3.SharedNativeScalarModel/v1',width=self.width,base_width=self.base_width,
            output_mult=self.output_mult,max_batch_size=self.max_batch_size,proof_sha256=self.axis.proof_sha256,
            filter_payload_sha256=self.axis.filter_payload_sha256,partitions=self.axis.partitions,
            objective='separate-source-native-partition-Float64-Gaussian-density-or-censor-mass',
            readout='native-fresh-affine-times-sqrt-width-over-base;input-divided-by-width-over-base',
            uncertainty='target-side-audit-not-likelihood-scale',training_admission=False)

    def get_extra_state(self):
        self.shared.validate_parameterization_configuration();cfg=self.shared.get_extra_state()
        if (self.shared is not self._owner[0] or set(self._modules)!={'shared','heads'} or type(self.heads) is not nn.ModuleDict
                or not exact_wire(self._local_state(),self._configuration)
                or (self.width,self.base_width,self.output_mult)!=(cfg['width'],cfg['mup']['base_width'],cfg['mup']['output_mult'])
                or tuple(self.heads)!=tuple(_sha(canonical(key)) for key,_ in self.axis.partitions)
                or any(type(h) is not nn.Linear or h._modules or h._buffers or set(h._parameters)!={'weight','bias'}
                       or tuple(h.weight.shape)!=(2,self.width) or tuple(h.bias.shape)!=(2,) for h in self.heads.values())):
            raise ValueError('native scalar shared owner/partition/readout drift')
        return dict(**copy.deepcopy(self._configuration),shared=cfg)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('native scalar checkpoint source/partition mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        for name,value in self.state_dict().items():
            if not isinstance(value,torch.Tensor) and not exact_wire(value,state_dict.get(prefix+name)):
                raise ValueError('native scalar checkpoint source/partition mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _partition(self,key):
        if not _valid_key(key) or _sha(canonical(key)) not in self.heads:raise ValueError('unregistered native scalar partition')
        return self.heads[_sha(canonical(key))]

    def _prepare(self,key,conditions):
        head=self._partition(key);conditions=tuple(conditions)
        if not 0<len(conditions)<=self.max_batch_size:raise ValueError('complete bounded native scalar batch required')
        return head,tuple(scalar_condition_graph(key,c) for c in conditions)

    def _evaluate(self,head,records):
        shared=self.shared
        graphs=[shared.payload.forward_task(r,task=t,identifiers=shared.sequence_identifiers()) for r,t in records]
        z=shared.backbone(graphs)[0][:,0]
        raw=head(z*(self.output_mult/(self.width/self.base_width))).double()
        return raw[:,0],F.softplus(raw[:,1])+torch.finfo(torch.float64).eps

    def forward(self,key,conditions):
        self.get_extra_state();return self._evaluate(*self._prepare(key,conditions))

    def source_batch_objectives(self,view,batch,*,split):
        """Authenticate every original row and return separate native partitions."""
        self.get_extra_state()
        filt=getattr(view,'effective_filter',None)
        if type(filt) is not dict or _sha(canonical(filt))!=self.axis.filter_payload_sha256:
            raise ValueError('scalar batch must use the exact verified native filter snapshot')
        require_native_scalar_batch(view,batch,split=split)
        if not 0<len(batch.scalar_rows)<=self.max_batch_size:
            raise ValueError('complete bounded native scalar batch required')
        groups=defaultdict(list)
        for i,row in enumerate(batch.scalar_rows):groups[row.partition_key].append((i,row))
        prepared=[]
        for key,rows in groups.items():
            head,records=self._prepare(key,(r.condition_inputs() for _,r in rows))
            nominal=key[0]=='ChEMBL37-AZ-experimental-logD7_4'
            targets=_validate_targets((r.target for _,r in rows),len(rows),nominal=nominal)
            prepared.append((key,tuple(i for i,_ in rows),head,records,nominal,targets))
        results=[]
        for key,indices,head,records,nominal,targets in prepared:
            mean,scale=self._evaluate(head,records)
            loss=native_scalar_likelihood(mean,scale,targets,nominal=nominal)
            results.append(ScalarPartitionObjective(key,indices,loss,mean,scale,tuple(t.kind for t in targets)))
        return tuple(results)


def source_scalar_mup_parameter_roles(model,*,identifier_axis):
    from .annotation_mup import annotation_mup_parameter_roles
    from .parameterization import Axis,ParameterRole,parameter_ledger
    from .mu_optimizer import _storage_signature
    if type(model) is not SharedNativeScalarModel:raise ValueError('exact native scalar owner required')
    model.get_extra_state()
    roles={'shared.'+name:role for name,role in annotation_mup_parameter_roles(model.shared,identifier_axis=identifier_axis).items()}
    for key,_ in model.axis.partitions:
        name='heads.'+_sha(canonical(key));finite=Axis('finite',_sha(canonical(['source-native-scalar-v1',key,['mean','raw_scale']])))
        roles[name+'.weight']=ParameterRole('output',(finite,Axis('width','hidden')))
        roles[name+'.bias']=ParameterRole('vector',(finite,))
    _storage_signature(model);parameter_ledger(model,roles)
    return roles
