"""A native irregular-time likelihood for observed calcium fluorescence traces."""
import copy
from hashlib import sha256
import math

import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.cema_native_trajectories import ARMS_SHA,ORACLE_SHA,CemaNativeTrajectorySource
from .annotation_multihead import SharedAnnotationModel
from .bioelectric_outcomes import _graph
from .mixed_payload_projection import exact_wire

K=32


def irregular_trace_likelihood(mean,scale,rate,times,indices,values):
    """Gaussian Markov likelihood. Omitted samples are integrated over their time gap."""
    m=mean[indices]
    gaps=times[indices[1:]]-times[indices[:-1]]
    attenuation=rate*gaps
    a=torch.exp(-attenuation)
    sd=scale*torch.sqrt(-torch.expm1(-2*attenuation))
    previous=values[:-1]-m[:-1]
    residual=torch.cat((values[:1]-m[:1],values[1:]-m[1:]-a*previous))
    std=torch.cat((scale.reshape(1),sd))
    return 0.5*(residual/std).square()+torch.log(std)+0.5*math.log(2*math.pi)


class SharedCemaTrajectoryModel(nn.Module):
    def __init__(self,shared):
        super().__init__()
        if not isinstance(shared,SharedAnnotationModel):raise ValueError('exact registered annotation owner required')
        shared.validate_parameterization_configuration()
        cfg=shared.get_extra_state();p=shared.payload.embedding.weight
        self.shared=shared;self._owner=(shared,)
        self.width=cfg['width'];self.base_width=cfg['mup']['base_width'];self.readout_mult=cfg['mup']['output_mult']
        self.head=nn.Linear(self.width,K+2,device=p.device,dtype=p.dtype)
        with torch.no_grad():
            factor=math.sqrt(self.width/self.base_width);self.head.weight.mul_(factor);self.head.bias.mul_(factor)
        self._configuration=self._local()

    def _local(self):
        return dict(schema='Mora3.SharedCemaTrajectoryModel/v1',width=self.width,base_width=self.base_width,
          readout_mult=self.readout_mult,source_manifest_sha256=ARMS_SHA,source_oracle_sha256=ORACLE_SHA,
          temporal_mean_basis=dict(kind='normalized-Gaussian-RBF',knots=K,coordinates='equal on closed native time span',bandwidth='one knot spacing'),
          likelihood='stationary initial Gaussian, continuous positive decay/variance, exact missing-time marginalization',
          targets='complete ordered reported fluorescence; 190 missing source values carry no likelihood',
          conditions='known literal exposure and subject context; native minute query grid; no trace/sample ID',
          training_admission=False)

    def get_extra_state(self):
        self.shared.validate_parameterization_configuration();cfg=self.shared.get_extra_state()
        if (self.shared is not self._owner[0] or set(self._modules)!={'shared','head'} or type(self.head) is not nn.Linear
                or self.head._modules or self.head._buffers or tuple(self.head.weight.shape)!=(K+2,self.width)
                or tuple(self.head.bias.shape)!=(K+2,) or not exact_wire(self._local(),self._configuration)
                or (self.width,self.base_width,self.readout_mult)!=(cfg['width'],cfg['mup']['base_width'],cfg['mup']['output_mult'])):
            raise ValueError('trajectory head/shared owner/likelihood configuration drift')
        return dict(**copy.deepcopy(self._configuration),shared=cfg)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('exact source trajectory checkpoint required')

    def reference_objective(self,source,rows):
        """Score native observed time positions only; no split, contrast or causal claim."""
        self.get_extra_state()
        rows=tuple(rows)
        if type(source) is not CemaNativeTrajectorySource or len(rows)>20:
            raise ValueError('exact bounded native CEMA trajectory source required')
        outcomes=source.require_rows(rows)
        if any(not values for values in outcomes):raise ValueError('original empty trajectory has no objective')
        shared=self.shared
        conditions=tuple(r.condition_inputs() for r in rows)
        unique=tuple(dict.fromkeys(tuple(c.items()) for c in conditions))
        records=tuple(_graph(dict(c)) for c in unique)
        graphs=[shared.payload.forward_task(r,task=t,identifiers=shared.sequence_identifiers()) for r,t in records]
        context=shared.backbone(graphs)[0][:,0]
        raw=self.head(context*(self.readout_mult/(self.width/self.base_width))).double()
        if not torch.isfinite(raw).all():raise ValueError('nonfinite trajectory parameters')
        time=torch.tensor(source.time,device=raw.device,dtype=torch.float64)
        normalized=(time-time[0])/(time[-1]-time[0])
        knots=torch.linspace(0,1,K,device=raw.device,dtype=torch.float64)
        basis=torch.softmax(-0.5*((normalized[:,None]-knots)*(K-1)).square(),dim=-1)
        mu=raw[:,:K]@basis.T
        eps=torch.finfo(raw.dtype).eps
        scale=F.softplus(raw[:,K])+eps;rate=F.softplus(raw[:,K+1])+eps
        scores=[];per_trace=[]
        for c,obs in zip(conditions,outcomes):
            u=unique.index(tuple(c.items()))
            indices,values=zip(*obs)
            ids=torch.tensor(indices,device=raw.device,dtype=torch.long)
            y=torch.tensor(values,device=raw.device,dtype=torch.float64)
            nll=irregular_trace_likelihood(mu[u],scale[u],rate[u],time,ids,y)
            if not torch.isfinite(nll).all():raise ValueError('nonfinite complete native trajectory likelihood')
            scores.append(nll);per_trace.append((u,len(obs)))
        loss=torch.cat(scores).mean()
        return dict(loss=loss,nll_by_trace=tuple(scores),mean=mu,scale=scale,rate_per_minute=rate,time=time,
            trace_condition_and_observation_count=tuple(per_trace),observed_count=sum(len(v) for v in outcomes))


def cema_trajectory_parameter_roles(model,*,identifier_axis):
    from .annotation_mup import annotation_mup_parameter_roles
    from .parameterization import Axis,ParameterRole,parameter_ledger
    from .mu_optimizer import _storage_signature
    if type(model) is not SharedCemaTrajectoryModel:raise ValueError('exact native CEMA trajectory owner required')
    model.get_extra_state()
    roles={'shared.'+name:r for name,r in annotation_mup_parameter_roles(model.shared,identifier_axis=identifier_axis).items()}
    labels=tuple(f'normalized-RBF-mean-{i}' for i in range(K))+('raw_scale','raw_rate_per_minute')
    output=Axis('finite',sha256((ARMS_SHA+'|'+ORACLE_SHA+'|'+'|'.join(labels)).encode('ascii')).hexdigest())
    roles['head.weight']=ParameterRole('output',(output,Axis('width','hidden')))
    roles['head.bias']=ParameterRole('vector',(output,))
    _storage_signature(model);parameter_ledger(model,roles)
    return roles
