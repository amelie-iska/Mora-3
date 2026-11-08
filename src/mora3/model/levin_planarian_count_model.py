"""Conditional native planarian count marginal likelihood, without role admission."""
import copy
from hashlib import sha256
import math
import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.levin_planarian_counts import NATIVE_SHA,OUTCOMES,SCOPES,PlanarianFragmentCountSource
from .annotation_multihead import SharedAnnotationModel
from .bioelectric_outcomes import _graph
from .mixed_payload_projection import exact_wire


def count_marginal_nll(logits,counts):
    if (not isinstance(logits,torch.Tensor) or logits.ndim!=2 or logits.shape[1]!=2 or not len(logits)
            or len(logits)!=len(counts)
            or any(type(row)is not tuple or len(row)!=2 or any(type(n)is not int or n<0 for n in row)
                   or sum(row)<1 for row in counts)):
        raise ValueError('exact ordered reported count-pair marginals required')
    c=torch.tensor(counts,device=logits.device,dtype=torch.float64)
    nll=-(c.mul(F.log_softmax(logits.double(),dim=1)).sum(1)+torch.lgamma(c.sum(1)+1)-torch.lgamma(c+1).sum(1))
    if not torch.isfinite(nll).all():
        raise ValueError('nonfinite reported count marginal likelihood')
    return nll


class SharedPlanarianCountModel(nn.Module):
    def __init__(self,shared):
        super().__init__()
        if not isinstance(shared,SharedAnnotationModel):raise ValueError('exact shared annotation owner required')
        shared.validate_parameterization_configuration()
        self.shared=shared;self._owner=(shared,)
        cfg=shared.get_extra_state();p=shared.payload.embedding.weight
        self.width=cfg['width'];self.base_width=cfg['mup']['base_width'];self.output_mult=cfg['mup']['output_mult']
        self.heads=nn.ModuleList(nn.Linear(self.width,2,device=p.device,dtype=p.dtype) for _ in SCOPES)
        with torch.no_grad():
            factor=math.sqrt(self.width/self.base_width)
            for h in self.heads:h.weight.mul_(factor);h.bias.mul_(factor)
        self._configuration=self._local()

    def _local(self):
        return dict(schema='Mora3.SharedPlanarianFragmentCountModel/v1',source_manifest_sha256=NATIVE_SHA,
            native_reported_endpoint_axis=OUTCOMES,native_empirical_scope_axis=SCOPES,width=self.width,
            base_width=self.base_width,readout_mult=self.output_mult,training_admission=False,
            loss='mean descriptive marginal count NLL; parent linkage/independence are not established',
            conditions='original intervention/cut arm and fragment only; no round identity, counts or source coordinates')

    def get_extra_state(self):
        self.shared.validate_parameterization_configuration()
        cfg=self.shared.get_extra_state()
        if (self.shared is not self._owner[0] or set(self._modules)!={'shared','heads'}
                or not exact_wire(self._configuration,self._local())
                or (self.width,self.base_width,self.output_mult)!=(cfg['width'],cfg['mup']['base_width'],cfg['mup']['output_mult'])
                or len(self.heads)!=len(SCOPES)
                or any(type(h)is not nn.Linear or h._modules or h._buffers or
                       tuple(h.weight.shape)!=(2,self.width) or tuple(h.bias.shape)!=(2,) for h in self.heads)):
            raise ValueError('planarian count head or scope configuration drift')
        return dict(**copy.deepcopy(self._configuration),shared=cfg)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):
            raise ValueError('exact native planarian count checkpoint required')

    def research_reference_objective(self,source,rows,*,allow_development_exposure=False):
        # The current parent study is proposed validation. A caller who requests
        # a research diagnostic must explicitly record that exposure.
        self.get_extra_state()
        if type(source)is not PlanarianFragmentCountSource or allow_development_exposure is not True or torch.is_grad_enabled():
            raise ValueError('research source reference requires explicitly recorded development exposure')
        rows=tuple(rows);counts=source.require_rows(rows)
        shared=self.shared
        tasks=tuple(_graph(r.conditions())for r in rows)
        graphs=[shared.payload.forward_task(g,task=t,identifiers=shared.sequence_identifiers())for g,t in tasks]
        context=shared.backbone(graphs)[0][:,0]*(self.output_mult/(self.width/self.base_width))
        logits=torch.stack(tuple(self.heads[SCOPES.index(r.assay_scope)](h)for h,r in zip(context,rows,strict=True)))
        nll=count_marginal_nll(logits,counts)
        return dict(loss=nll.mean(),group_nll=nll,logits=logits,groups=len(rows),training_admission=False,
                    source_role_assigned=False,development_exposure=True)


def planarian_count_parameter_roles(model,*,identifier_axis):
    from .annotation_mup import annotation_mup_parameter_roles
    from .parameterization import Axis,ParameterRole,parameter_ledger
    from .mu_optimizer import _storage_signature
    if type(model)is not SharedPlanarianCountModel:raise ValueError('exact planarian fragment count model required')
    model.get_extra_state()
    roles={'shared.'+name:r for name,r in annotation_mup_parameter_roles(model.shared,identifier_axis=identifier_axis).items()}
    for i,scope in enumerate(SCOPES):
        axis=Axis('finite',sha256((NATIVE_SHA+'|'+scope+'|'+'|'.join(OUTCOMES)).encode()).hexdigest())
        roles[f'heads.{i}.weight']=ParameterRole('output',(axis,Axis('width','hidden')))
        roles[f'heads.{i}.bias']=ParameterRole('vector',(axis,))
    _storage_signature(model);parameter_ledger(model,roles)
    return roles
