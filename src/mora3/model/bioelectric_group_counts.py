"""Native aggregate laterality likelihood for a direct HCN4 perturbation study."""
import copy
from hashlib import sha256
import math
import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.hcn4_native_laterality import CATEGORIES,LEDGER_SHA,MARKERS,SOURCE_SHA,Hcn4SourceCountGroups
from .annotation_multihead import SharedAnnotationModel
from .bioelectric_outcomes import _graph
from .mixed_payload_projection import exact_wire


class SharedHcn4LateralityModel(nn.Module):
    def __init__(self,shared):
        super().__init__()
        if not isinstance(shared,SharedAnnotationModel):raise ValueError('exact registered annotation owner required')
        shared.validate_parameterization_configuration()
        self.shared=shared;self._owner=(shared,)
        c=shared.get_extra_state();p=shared.payload.embedding.weight
        self.width=c['width'];self.base_width=c['mup']['base_width'];self.readout_mult=c['mup']['output_mult']
        self.heads=nn.ModuleList(nn.Linear(self.width,len(CATEGORIES),device=p.device,dtype=p.dtype) for _ in MARKERS)
        with torch.no_grad():
            factor=math.sqrt(self.width/self.base_width)
            for h in self.heads:h.weight.mul_(factor);h.bias.mul_(factor)
        self._configuration=self._local()

    def _local(self):
        return dict(schema='Mora3.SharedHcn4LateralityModel/v1',width=self.width,base_width=self.base_width,readout_mult=self.readout_mult,
            native_marker_axis=MARKERS,native_category_axis=CATEGORIES,source_manifest_sha256=SOURCE_SHA,native_count_ledger_sha256=LEDGER_SHA,
            likelihood='mean of aggregate multinomial negative log probabilities per native reported group; no inferred replicate pairing',
            conditions='known original perturbation group and queried marker only; sample counts/outcomes/source IDs excluded',
            training_admission=False)

    def get_extra_state(self):
        self.shared.validate_parameterization_configuration();cfg=self.shared.get_extra_state()
        if (self.shared is not self._owner[0] or set(self._modules)!={'shared','heads'} or not exact_wire(self._local(),self._configuration)
                or (self.width,self.base_width,self.readout_mult)!=(cfg['width'],cfg['mup']['base_width'],cfg['mup']['output_mult'])
                or len(self.heads)!=len(MARKERS) or any(type(h) is not nn.Linear or h._modules or h._buffers
                or tuple(h.weight.shape)!=(len(CATEGORIES),self.width) or tuple(h.bias.shape)!=(len(CATEGORIES),) for h in self.heads)):
            raise ValueError('native group likelihood model configuration drift')
        return dict(**copy.deepcopy(self._configuration),shared=cfg)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('exact native grouped-count checkpoint required')

    def reference_objective(self,source,rows):
        self.get_extra_state()
        if type(source) is not Hcn4SourceCountGroups:raise ValueError('exact source group view required')
        rows=tuple(rows)
        targets=source.require_rows(rows)
        shared=self.shared
        graphs=[]
        for row in rows:
            r,t=_graph(row.conditions())
            graphs.append(shared.payload.forward_task(r,task=t,identifiers=shared.sequence_identifiers()))
        latent=shared.backbone(graphs)[0][:,0]*(self.readout_mult/(self.width/self.base_width))
        logits=torch.stack(tuple(self.heads[row.source_table-1](h) for h,row in zip(latent,rows,strict=True))).double()
        counts=torch.tensor(targets,device=logits.device,dtype=torch.float64)
        ll=counts.mul(F.log_softmax(logits,dim=1)).sum(1)+torch.lgamma(counts.sum(1)+1)-torch.lgamma(counts+1).sum(1)
        if not torch.isfinite(ll).all():raise ValueError('nonfinite native count likelihood')
        return dict(loss=-ll.mean(),group_nll=-ll,logits=logits,groups=len(rows),observed_count_positions=counts.numel())


def hcn4_laterality_parameter_roles(model,*,identifier_axis):
    from .annotation_mup import annotation_mup_parameter_roles
    from .parameterization import Axis,ParameterRole,parameter_ledger
    from .mu_optimizer import _storage_signature
    if type(model) is not SharedHcn4LateralityModel:raise ValueError('exact HCN4 group model required')
    model.get_extra_state()
    roles={'shared.'+name:r for name,r in annotation_mup_parameter_roles(model.shared,identifier_axis=identifier_axis).items()}
    for i,marker in enumerate(MARKERS):
        axis=Axis('finite',sha256((LEDGER_SHA+'|'+marker+'|'+'|'.join(CATEGORIES)).encode('utf8')).hexdigest())
        roles[f'heads.{i}.weight']=ParameterRole('output',(axis,Axis('width','hidden')))
        roles[f'heads.{i}.bias']=ParameterRole('vector',(axis,))
    _storage_signature(model);parameter_ledger(model,roles)
    return roles
