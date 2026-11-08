"""Ordered two-connexin source FETCH readout, sharing native annotation transforms."""
import copy
from hashlib import sha256
import math
import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.lincx_fetch_native_rows import PANELS,SOURCE_SHA,LinCxNativeFetchSource
from mora3.data.provenance import Provenance
from mora3.data.schema import Edge,GraphRecord,Measurement,Node
from mora3.data.tasks import ConditionalTask
from .annotation_multihead import SharedAnnotationModel
from .mixed_payload_projection import exact_wire


class SharedLinCxFetchModel(nn.Module):
    def __init__(self,shared,*,max_batch_size=256):
        super().__init__()
        if (not isinstance(shared,SharedAnnotationModel) or shared.backbone.configuration()['id_width']!=2
                or type(max_batch_size) is not int or max_batch_size<1):
            raise ValueError('exact shared native width model with ordered rank-two graph and explicit batch bound required')
        shared.validate_parameterization_configuration()
        self.shared=shared;self._owner=(shared,)
        cfg=shared.get_extra_state();p=shared.payload.embedding.weight
        self.width=cfg['width'];self.base_width=cfg['mup']['base_width'];self.readout_mult=cfg['mup']['output_mult']
        self.max_batch_size=max_batch_size
        self.heads=nn.ModuleList(nn.Linear(self.width,2,device=p.device,dtype=p.dtype) for _ in PANELS)
        with torch.no_grad():
            scale=math.sqrt(self.width/self.base_width)
            for h in self.heads:h.weight.mul_(scale);h.bias.mul_(scale)
        self._config=self._local()

    def _local(self):
        return dict(schema='Mora3.SharedLinCxFetchModel/v1',width=self.width,base_width=self.base_width,readout_mult=self.readout_mult,
            max_batch_size=self.max_batch_size,source_manifest_sha256=SOURCE_SHA,literal_panel_axis=PANELS,
            input='known ordered exact source connexin variants and assay relation; no observed outcome, SEM, source ID or derived residues',
            objective='Gaussian native reported mean only; reported standard error retained as audit, never calibration target or predictor',
            target_coordinate='native reported FETCH docking score; no current or tissue efficacy inference',training_admission=False)

    def get_extra_state(self):
        self.shared.validate_parameterization_configuration()
        cfg=self.shared.get_extra_state()
        if (self.shared is not self._owner[0] or self.shared.backbone.configuration()['id_width']!=2
                or set(self._modules)!={'shared','heads'} or not exact_wire(self._config,self._local())
                or (self.width,self.base_width,self.readout_mult)!=(cfg['width'],cfg['mup']['base_width'],cfg['mup']['output_mult'])
                or len(self.heads)!=len(PANELS) or any(type(h) is not nn.Linear or h._modules or h._buffers
                    or tuple(h.weight.shape)!=(2,self.width) or tuple(h.bias.shape)!=(2,) for h in self.heads)):
            raise ValueError('ordered native FETCH shared owner or axis drift')
        return dict(**copy.deepcopy(self._config),shared=cfg)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('exact LinCx native checkpoint required')

    def reference_objective(self,source,rows):
        self.get_extra_state()
        rows=tuple(rows)
        if type(source) is not LinCxNativeFetchSource or len(rows)>self.max_batch_size:
            raise ValueError('exact native table and bounded observed group batch required')
        targets=source.require_rows(rows)
        shared=self.shared;ref=shared.backbone.feature.weight
        ids=torch.eye(2,device=ref.device,dtype=ref.dtype)
        task=ConditionalTask(((('node',0,'variant'),'condition'),(('node',1,'variant'),'condition'),
            (('node',1,'requested_FETCH_score'),'target')),topology_known=True)
        graphs=[]
        for row in rows:
            (one,two),panel=row.conditions()
            r=GraphRecord('known-native-paired-docking-assay',(
                Node('source-assay-first-connexin',(Measurement('variant','|'.join(one),None,'observed',True),)),
                Node('source-assay-second-connexin',(Measurement('variant','|'.join(two),None,'observed',True),
                    Measurement('requested_FETCH_score',None,None,'missing',False)))),
                (Edge(0,1,'known-ordered-FETCH-panel:'+panel,True),),
                (Provenance('native-model-input','LinCx-FETCH-table-S1-v1','input','0'*64,'source-admission-external'),))
            graphs.append(shared.payload.forward_task(r,task=task,identifiers=ids))
        hidden=shared.backbone(graphs)[0][:,0]*(self.readout_mult/(self.width/self.base_width))
        raw=torch.stack(tuple(self.heads[PANELS.index(row.panel)](z) for z,row in zip(hidden,rows,strict=True))).double()
        mean=raw[:,0];scale=F.softplus(raw[:,1])+torch.finfo(torch.float64).eps
        y=torch.tensor(targets,device=raw.device,dtype=raw.dtype)
        terms=0.5*((y-mean)/scale).square()+scale.log()+0.5*math.log(2*math.pi)
        if not torch.isfinite(terms).all():raise ValueError('finite native FETCH likelihood required')
        return dict(loss=terms.mean(),terms=terms,mean=mean,scale=scale,observed_groups=len(rows))


def lincx_fetch_parameter_roles(model,*,identifier_axis):
    from .annotation_mup import annotation_mup_parameter_roles
    from .parameterization import Axis,ParameterRole,parameter_ledger
    from .mu_optimizer import _storage_signature
    if type(model) is not SharedLinCxFetchModel:raise ValueError('exact native FETCH model required')
    model.get_extra_state()
    roles={'shared.'+name:r for name,r in annotation_mup_parameter_roles(model.shared,identifier_axis=identifier_axis).items()}
    for i,panel in enumerate(PANELS):
        axis=Axis('finite',sha256((SOURCE_SHA+'|'+panel+'|mean|raw_scale').encode('utf8')).hexdigest())
        roles[f'heads.{i}.weight']=ParameterRole('output',(axis,Axis('width','hidden')))
        roles[f'heads.{i}.bias']=ParameterRole('vector',(axis,))
    _storage_signature(model);parameter_ledger(model,roles)
    return roles
