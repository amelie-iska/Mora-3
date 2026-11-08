"""Native source-bound physical metadata training on the GO/EC shared encoder.

Assertion existence/type are explicit supplied queries, not predicted discovery.
Only unit status, form-scope status and observed method are supervised. No numeric
property or canonical/form generation admission is conferred by this trainer.
"""
from dataclasses import dataclass
import torch
from mora3.data.uniprot_physical import physical_batch
from mora3.model.annotation_trainer import AnnotationBatch, _digest
from mora3.model.annotation_multihead import SharedAnnotationModel
from mora3.model.physical_head import StructuredPhysicalHead
from mora3.model.uniprot_go import annotation_graph_task


@dataclass(frozen=True)
class PhysicalTrainingBatch(AnnotationBatch):
    parameter_axis: tuple
    physical_applicable: tuple

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self,'parameter_axis',tuple(self.parameter_axis))
        object.__setattr__(self,'physical_applicable',tuple(self.physical_applicable))
        if len(self.physical_applicable)!=len(self.views) or any(type(x) is not bool for x in self.physical_applicable):
            raise ValueError('Boolean physical applicability per row required')
        # Reproject exact source views. A caller-supplied PhysicalBatch cannot
        # smuggle changed parents, units or observation masks into training.
        object.__setattr__(self,'physical',physical_batch(self.views,parameter_axis=self.parameter_axis))

    def binding(self):
        return _digest(dict(annotation=super().binding(),physical=self.physical.binding_sha256,
                            applicable=self.physical_applicable))

    def applicability(self,device):
        result=super().applicability(device)
        result.update(physical=self.physical,
                      physical_applicable=torch.tensor(self.physical_applicable,dtype=torch.bool,device=device))
        return result


class SharedPhysicalModel(SharedAnnotationModel):
    def __init__(self,vocabulary,go_axis,ec_axis,*,parameter_axis,method_axis,**kwargs):
        super().__init__(vocabulary,go_axis,ec_axis,**kwargs)
        spec=self._configuration.get('mup')
        self.physical=StructuredPhysicalHead(self._configuration['width'],parameter_axis,method_axis,
            parameterization_policy='mup-physical-v1' if spec else 'legacy',
            base_width=spec['base_width'] if spec else None,output_mult=spec['output_mult'] if spec else 1.)
        self._configuration.update(physical=self.physical.get_extra_state(),
                                   objective='sum-observed-applicable-GO-EC-physical-v1')

    def forward(self,views):
        if not views:raise ValueError('nonempty prediction batch required')
        device=next(self.parameters()).device
        graphs=[]
        for v in views:
            record,task=annotation_graph_task(v)
            graphs.append(self.payload.forward_task(record,task=task,identifiers=self.sequence_identifiers()))
        latent=self.backbone(graphs)[0][:,0]
        targets=physical_batch(views,parameter_axis=self.physical.parameter_axis).targets()
        readout=self._readout_features(latent)
        return dict(GO=self.go(readout),EC=self.ec(readout),physical=self.physical(latent,targets))

    def objective(self,logits,views,*,applicable):
        if set(logits)!={'GO','EC','physical'} or set(applicable)!={'GO','EC','physical','physical_applicable'}:
            raise ValueError('all annotation/physical logits and applicability required')
        batch=physical_batch(views,parameter_axis=self.physical.parameter_axis)
        if batch!=applicable['physical']:raise ValueError('physical source binding mismatch')
        active=applicable['physical_applicable'];device=logits['GO'].device
        if not isinstance(active,torch.Tensor) or active.dtype!=torch.bool or active.shape!=(len(views),) or active.device!=device:
            raise ValueError('Boolean physical applicability on logits device required')
        losses=super().objective({k:logits[k] for k in ('GO','EC')},views,
                                applicable={k:applicable[k] for k in ('GO','EC')})
        targets=batch.targets()
        keep=torch.tensor([bool(active[i]) for i,rows in enumerate(targets) for _ in rows],device=device,dtype=torch.bool)
        selected=[[r for r in rows] if bool(active[i]) else [] for i,rows in enumerate(targets)]
        losses['physical']=self.physical.objective({k:z[keep] for k,z in logits['physical'].items()},selected)
        losses['total']=losses['total']+losses['physical']
        return losses
