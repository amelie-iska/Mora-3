"""Explicit unpaired annotation/image/response native cyclic training.

Equal updates per active objective are an experimental schedule, not inferred
biological pairing or a calibrated corpus mixture. Admission remains source-owned.
"""
from .configuration_snapshot import configuration_snapshot
from .mixed_payload_projection import exact_wire
import copy
import math
import torch
from torch import nn
from .annotation_trainer import AnnotationTrainer, _digest
from .cooperative_training import objective_schedule
from .physical_trainer import PhysicalTrainingBatch
from mora3.data.sciplex_admission import ObservationalSciPlexTask


class UnifiedModel(nn.Module):
    def __init__(self, native, annotation, *, parameterization_policy="legacy"):
        super().__init__()
        if parameterization_policy not in ('legacy','mup-cooperative-v1'):
            raise ValueError('known explicit unified parameterization policy required')
        new=parameterization_policy!='legacy'
        if new:
            native.validate_parameterization_configuration()
            annotation.validate_parameterization_configuration()
            ns=native.response.get_extra_state();an=annotation.get_extra_state()
            if (getattr(native,'parameterization_policy','legacy')!=parameterization_policy
                    or ns['mup']['base_width']!=an['mup']['base_width']
                    or ns['mup']['output_mult']!=an['mup']['output_mult']):
                raise ValueError('unified native/annotation parameterization configuration mismatch')
        elif (getattr(native,'parameterization_policy','legacy')!='legacy'
                or annotation.get_extra_state().get('version')!=1):
            raise ValueError('new native/annotation owners require explicit unified parameterization')
        self.parameterization_policy=parameterization_policy
        self._parameterization_declaration=parameterization_policy
        if not (native.image.backbone is native.response.backbone is annotation.backbone):
            raise ValueError('one shared native backbone required')
        self.native=native
        self.annotation=annotation
        self.configuration=dict(version=1,architecture='unpaired-native-annotation-image-response-v1',
            native=native.get_extra_state(),annotation=annotation.get_extra_state(),
            payload=annotation.payload.get_extra_state(),pairing='none-inferred')
        if new:self.configuration.update(version=2,parameterization_policy=parameterization_policy)

    def validate_parameterization_configuration(self):
        if self._parameterization_declaration=='legacy':
            raise ValueError('explicit unified parameterization required')
        if (self.parameterization_policy!=self._parameterization_declaration
                or self.configuration.get('parameterization_policy')!=self._parameterization_declaration):
            raise ValueError('unified parameterization policy drift')
        self.native.validate_parameterization_configuration()
        self.annotation.validate_parameterization_configuration()
        if (not exact_wire(self.configuration.get('native'),self.native.get_extra_state())
                or not exact_wire(self.configuration.get('annotation'),self.annotation.get_extra_state())
                or not exact_wire(self.configuration.get('payload'),self.annotation.payload.get_extra_state())
                or not (self.native.image.backbone is self.native.response.backbone is self.annotation.backbone)):
            raise ValueError('unified shared owner configuration drift')

    def get_extra_state(self):
        if self._parameterization_declaration!='legacy':self.validate_parameterization_configuration()
        return configuration_snapshot(self.configuration)

    def set_extra_state(self,state):
        if state!=self.configuration:raise ValueError('unified configuration mismatch')


class UnifiedTrainer(AnnotationTrainer):
    """Own all heads/Adam once; reuse staged full-state restore validation."""
    def __init__(self,model,batch,pools,*,batch_size,lr=.001):
        if getattr(model,'parameterization_policy','legacy')!='legacy':
            raise ValueError('new unified owner requires complete cooperative MuAdamW family; legacy Adam trainer unavailable')
        if not isinstance(batch,PhysicalTrainingBatch):
            raise ValueError('source-bound physical annotation batch required')
        if not math.isfinite(lr) or lr<=0:raise ValueError('positive finite learning rate required')
        self.model=model;self.batch=batch;self.steps=0
        self.pools={k:tuple(v) for k,v in pools.items()}
        objective_schedule(self.pools,step=0,batch_size=batch_size)
        for kind,tasks in self.pools.items():
            for task in tasks:
                if kind=='image':
                    from mora3.data.livecell_graph import LiveCellGraphTask
                    from mora3.data.raster_morphology import RasterMorphologyTask
                    if isinstance(task,LiveCellGraphTask):task.example.require_training()
                    elif isinstance(task,RasterMorphologyTask):task.require_training()
                    else:raise ValueError('admitted native image task required')
                    if task.direction!='image_to_mask':raise ValueError('image_to_mask required')
                elif not isinstance(task,ObservationalSciPlexTask) or task.input_metadata()['task']!=kind:
                    raise ValueError('admitted matching observational task required')
        self.batch_size=copy.deepcopy(batch_size)
        self.native_kinds=tuple(k for k in ('image','response','intervention') if self.pools.get(k))
        p=next(model.parameters())
        self.binding=_digest(dict(configuration=model.get_extra_state(),annotation=batch.binding(),
            native=model.native.binding(self.pools,batch_size=batch_size),
            schedule=('annotation',)+self.native_kinds,policy='equal-active-objective-cyclic-v1',
            device=str(p.device),dtype=str(p.dtype),deterministic=torch.are_deterministic_algorithms_enabled(),
            optimizer='Adam',lr=lr,betas=[.9,.999],eps=1e-8,weight_decay=0))
        self.optimizer=torch.optim.Adam(model.parameters(),lr=lr)

    def step(self):
        kinds=('annotation',)+self.native_kinds
        round_index,slot=divmod(self.steps,len(kinds))
        kind=kinds[slot]
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        if kind=='annotation':
            ann=self.model.annotation
            loss=ann.objective(ann(self.batch.views),self.batch.views,
                applicable=self.batch.applicability(next(self.model.parameters()).device))['total']
            count=len(self.batch.views)
        else:
            selected,tasks=objective_schedule(self.pools,
                step=round_index*len(self.native_kinds)+slot-1,batch_size=self.batch_size)
            assert selected==kind
            loss=self.model.native.objective(kind,tasks);count=len(tasks)
        if not torch.isfinite(loss):raise ValueError('nonfinite unified loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError('nonfinite unified gradient')
        self.optimizer.step();self.steps+=1
        return dict(step=self.steps,kind=kind,examples=count,loss=float(loss.detach()))
