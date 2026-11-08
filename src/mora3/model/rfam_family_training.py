"""Exact-byte-bound Rfam alternating Adam training and explicit i99 continuation.

Consumes reader-admitted immutable views; binding is not independent admission.
Unknown family labels remain unknown. No heldout eligibility is conferred.
"""
from dataclasses import dataclass
import hashlib
import math
import torch
from mora3.data.rfam_family import RfamFamilyView, project_family_task
from .annotation_trainer import AnnotationTrainer, _digest


@dataclass(frozen=True)
class RfamFamilyBatch:
    prediction: tuple
    generation: tuple

    def __post_init__(self):
        for direction in ('prediction','generation'):
            vs=tuple(getattr(self,direction));object.__setattr__(self,direction,vs)
            if not vs or any(not isinstance(v,RfamFamilyView) for v in vs):
                raise ValueError('nonempty reader-admitted views required')
            for v in vs:
                task='Rfam_source_family_'+direction+'_prior'
                if v.provenance().get('envelope',{}).get('task_name')!=task:
                    raise ValueError('Rfam batch direction mismatch')
                project_family_task(dict(schema='mora3.CorpusTask/v1',task=task,
                    conditions=v.inputs(),targets=v.targets()),direction)

    def binding(self):
        # Length-delimited field hashes bind all bytes, row order, audit and groups.
        return _digest({d:[[hashlib.sha256(x).hexdigest() for x in
            (v.condition_bytes,v.target_bytes,v.provenance_bytes)] for v in getattr(self,d)]
            for d in ('prediction','generation')})

    def legacy_binding(self):
        return {d:hashlib.sha256(b''.join(v.condition_bytes+v.target_bytes+v.provenance_bytes
            for v in getattr(self,d))).hexdigest() for d in ('prediction','generation')}


class RfamFamilyTrainer(AnnotationTrainer):
    """Alternating full batches; transactional full-state restore inherited."""
    schedule=('prediction','generation')

    def __init__(self,model,batch,*,lr=.001):
        if not isinstance(batch,RfamFamilyBatch) or not math.isfinite(lr) or lr<=0:
            raise ValueError('bound batch and positive finite learning rate required')
        for d in self.schedule:model._views(getattr(batch,d),d)
        self.model=model;self.batch=batch;self.steps=0
        self.optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        self.binding=self._current_binding()

    def _current_binding(self):
        return _digest(dict(batch=self.batch.binding(),configuration=self.model.get_extra_state(),
            schedule=self.schedule,optimizer=self.optimizer.state_dict()['param_groups'],policy=self.policy()))

    def policy(self):
        p=next(self.model.parameters())
        return dict(device=str(p.device),dtype=str(p.dtype),threads=torch.get_num_threads(),
            deterministic=torch.are_deterministic_algorithms_enabled(),
            cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32)

    def step(self):
        # Recompute binding to detect runtime precision/determinism policy changes.
        if self._current_binding()!=self.binding:raise ValueError('Rfam runtime policy/batch mismatch')
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        direction=self.schedule[self.steps%2]
        loss=getattr(self.model,direction+'_objective')(getattr(self.batch,direction))
        if not torch.isfinite(loss):raise ValueError('nonfinite Rfam loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError('nonfinite Rfam gradient')
        self.optimizer.step();self.steps+=1
        return dict(step=self.steps,direction=direction,loss=loss.item())

    def load_state_dict(self,state):
        if self._current_binding()!=self.binding:
            raise ValueError('Rfam runtime policy/batch mismatch')
        # Torch accepts malformed/nonfinite Adam slots on load; reject before any
        # live model/optimizer/RNG mutation. Inactive parameters may lack slots.
        validate_adam_state(self.model, state)
        super().load_state_dict(state)

    def import_i99_state(self,state):
        """Explicit local diagnostic migration; does not overwrite old files.

        i99 did not record CPU thread/determinism flags in its checkpoint. Caller
        must reproduce its saved script policy; this is not general old-format
        compatibility or evidence of historical losses/timing.
        """
        p=next(self.model.parameters())
        if (state.get('binding')!=self.batch.legacy_binding() or state.get('step')!=4
                or tuple(state.get('schedule',()))!=self.schedule or state.get('seed')!=9901
                or state.get('device')!=p.device.type or state.get('dtype')!='FP32'
                or p.dtype!=torch.float32):
            raise ValueError('i99 checkpoint source/schedule/precision mismatch')
        self.load_state_dict(dict(version=1,binding=self.binding,steps=4,model=state['model'],
            optimizer=state['adam'],cpu_rng=state['cpu_rng'],cuda_rng=state['cuda_rng']))


def validate_adam_state(model, state):
    """Reject malformed Adam slots and nonfinite weights before live mutation."""
    opt=state['optimizer'];groups=opt['param_groups']
    ids=[i for g in groups for i in g['params']];params=list(model.parameters())
    if len(ids)!=len(params) or len(set(ids))!=len(ids) or set(opt['state'])-set(ids):
        raise ValueError('Rfam Adam parameter allocation mismatch')
    for pid,p in zip(ids,params):
        slot=opt['state'].get(pid)
        if slot is None:continue
        if set(slot)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Rfam Adam slots mismatch')
        for key in ('exp_avg','exp_avg_sq'):
            x=slot[key]
            if not isinstance(x,torch.Tensor) or x.shape!=p.shape or x.dtype!=p.dtype or not torch.isfinite(x).all():
                raise ValueError('invalid Rfam Adam moment')
            if key=='exp_avg_sq' and (x<0).any():raise ValueError('negative Adam variance')
        step=slot['step']
        if not isinstance(step,torch.Tensor) or step.numel()!=1 or not torch.isfinite(step).all() or step.item()<0 or step.item()!=int(step.item()) or step.item()>state['steps']:
            raise ValueError('invalid Adam step')
    for value in state['model'].values():
        if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError('nonfinite Rfam model state')
