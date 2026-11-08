"""Admitted observational response CFM training with exact full-state continuation.

Fixed round-robin source pool; independent standard-normal noise and uniform time
per update. Only complete observed expression vectors are supported: missing
coordinates are rejected rather than silently filled into the noisy state. No
physical dose, causal response, heldout or molecular-generation admission follows.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import torch
from mora3.data.sciplex_admission import ObservationalSciPlexTask
from .annotation_trainer import AnnotationTrainer, _digest
from .response_flow import ResponseFlowHead
from .flow_matching import gaussian_path, velocity_loss


@dataclass(frozen=True)
class ResponseFlowBatch:
    tasks: tuple

    def __post_init__(self):
        object.__setattr__(self,'tasks',tuple(self.tasks))
        if not self.tasks or any(not isinstance(t,ObservationalSciPlexTask)
                or t.input_metadata()['task']!='response' for t in self.tasks):
            raise ValueError('nonempty admitted response-only pool required')
        axes=[]
        for task in self.tasks:
            axes.append(tuple(json.loads(task.task.sidecar_bytes)['targets']['expression']['axis_ids']))
            projected=task.project(for_training=True)
            y=projected['targets']['expression']
            if (not projected['target_masks']['expression'].all()
                    or not torch.isfinite(y).all() or (y<0).any()):
                raise ValueError('complete finite nonnegative source counts required')
        if any(a!=axes[0] for a in axes):
            raise ValueError('homogeneous ordered gene axis required')

    @property
    def axis_ids(self):
        return tuple(json.loads(self.tasks[0].task.sidecar_bytes)['targets']['expression']['axis_ids'])

    def binding(self):
        return _digest(dict(version=1,axis=self.axis_ids,
            tasks=[hashlib.sha256(t.canonical_bytes()).hexdigest() for t in self.tasks]))


class ResponseFlowTrainer(AnnotationTrainer):
    """Uses the existing staged model/Adam/CPU+CUDA RNG restore implementation.

    Checkpoint file creation must be exclusive at the caller. Pool identity binds
    hidden targets and receipts for resume only; neither enters observed context.
    All native parameters are owned once; deterministic output heads stay inactive.
    """
    def __init__(self,model,batch,*,batch_size,sigma_min=.01,lr=.001,prepare_tasks=False,
                 mu_family=None,mu_member='target',weight_decay=0.,decay_aliases=()):
        if type(prepare_tasks) is not bool:
            raise ValueError('prepare_tasks must be an explicit bool')
        if (not isinstance(model,ResponseFlowHead) or not isinstance(batch,ResponseFlowBatch)
                or type(batch_size) is not int or not 1<=batch_size<=len(batch.tasks)
                or type(sigma_min) not in (int,float) or not math.isfinite(sigma_min)
                or not 0<=sigma_min<1 or type(lr) not in (int,float)
                or not math.isfinite(lr) or lr<=0):
            raise ValueError('native flow model, admitted pool, bounded batch, finite path/lr required')
        model.validate_configuration()
        if model.config.get('version') in (2,4) and model.config['sigma_min'] != sigma_min:
            raise ValueError('denoiser and training path sigma mismatch')
        if model.native.axis_ids!=batch.axis_ids:
            raise ValueError('flow axis mismatch')
        if model.config['version'] in (3,4):
            if mu_family is None:
                raise ValueError('explicit qualified native-flow muP family required')
            from .flow_mup import flow_mup_parameter_roles
            from .mu_optimizer import MuAdamW
            optimizer=MuAdamW(model,mu_family,flow_mup_parameter_roles(model),base_lr=lr,
                weight_decay=weight_decay,decay_aliases=decay_aliases,member=mu_member)
        else:
            if mu_family is not None or mu_member!='target' or weight_decay!=0 or decay_aliases!=():
                raise ValueError('legacy flow trainer accepts no muP optimizer policy')
            optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        self.model=model;self.batch=batch;self.batch_size=batch_size
        self.sigma_min=sigma_min;self.steps=0
        p=next(model.parameters())
        # Runtime-only immutable raw projections. No learned context, autograd
        # graph, log-target transform or random variate survives a backward.
        # Canonical bytes and all checkpoint semantics remain reference-identical.
        from .sciplex_preparation import PreparedSciPlexTask
        self.training_tasks=(tuple(PreparedSciPlexTask.from_admitted(t,
            axis_ids=batch.axis_ids,device=p.device,dtype=p.dtype) for t in batch.tasks)
            if prepare_tasks else batch.tasks)
        self.optimizer=optimizer
        binding=dict(version=1,trainer='observational-response-cfm-v1',
            pool=batch.binding(),configuration=model.get_extra_state(),
            path=dict(name='gaussian-conditional-2210.02747v2-eq22-23',sigma_min=sigma_min,
                space='log1p-counts',noise='standard-normal',time='uniform-[0,1)',
                availability='complete-observed-only'),
            schedule='round-robin-wrapping',batch_size=batch_size,
            optimizer='Adam',lr=lr,betas=[.9,.999],eps=1e-8,weight_decay=0,
            dtype=str(p.dtype),device=str(p.device))
        if model.config['version'] in (3,4):
            binding.update(version=2,trainer='observational-response-cfm-mup-v1',optimizer='MuAdamW',
                weight_decay=weight_decay,parameterization=self.optimizer.state_dict()['mora3_mu_adamw'])
        self.binding=_digest(binding)

    def state_dict(self):
        self.model.validate_configuration()
        return super().state_dict()

    def load_state_dict(self, state):
        self.model.validate_configuration()
        return super().load_state_dict(state)

    def step(self):
        self.model.validate_configuration()
        indices=tuple((self.steps*self.batch_size+i)%len(self.batch.tasks)
                      for i in range(self.batch_size))
        tasks=tuple(self.training_tasks[i] for i in indices)
        p=next(self.model.parameters())
        projections=[t.project(device=p.device,dtype=p.dtype,for_training=True) for t in tasks]
        y=torch.stack([torch.log1p(v['targets']['expression']) for v in projections])
        available=torch.stack([v['target_masks']['expression'] for v in projections])
        if not available.all():raise ValueError('partial flow states unsupported')
        noise=torch.randn_like(y)
        time=torch.rand(len(tasks),device=p.device,dtype=p.dtype)
        state,velocity=gaussian_path(noise,y,time,sigma_min=self.sigma_min)
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        loss=velocity_loss(self.model(tasks,state,time),velocity,available)
        if loss is None or not torch.isfinite(loss):raise ValueError('nonfinite flow loss')
        loss.backward()
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in self.model.parameters()):
            raise ValueError('nonfinite flow gradient')
        self.optimizer.step();self.steps+=1
        return dict(loss=float(loss.detach()),indices=indices)
