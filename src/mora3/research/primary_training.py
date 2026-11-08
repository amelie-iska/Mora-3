"""Resumable native MuAdamW update primitive, not a campaign admission issuer.

The caller supplies already authorized source batches/objectives. A launch must
separately require the full owner, data, operator, recovery and resource gates.
Cursor bytes are produced by source readers, never inferred from model outputs.
"""
import copy
import math
from pathlib import Path
import random
import re

import numpy as np
import torch
from torch import nn

from mora3.model.mu_optimizer import MuAdamW
from mora3.model.mixed_payload_projection import exact_wire

from .mup_campaign_schedule import NativeMuWarmupCosine, SCHEMA as COSINE_SCHEMA

_SHA=re.compile('[0-9a-f]{64}\\Z')


def _rng_state():
    state=np.random.get_state()
    return dict(python=random.getstate(),numpy=(state[0],state[1].tolist(),state[2],state[3],state[4]),
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def _validate_rng(state):
    if type(state)is not dict or set(state)!={'python','numpy','torch_cpu','torch_cuda'}:
        raise ValueError('complete Python/NumPy/Torch RNG state required')
    random.Random(0).setstate(state['python'])
    n=state['numpy']
    if (type(n)is not tuple or len(n)!=5 or n[0]!='MT19937' or type(n[1])is not list
            or len(n[1])!=624 or any(type(v)is not int or not 0<=v<2**32 for v in n[1])
            or type(n[2])is not int or not 0<=n[2]<=624 or type(n[3])is not int or n[3]not in(0,1)
            or type(n[4])not in(int,float) or not math.isfinite(n[4])):
        raise ValueError('exact native NumPy MT19937 state required')
    np.random.RandomState(0).set_state((n[0],np.array(n[1],dtype=np.uint32),*n[2:]))
    cpu=state['torch_cpu']
    if not isinstance(cpu,torch.Tensor) or cpu.dtype!=torch.uint8 or cpu.device.type!='cpu' or cpu.ndim!=1:
        raise ValueError('exact CPU generator byte state required')
    torch.Generator().set_state(cpu)
    cuda=state['torch_cuda']
    expected=torch.cuda.device_count() if torch.cuda.is_initialized() else 0
    if type(cuda)is not list or len(cuda)!=expected:
        raise ValueError('initialized CUDA generator topology differs')
    for i,value in enumerate(cuda):
        if not isinstance(value,torch.Tensor) or value.dtype!=torch.uint8 or value.device.type!='cpu' or value.ndim!=1:
            raise ValueError('exact CUDA generator byte states required')
        torch.Generator(device=torch.device('cuda',i)).set_state(value)


def _restore_rng(state):
    random.setstate(state['python'])
    n=state['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),*n[2:]))
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:torch.cuda.set_rng_state_all(state['torch_cuda'])


class NativeMuUpdateEngine:
    """One optimizer, explicit task cycle, full RNG and reader-cursor recovery.

    Task-cycle multiplicities declare experimental update weights; they do not
    assert corpus proportionality, biological pairing, or comparable loss units.
    This primitive does not read credentials, initialize W&B or launch training.
    """
    def __init__(self,model,optimizer,*,binding_sha256,task_cycle,source_cursors,scheduler=None,
                 profile_stage_controller=None,gradient_clip_max_norm=None):
        if (not isinstance(model,nn.Module) or type(optimizer)is not MuAdamW
                or optimizer._mup_model()is not model or type(binding_sha256)is not str
                or not _SHA.fullmatch(binding_sha256)
                or type(task_cycle)is not tuple or not task_cycle
                or any(type(name)is not str or not name for name in task_cycle)
                or type(source_cursors)is not dict or set(source_cursors)!=set(task_cycle)
                or any(type(value)is not bytes or not value for value in source_cursors.values())
                or scheduler is not None and(type(scheduler) not in (torch.optim.lr_scheduler.StepLR, NativeMuWarmupCosine)
                    or scheduler.optimizer is not optimizer or scheduler.last_epoch!=0 or scheduler._step_count!=1)):
            raise ValueError('exact native owner/optimizer/binding/task cycle and source cursor states required')
        if gradient_clip_max_norm is not None and (
            type(gradient_clip_max_norm) not in (int,float)
            or not math.isfinite(gradient_clip_max_norm) or gradient_clip_max_norm <= 0
        ):
            raise ValueError('Explicit finite positive global gradient clip max norm required')
        optimizer._validate_live()
        if type(scheduler) is NativeMuWarmupCosine:
            scheduler.state_dict()
        self._gradient_clip_max_norm = (None if gradient_clip_max_norm is None else float(gradient_clip_max_norm))
        self._profile_owner_path=None
        if profile_stage_controller is not None:
            from mora3.model.profile_reward_stage import ProfileRewardStageController
            if type(profile_stage_controller)is not ProfileRewardStageController:
                raise ValueError('exact native staged profile reward controller required')
            paths=[name for name,module in model.named_modules()if module is profile_stage_controller.template]
            if len(paths)!=1:raise ValueError('profile reward must bind the exact registered live reconstruction')
            self._profile_owner_path=paths[0];profile_stage_controller.require_current(0)
        self.profile_stage_controller=profile_stage_controller
        self.model=model;self.optimizer=optimizer;self.scheduler=scheduler
        self._schedule_owner=(scheduler,)
        self._configured_update_policy=(type(scheduler),self._gradient_clip_max_norm)
        self.binding_sha256=binding_sha256;self.task_cycle=task_cycle
        self.source_cursors=copy.deepcopy(source_cursors);self.steps=0
        self._scheduler_constants=None if type(scheduler) is not torch.optim.lr_scheduler.StepLR else {
            k:copy.deepcopy(v) for k,v in scheduler.state_dict().items()
            if k not in ('last_epoch','_step_count','_last_lr')}

    def _update_policy(self):
        kind,clip = self._configured_update_policy
        cosine = kind is NativeMuWarmupCosine
        if clip is None and not cosine:
            return None
        if (self.scheduler is not self._schedule_owner[0] or type(self.scheduler) is not kind
                or not exact_wire(self._gradient_clip_max_norm,clip)):
            raise ValueError('Exact configured scheduler and gradient clip policy ownership required')
        return dict(
            schema='Mora3.NativeMuUpdatePolicy/v1',
            scheduler=COSINE_SCHEMA if cosine else ('StepLR-v1' if self.scheduler is not None else None),
            gradient_clip_max_norm=clip,
            gradient_clip_norm_type=2. if clip is not None else None,
        )

    def _validate_scheduler(self,state,steps,optimizer):
        if type(self.scheduler) is NativeMuWarmupCosine:
            self.scheduler.validate_state(state,steps,optimizer)
            return
        if self.scheduler is None:
            if state is not None:raise ValueError('unexpected scheduler state')
            return
        expected=set(self.scheduler.state_dict())
        if (type(state)is not dict or set(state)!=expected
                or not exact_wire({k:v for k,v in state.items() if k in self._scheduler_constants},self._scheduler_constants)
                or type(state['last_epoch'])is not int or state['last_epoch']!=steps
                or type(state['_step_count'])is not int or state['_step_count']!=steps+1
                or not exact_wire(state['_last_lr'],[g['lr'] for g in optimizer.param_groups])):
            raise ValueError('exact scheduler policy/step/common native LR state required')

    @property
    def next_task(self):return self.task_cycle[self.steps%len(self.task_cycle)]

    def _bound_profile_stage(self):
        controller=self.profile_stage_controller
        if controller is not None and self.model.get_submodule(self._profile_owner_path)is not controller.template:
            raise ValueError('registered staged reconstruction owner changed')
        return controller

    def prepare_profile_stage(self):
        controller=self._bound_profile_stage()
        if controller is None:raise ValueError('no staged profile reward attached')
        return controller.prepare(self.steps)

    def update(self,task,loss,*,cursor_before,cursor_after):
        policy = self._update_policy()
        if (task!=self.next_task or type(cursor_before)is not bytes or type(cursor_after)is not bytes
                or self.source_cursors[task]!=cursor_before or not cursor_after or cursor_after==cursor_before):
            raise ValueError('exact next task and advancing authorized source-reader cursor required')
        if (not isinstance(loss,torch.Tensor) or loss.ndim!=0 or not loss.is_floating_point()
                or not loss.requires_grad or not bool(torch.isfinite(loss))):
            raise ValueError('finite scalar differentiable native objective required')
        controller=self._bound_profile_stage()
        if controller is not None:controller.require_current(self.steps)
        self.optimizer._validate_live()
        if type(self.scheduler) is NativeMuWarmupCosine:
            self.scheduler.require_step_available(self.steps)
        if self.scheduler is not None:self._validate_scheduler(self.scheduler.state_dict(),self.steps,self.optimizer)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        active=[p for p in self.model.parameters() if p.grad is not None]
        if not active or any(not bool(torch.isfinite(p.grad).all()) for p in active):
            if policy is not None:
                self.optimizer.zero_grad(set_to_none=True)
            raise ValueError('finite nonempty native gradients required; optimizer/cursor not advanced')
        gradient_norm = None
        if self._gradient_clip_max_norm is not None:
            try:
                gradient_norm = float(nn.utils.clip_grad_norm_(
                    active,self._gradient_clip_max_norm,norm_type=2.,error_if_nonfinite=True,foreach=False,
                ))
            except RuntimeError as exc:
                self.optimizer.zero_grad(set_to_none=True)
                raise ValueError('finite global gradient norm required; optimizer/cursor not advanced') from exc
        self.optimizer.step()
        if self.scheduler is not None:self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.source_cursors[task]=cursor_after;self.steps+=1
        result = dict(step=self.steps,task=task,loss=float(loss.detach()),
                      active_parameter_tensors=len(active),source_cursor=cursor_after)
        if gradient_norm is not None:
            result['gradient_norm_before_clip'] = gradient_norm
        return result

    def checkpoint_state(self):
        self.optimizer._validate_live()
        if any(p.grad is not None for p in self.model.parameters()):
            raise ValueError('checkpoint only after completed update with cleared gradients')
        controller=self._bound_profile_stage()
        policy = self._update_policy()
        if policy is not None:
            schedule_state = None if self.scheduler is None else self.scheduler.state_dict()
            self._validate_scheduler(schedule_state,self.steps,self.optimizer)
        state=dict(schema='Mora3.NativeMuUpdateState/v1',binding_sha256=self.binding_sha256,
            task_cycle=self.task_cycle,source_cursors=self.source_cursors,steps=self.steps,
            model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            scheduler=(schedule_state if policy is not None
                       else None if self.scheduler is None else self.scheduler.state_dict()),rng=_rng_state())
        if controller is not None:
            state.update(schema='Mora3.NativeMuUpdateState/v2',profile_stage=dict(
                reconstruction_owner_path=self._profile_owner_path,state=controller.checkpoint(self.steps)))
        if policy is not None:
            state.update(schema='Mora3.NativeMuUpdateState/v3',update_policy=policy)
        return copy.deepcopy(state)

    def save_checkpoint(self,path):
        state=self.checkpoint_state()
        # Partial failed new files remain inspectable; no replacement/cleanup.
        with Path(path).open('xb') as handle:torch.save(state,handle)

    def restore_state(self,state):
        controller=self._bound_profile_stage()
        keys={'schema','binding_sha256','task_cycle','source_cursors','steps','model','optimizer','scheduler','rng'}
        schema='Mora3.NativeMuUpdateState/v1'
        if controller is not None:keys.add('profile_stage');schema='Mora3.NativeMuUpdateState/v2'
        policy = self._update_policy()
        if policy is not None:
            keys.add('update_policy')
            schema='Mora3.NativeMuUpdateState/v3'
        if (type(state)is not dict or set(state)!=keys or state['schema']!=schema
                or state['binding_sha256']!=self.binding_sha256 or state['task_cycle']!=self.task_cycle
                or type(state['steps'])is not int or state['steps']<0
                or type(state['source_cursors'])is not dict or set(state['source_cursors'])!=set(self.task_cycle)
                or any(type(v)is not bytes or not v for v in state['source_cursors'].values())
                or (state['scheduler']is None)!=(self.scheduler is None)):
            raise ValueError('exact source/model/mixture/cursor/checkpoint binding required')
        if policy is not None and not exact_wire(state['update_policy'],policy):
            raise ValueError('exact checkpointed schedule and gradient clip policy required')
        _validate_rng(state['rng'])
        live=self.model.state_dict()
        if type(state['model'])not in(dict,type(live)) or set(state['model'])!=set(live):
            raise ValueError('complete model checkpoint keys required')
        for name,value in state['model'].items():
            if isinstance(value,torch.Tensor) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError('nonfinite model checkpoint refused')
        selected=getattr(self.optimizer._mup_family,self.optimizer._mup_member)
        for entry in selected.parameters:
            values=[state['model'][name] for name in entry.aliases]
            if any(not isinstance(v,torch.Tensor) or not torch.equal(v,values[0]) for v in values):
                raise ValueError('checkpoint tied alias values disagree')
        # Paired copy is required: copied optimizer must own copied parameters.
        model,opt,scheduler=copy.deepcopy((self.model,self.optimizer,self.scheduler))
        model.load_state_dict(state['model'],strict=True)
        opt.load_state_dict(state['optimizer'])
        self._validate_scheduler(state['scheduler'],state['steps'],opt)
        if scheduler is not None:scheduler.load_state_dict(state['scheduler'])
        opt._validate_live()
        if scheduler is not None and scheduler.optimizer is not opt:
            raise ValueError('staged scheduler ownership drift')
        staged_profile=None
        if controller is not None:
            stage=state['profile_stage']
            if (type(stage)is not dict or set(stage)!={'reconstruction_owner_path','state'}
                    or stage['reconstruction_owner_path']!=self._profile_owner_path):
                raise ValueError('exact checkpointed registered reconstruction path required')
            staged_profile=controller.stage_restore(stage['state'],steps=state['steps'],
                staged_template=model.get_submodule(self._profile_owner_path))
        # Only a fully staged recovery may mutate live owners or global RNG.
        self.model.load_state_dict(state['model'],strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler is not None:self.scheduler.load_state_dict(state['scheduler'])
        if controller is not None:controller.apply_staged_restore(staged_profile)
        self.steps=state['steps'];self.source_cursors=copy.deepcopy(state['source_cursors'])
        self.optimizer.zero_grad(set_to_none=True)
        _restore_rng(state['rng'])

    def load_checkpoint(self,path):
        self.restore_state(torch.load(path,map_location='cpu',weights_only=True))
