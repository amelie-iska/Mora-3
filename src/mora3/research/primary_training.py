"""Resumable native MuAdamW update primitive, not a campaign admission issuer.

The caller supplies already authorized source batches/objectives. A launch must
separately require the full owner, data, operator, recovery and resource gates.
Cursor bytes are produced by source readers, never inferred from model outputs.
"""
import copy
import json
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
_PATHSPACE_FIELDS={'schema','trial_id','task_key','source_snapshot_sha256',
    'policy_module_path','history_module_path','prefix_binding_sha256',
    'reward_binding_sha256','sampling_method','terminal_object'}


def _trajectory_freeze_wire(binding):
    """Validate an explicit per-trial full-path method/guide/source declaration."""
    if (type(binding)is not dict or set(binding)!=_PATHSPACE_FIELDS
            or binding.get('schema')!='Mora3.NativePathspaceBinding/v1'
            or any(type(binding.get(k))is not str or not binding[k] for k in
                   ('trial_id','task_key','policy_module_path','history_module_path'))
            or any(type(binding.get(k))is not str or not _SHA.fullmatch(binding[k]) for k in
                   ('source_snapshot_sha256','prefix_binding_sha256','reward_binding_sha256'))
            or binding.get('sampling_method')!='native-unpruned-history-conditioned-forced-horizon'
            or binding.get('terminal_object')!='entire-ordered-reasoning-trajectory'):
        raise ValueError('Exact predeclared native trajectory source/guide/method binding required')
    return json.dumps(binding,sort_keys=True,separators=(',',':'),allow_nan=False).encode()




def _forest_freeze_wire(binding):
    """Pin the optional known-skeleton forest objective, source segment and trial."""
    expected={'schema','trial_id','task_key','source_snapshot_sha256','route_sha256',
              'objective_sha256','heads_sha256','sampling_method','skeleton_policy','total_windows'}
    if (type(binding)is not dict or set(binding)!=expected
            or (binding.get('schema'),binding.get('skeleton_policy'))not in(
                ('Mora3.NativeForestObjectiveBinding/v1','source-declared-fixed-skeleton'),
                ('Mora3.NativeForestObjectiveBinding/v2','source-declared-flow-skeleton-plus-prefix-topology-NLL-v1'))
            or any(type(binding.get(k))is not str or not 0<len(binding[k])<=4096
                   for k in ('trial_id','task_key'))
            or any(type(binding.get(k))is not str or not _SHA.fullmatch(binding[k])
                   for k in ('source_snapshot_sha256','route_sha256','objective_sha256','heads_sha256'))
            or type(binding.get('total_windows'))is not int or not 1<=binding['total_windows']<=1_024_000_000
            or binding.get('sampling_method')!='engine-owned-gaussian-and-uniform-time-v1'):
        raise ValueError('Exact frozen native forest source/objective/trial binding required')
    return json.dumps(binding,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _require_forest_cursor(binding,cursor):
    """Reject malformed/wrong-source/out-of-range positions before state copy."""
    if type(cursor)is not bytes or len(cursor)>4096:
        raise ValueError('Bounded exact forest source cursor bytes required')
    try:
        value=json.loads(cursor)
    except (ValueError,UnicodeError,RecursionError)as exc:
        raise ValueError('Malformed forest source cursor')from exc
    if (type(value)is not dict or set(value)!={'schema','route_sha256','position'}
            or value['schema']!='Mora3.ForestTrainingCursor/v1'
            or value['route_sha256']!=binding['route_sha256']or type(value['position'])is not int
            or not 0<=value['position']<=binding['total_windows']
            or json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()!=cursor):
        raise ValueError('Exact canonical frozen forest source cursor required')


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
                 profile_stage_controller=None,gradient_clip_max_norm=None,sampler_generator=None,
                 trajectory_objective_binding=None,forest_objective_binding=None):
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
        if sampler_generator is not None and not isinstance(sampler_generator,torch.Generator):
            raise ValueError('Exact isolated native trajectory sampler generator required')
        if trajectory_objective_binding is not None:
            if sampler_generator is None:
                raise ValueError('Frozen trajectory objective requires a checkpoint-owned sampler generator')
            if type(trajectory_objective_binding) is not dict or trajectory_objective_binding.get('task_key') not in task_cycle:
                raise ValueError('Frozen trajectory task must belong to exact native task cycle')
            trajectory_freeze_wire=_trajectory_freeze_wire(trajectory_objective_binding)
        else:
            trajectory_freeze_wire=None
        if forest_objective_binding is not None:
            if (sampler_generator is None or type(forest_objective_binding)is not dict
                    or forest_objective_binding.get('task_key')not in task_cycle):
                raise ValueError('Frozen forest task requires a checkpoint-owned sampler and native task cycle')
            forest_freeze_wire=_forest_freeze_wire(forest_objective_binding)
            _require_forest_cursor(forest_objective_binding,source_cursors[forest_objective_binding['task_key']])
        else:
            forest_freeze_wire=None
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
        self.sampler_generator=sampler_generator
        self._sampler_owner=(sampler_generator,)
        self._sampler_device=None if sampler_generator is None else str(sampler_generator.device)
        self._trajectory_objective_binding=copy.deepcopy(trajectory_objective_binding)
        self._trajectory_frozen_wire=trajectory_freeze_wire
        self._forest_objective_binding=copy.deepcopy(forest_objective_binding)
        self._forest_frozen_wire=forest_freeze_wire
        self._scheduler_constants=None if type(scheduler) is not torch.optim.lr_scheduler.StepLR else {
            k:copy.deepcopy(v) for k,v in scheduler.state_dict().items()
            if k not in ('last_epoch','_step_count','_last_lr')}

    def _trajectory_state(self):
        binding=self._trajectory_objective_binding
        if binding is None:
            if self._trajectory_frozen_wire is not None:
                raise ValueError('Frozen trajectory source/method binding was removed')
            return None
        if (_trajectory_freeze_wire(binding)!=self._trajectory_frozen_wire
                or self.sampler_generator is None):
            raise ValueError('Frozen trajectory source/guide/method binding changed')
        return copy.deepcopy(binding)

    def require_trajectory_objective_binding(self,**actual):
        frozen=self._trajectory_state()
        if frozen is None or not exact_wire(frozen,actual):
            raise ValueError('Exact predeclared frozen trajectory source/guide/method/trial binding required')

    def _forest_state(self):
        binding=self._forest_objective_binding
        if binding is None:
            if self._forest_frozen_wire is not None:
                raise ValueError('Frozen forest objective was removed')
            return None
        if (_forest_freeze_wire(binding)!=self._forest_frozen_wire
                or self.sampler_generator is None or binding['task_key']not in self.task_cycle):
            raise ValueError('Frozen forest source/objective/task binding changed')
        _require_forest_cursor(binding,self.source_cursors[binding['task_key']])
        return copy.deepcopy(binding)

    def require_forest_objective_binding(self,**actual):
        frozen=self._forest_state()
        if frozen is None or not exact_wire(frozen,actual):
            raise ValueError('Exact checkpoint-bound forest source/objective/trial required')

    def _sampler_state(self):
        """Exact independent on-policy sampler cursor, separate from global Torch RNG."""
        sampler=self.sampler_generator
        if sampler is not self._sampler_owner[0] or (sampler is not None and
                (not isinstance(sampler,torch.Generator) or str(sampler.device)!=self._sampler_device)):
            raise ValueError('Exact bound trajectory sampler generator ownership required')
        return None if sampler is None else dict(device=self._sampler_device,state=sampler.get_state().clone())

    def _validate_sampler_restore(self,state):
        if (type(state)is not dict or set(state)!={'device','state'}
                or state['device']!=self._sampler_device or not isinstance(state['state'],torch.Tensor)
                or state['state'].dtype!=torch.uint8 or state['state'].device.type!='cpu'
                or state['state'].ndim!=1):
            raise ValueError('Exact native trajectory sampler device and byte cursor required')
        try:
            torch.Generator(device=self._sampler_device).set_state(state['state'])
        except (RuntimeError,ValueError) as exc:
            raise ValueError('Invalid native trajectory sampler generator cursor') from exc

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
        sampler_state=self._sampler_state()
        if sampler_state is not None:
            state.update(schema='Mora3.NativeMuUpdateState/v4',sampler_rng=sampler_state)
        trajectory_state=self._trajectory_state()
        if trajectory_state is not None:
            state.update(schema='Mora3.NativeMuUpdateState/v5',
                         trajectory_objective_binding=trajectory_state)
        forest_state=self._forest_state()
        if forest_state is not None:
            state.update(schema='Mora3.NativeMuUpdateState/v6',forest_objective_binding=forest_state)
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
        sampler_state=self._sampler_state()
        if sampler_state is not None:
            keys.add('sampler_rng')
            schema='Mora3.NativeMuUpdateState/v4'
        trajectory_state=self._trajectory_state()
        if trajectory_state is not None:
            keys.add('trajectory_objective_binding')
            schema='Mora3.NativeMuUpdateState/v5'
        forest_state=self._forest_state()
        if forest_state is not None:
            keys.add('forest_objective_binding')
            schema='Mora3.NativeMuUpdateState/v6'
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
        if sampler_state is not None:self._validate_sampler_restore(state['sampler_rng'])
        if trajectory_state is not None and not exact_wire(
                state['trajectory_objective_binding'],trajectory_state):
            raise ValueError('Checkpoint frozen trajectory source/guide/method binding differs')
        if forest_state is not None and not exact_wire(state['forest_objective_binding'],forest_state):
            raise ValueError('Checkpoint frozen forest source/objective/trial binding differs')
        if forest_state is not None:
            _require_forest_cursor(forest_state,state['source_cursors'][forest_state['task_key']])
        # Native Adam moments are allocated only by completed optimizer updates.
        # A checkpoint cannot carry moments at step zero or moments from a later
        # step than its durable source-cursor/update counter. Reject before the
        # staged copy so an impossible same-run resume never mutates live owners.
        optimizer_state=state['optimizer']
        if type(optimizer_state) is not dict or type(optimizer_state.get('state')) is not dict:
            raise ValueError('exact checkpoint optimizer clock state required')
        for slots in optimizer_state['state'].values():
            step=slots.get('step') if type(slots) is dict else None
            if (type(step) is not torch.Tensor or step.numel()!=1
                    or not bool(torch.isfinite(step).all())
                    or not 1 <= float(step.item()) <= state['steps']):
                raise ValueError('checkpoint optimizer step exceeds completed native update clock')
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
        if sampler_state is not None:self.sampler_generator.set_state(state['sampler_rng']['state'])
        _restore_rng(state['rng'])

    def load_checkpoint(self,path):
        self.restore_state(torch.load(path,map_location='cpu',weights_only=True))
