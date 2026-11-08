"""Bounded source-bound conditional TB updates; diagnostic latent reward only.

No molecular decoder or therapeutic reward is implied.
The normalized Gaussian over Euclidean states is uniform over terminal layers.
"""
import copy
import hashlib
import math
from dataclasses import dataclass
import torch
from .profile_gfn import ProfileGFNPolicy, conditional_trajectory_balance
from mora3.reasoning.gfn_sampling import sample_paths, score_paths


@dataclass(frozen=True)
class LatentGaussianReward:
    """Immutable normalized diagnostic density, independent of hidden targets."""
    scale: float = 1.

    def __post_init__(self):
        if type(self.scale) not in (int,float) or not math.isfinite(self.scale) or self.scale<=0:
            raise ValueError('Positive finite diagnostic scale required')

    def binding(self):
        return dict(identity='normalized-isotropic-origin-Gaussian-uniform-layer-v1',
                    scale=self.scale,role='diagnostic-latent-density-not-therapeutic')

    @torch.no_grad()
    def __call__(self,terminal,*,horizon):
        if (type(horizon) is not int or horizon<1 or terminal.ndim!=2
                or not terminal.is_floating_point() or not torch.isfinite(terminal).all()):
            raise ValueError('Finite floating terminal matrix and positive horizon required')
        result=(-.5*(terminal/self.scale).square()-math.log(self.scale)-.5*math.log(2*math.pi)).sum(1)-math.log(horizon)
        if not torch.isfinite(result).all():raise ValueError('Unrepresentable log reward')
        return result.detach()


class ProfileGFNTrainer:
    """One immutable admitted batch with strict full-state recovery; no scheduler.

    The caller supplies a source-bound policy and native admitted views. This
    trainer retains exact input/target/provenance digests outside neural inputs.
    Only the declared diagnostic provider is currently supported.
    """
    def __init__(self,policy,views,*,reward,lr=.001,seed=19):
        if not isinstance(policy,ProfileGFNPolicy) or type(reward) is not LatentGaussianReward:
            raise ValueError('Native policy and explicit diagnostic reward required')
        if type(lr) not in (int,float) or not math.isfinite(lr) or lr<=0 or type(seed) is not int:
            raise ValueError('Finite positive learning rate and integer seed required')
        views=tuple(views)
        if not views:raise ValueError('Nonempty native batch required')
        policy.context(views) # validate directions, full axes and native applicability
        self.policy=policy;self.views=views;self.reward=reward
        self.lr=lr;self.step_number=0
        self._initial_runtime=self._runtime()
        p=next(policy.parameters())
        self.generator=torch.Generator(device=p.device).manual_seed(seed)
        self.optimizer=torch.optim.Adam(policy.parameters(),lr=lr)
        self._optimizer_config=copy.deepcopy(self.optimizer.state_dict()["param_groups"])
        digest=hashlib.sha256()
        for v in views:
            for data in (v.input_bytes,v.target_bytes,v.provenance_bytes):
                digest.update(len(data).to_bytes(8,'big'));digest.update(data)
        self._binding=dict(version=1,policy=policy.get_extra_state(),reward=reward.binding(),
            batch_sha256=digest.hexdigest(),rows=len(views),lr=lr,dtype=str(p.dtype),device=str(p.device),
            optional_stop=True)

    def binding(self):
        return copy.deepcopy(self._binding)

    def _check(self):
        p=next(self.policy.parameters())
        if (self.optimizer.state_dict()["param_groups"]!=self._optimizer_config
                or self._runtime()!=self._initial_runtime
                or self.policy.get_extra_state()!=self._binding['policy']
                or self.reward.binding()!=self._binding['reward']
                or any(g['lr']!=self._binding['lr'] for g in self.optimizer.param_groups)
                or str(p.dtype)!=self._binding['dtype'] or str(p.device)!=self._binding['device']):
            raise ValueError('Trainer binding changed')

    def step(self):
        self._check()
        context=self.policy.context(self.views)
        paths=sample_paths(self.policy,len(self.views),context=context,generator=self.generator)
        x=paths['states']
        terminal=x[torch.arange(len(x),device=x.device),paths['lengths']-1]
        reward=self.reward(terminal,horizon=self.policy.horizon)
        result=conditional_trajectory_balance(self.policy.log_partition(context),
            **score_paths(self.policy,paths,context=context),log_reward=reward,mask=paths['mask'])
        self.optimizer.zero_grad(set_to_none=True)
        result['loss'].backward()
        gradients=[p.grad for p in self.policy.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError('Finite nonempty gradients required')
        self.optimizer.step();self.step_number+=1
        return dict(step=self.step_number,loss=float(result['loss'].detach()),
                    mean_log_reward=float(reward.mean()),reward_role=self.reward.binding()['role'])

    def _runtime(self):
        return dict(threads=torch.get_num_threads(),
            deterministic=torch.are_deterministic_algorithms_enabled(),
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32)

    def state_dict(self):
        """Detached full-state snapshot; exclusive file creation belongs to caller."""
        self._check()
        if (self._runtime()!=self._initial_runtime or self.optimizer.state_dict()["param_groups"]!=self._optimizer_config):
            raise ValueError("GFN runtime/optimizer binding changed")
        return copy.deepcopy(dict(version=1,binding=self.binding(),runtime=self._runtime(),
            steps=self.step_number,model=self.policy.state_dict(),optimizer=self.optimizer.state_dict(),
            initialized_slots=sorted(self.optimizer.state_dict()['state']),
            modes={n:m.training for n,m in self.policy.named_modules()},
            gradients={n:p.grad for n,p in self.policy.named_parameters()},
            generator_rng=self.generator.get_state(),cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if next(self.policy.parameters()).is_cuda else None))

    def load_state_dict(self,state):
        """Validate all candidates off-line before changing live model/Adam/RNG."""
        from .rfam_family_training import validate_adam_state
        self._check()
        if (self.optimizer.state_dict()['param_groups']!=self._optimizer_config
                or state.get('version')!=1 or state.get('binding')!=self.binding()
                or state.get('runtime')!=self._initial_runtime
                or self._runtime()!=self._initial_runtime
                or self.policy.get_extra_state()!=self._binding['policy']):
            raise ValueError('GFN checkpoint source/config/runtime binding mismatch')
        if type(state.get('steps')) is not int or state['steps']<0:
            raise ValueError('Invalid GFN checkpoint step')
        expected=self.optimizer.state_dict()['param_groups'];groups=state['optimizer']['param_groups']
        if groups!=expected:
            raise ValueError('GFN Adam configuration/allocation mismatch')
        validate_adam_state(self.policy,state)
        current=self.policy.state_dict()
        if state['model'].keys()!=current.keys():raise ValueError('GFN model allocation mismatch')
        for name,value in current.items():
            incoming=state['model'][name]
            if isinstance(value,torch.Tensor) and (not isinstance(incoming,torch.Tensor)
                    or incoming.shape!=value.shape or incoming.dtype!=value.dtype):
                raise ValueError('GFN model precision/shape mismatch')
        slots=state['optimizer']['state']
        if (sorted(slots)!=state['initialized_slots'] or (state['steps']>0 and not slots)
                or any(s['step'].item()<1 for s in slots.values())
                or (state['steps']==0 and slots)):
            raise ValueError('GFN incomplete Adam history')
        modes=state['modes'];modules=dict(self.policy.named_modules())
        if modes.keys()!=modules.keys() or any(type(v) is not bool for v in modes.values()):
            raise ValueError('GFN module mode mismatch')
        params=dict(self.policy.named_parameters());grads=state['gradients']
        if grads.keys()!=params.keys():raise ValueError('GFN gradient allocation mismatch')
        ids=[i for group in groups for i in group["params"]]
        for name,pid in zip(params,ids):
            g=grads[name]
            p=params[name]
            if g is not None and (not isinstance(g,torch.Tensor) or g.shape!=p.shape
                    or g.dtype!=p.dtype or not torch.isfinite(g).all()):
                raise ValueError('Invalid GFN gradient')
            if g is not None and (pid not in slots or slots[pid]['step'].item()!=state['steps']):
                raise ValueError('GFN active parameter missing current Adam history')
        # Both objects copied together to preserve optimizer parameter ownership.
        candidate,adam=copy.deepcopy((self.policy,self.optimizer))
        candidate.load_state_dict(state['model'],strict=True)
        adam.load_state_dict(state['optimizer'])
        device=next(self.policy.parameters()).device
        local=torch.Generator(device=device);local.set_state(state['generator_rng'].cpu())
        cpu=torch.Generator();cpu.set_state(state['cpu_rng'].cpu())
        cuda=state['cuda_rng']
        if device.type=='cuda':
            if not isinstance(cuda,list) or len(cuda)!=torch.cuda.device_count():
                raise ValueError('GFN CUDA RNG allocation mismatch')
            for i,rng in enumerate(cuda):torch.Generator(device=f'cuda:{i}').set_state(rng.cpu())
        elif cuda is not None:raise ValueError('Unexpected GFN CUDA RNG')
        self.policy.load_state_dict(candidate.state_dict())
        self.optimizer.load_state_dict(adam.state_dict())
        for name,m in modules.items():m.training=modes[name]
        for name,p in params.items():p.grad=None if grads[name] is None else grads[name].to(p.device).clone()
        self.step_number=state['steps'];self.generator.set_state(local.get_state())
        torch.set_rng_state(cpu.get_state())
        if cuda is not None:torch.cuda.set_rng_state_all([r.cpu() for r in cuda])
