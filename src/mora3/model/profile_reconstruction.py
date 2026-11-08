"""Fixed frozen-policy latent draws for experimental source reconstruction.

Sampling once from a copied policy authenticates row/path association. Subsequent
reconstruction changes the decoder/profile encoder, never the frozen draw. This
is neither on-policy GFN training nor a molecular likelihood or therapeutic reward.
"""
import copy
import hashlib
import json
import torch
from .profile_latent_decoder import ProfileLatentSelfiesDecoder
from .profile_gfn import ProfileGFNPolicy
from mora3.reasoning.gfn_sampling import sample_paths


def _digest(value):
    h=hashlib.sha256()
    # Per-call only: repeated exact immutable string tuples (full biological
    # axes in aliased extra-state) have the same legacy byte stream. Never cache
    # tensors, mutable containers, subclasses, or values across calls.
    axes = {}
    remaining = 8 * 1024 * 1024
    def visit(x):
        nonlocal remaining
        if type(x) is tuple and len(x) >= 16:
            previous = axes.get(id(x))
            if previous is not None:
                h.update(previous[1])
                return
            if (remaining and all(type(v) is str for v in x)
                    and 12 * sum(map(len, x)) + 20 * (len(x) + 1) <= remaining):
                chunks = []
                for v in (len(x), *x):
                    data = json.dumps(v, sort_keys=True, allow_nan=False).encode()
                    chunks.extend((len(data).to_bytes(8, 'big'), data))
                data = b''.join(chunks)
                axes[id(x)] = (x, data)  # retain identity for this traversal
                remaining -= len(data)
                h.update(data)
                return
        if isinstance(x,torch.Tensor):
            visit((str(x.dtype),tuple(x.shape)))
            data=x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        elif isinstance(x,dict):
            for k in sorted(x):visit(k);visit(x[k])
            return
        elif isinstance(x,(tuple,list)):
            visit(len(x))
            for v in x:visit(v)
            return
        else:data=json.dumps(x,sort_keys=True,allow_nan=False).encode()
        h.update(len(data).to_bytes(8,'big'));h.update(data)
    visit(value)
    return h.hexdigest()


def _batch(views):
    h=hashlib.sha256()
    for v in views:
        for data in (v.input_bytes,v.target_bytes,v.provenance_bytes):
            h.update(len(data).to_bytes(8,'big'));h.update(data)
    return h.hexdigest()


class FixedProfileReconstruction:
    """Source/target bytes and frozen sampling policy remain outside features.

    Retains detached terminal samples only; full path digest plus policy and RNG
    states identify the original draw. Checkpoint persistence belongs to trainer.
    """
    def __init__(self,model,policy,views,*,seed):
        if (not isinstance(model,ProfileLatentSelfiesDecoder)
                or not isinstance(policy,ProfileGFNPolicy) or type(seed) is not int
                or model.decoder.profiles is not policy.profiles
                or model._policy!=policy.get_extra_state()):
            raise ValueError('Matching native decoder/policy and integer seed required')
        views=tuple(views)
        if not views:raise ValueError('Nonempty source batch required')
        model.decoder.teacher_batch(views) # source-observed molecules must be encodable
        self._views=views
        frozen=copy.deepcopy(policy).eval()
        p=next(frozen.parameters())
        generator=torch.Generator(device=p.device).manual_seed(seed)
        before=generator.get_state().clone()
        with torch.no_grad():
            context=frozen.context(views)
            paths=sample_paths(frozen,len(views),context=context,generator=generator)
            self._latents=model.terminal_latents(paths).clone()
        self._configuration=copy.deepcopy(model.get_extra_state())
        self._binding=dict(version=1,semantics='fixed-frozen-policy-source-reconstruction-v1',
            batch_sha256=_batch(views),rows=len(views),policy_sha256=_digest(policy.state_dict()),
            policy_modes={n:m.training for n,m in policy.named_modules()},sampling_mode='eval',
            path_sha256=_digest(paths),latent_sha256=_digest(self._latents),
            generator_before_sha256=_digest(before),generator_after_sha256=_digest(generator.get_state()),
            seed=seed,device=str(p.device),precision=str(p.dtype),configuration=self._configuration)

    def binding(self):
        return copy.deepcopy(self._binding)

    def latents(self):
        if _digest(self._latents)!=self._binding['latent_sha256']:
            raise ValueError('Fixed latent bytes changed')
        return self._latents.clone()

    def validate_policy(self,policy):
        if _digest(policy.state_dict())!=self._binding['policy_sha256']:
            raise ValueError('Frozen policy checkpoint changed')

    def objective(self,model,*,views=None):
        views=self._views if views is None else tuple(views)
        if _batch(views)!=self._binding['batch_sha256']:
            raise ValueError('Reconstruction batch/order changed')
        if model.get_extra_state()!=self._configuration:
            raise ValueError('Reconstruction configuration changed')
        return model.objective(views,self.latents())

    def subset_objective(self,model,indices):
        """Select views and their original fixed terminals with the same indices.

        Full source bytes remain authenticated even when only a subset is active.
        Empty/repeated rows are rejected; intentional oversampling needs its own
        explicit scheduling contract rather than a hidden subset weighting.
        """
        indices=tuple(indices)
        if (not indices or any(type(i) is not int or not 0<=i<len(self._views)
                               for i in indices) or len(set(indices))!=len(indices)):
            raise ValueError('Nonempty unique bounded integer row indices required')
        if _batch(self._views)!=self._binding['batch_sha256']:
            raise ValueError('Reconstruction batch/order changed')
        if model.get_extra_state()!=self._configuration:
            raise ValueError('Reconstruction configuration changed')
        latents=self.latents()
        return model.objective(tuple(self._views[i] for i in indices),latents[list(indices)])

    def state_dict(self):
        """Detached draw archive; digest detects corruption, not hostile forgery."""
        payload=dict(version=1,binding=self.binding(),latents=self.latents())
        return dict(payload,sha256=_digest(payload))

    @classmethod
    def from_state_dict(cls,model,policy,views,state):
        """Restore a trusted draw archive without sampling or changing RNG/model.

        This is batch persistence only, not optimizer/trainer recovery. The
        archive's digest is an integrity check, not a signature or source proof.
        """
        if not isinstance(state,dict) or set(state)!={'version','binding','latents','sha256'}:
            raise ValueError('Invalid fixed reconstruction archive')
        payload={k:state[k] for k in ('version','binding','latents')}
        if state['version']!=1 or _digest(payload)!=state['sha256']:
            raise ValueError('Fixed reconstruction archive integrity mismatch')
        if (not isinstance(model,ProfileLatentSelfiesDecoder)
                or not isinstance(policy,ProfileGFNPolicy)
                or model._policy!=policy.get_extra_state()):
            raise ValueError('Native matching reconstruction configuration required')
        views=tuple(views);binding=state['binding'];latent=state['latents']
        p=model.latent_projection.weight
        if (not isinstance(binding,dict) or binding.get('version')!=1
                or binding.get('semantics')!='fixed-frozen-policy-source-reconstruction-v1'
                or binding.get('configuration')!=model.get_extra_state()
                or binding.get('batch_sha256')!=_batch(views)
                or binding.get('rows')!=len(views) or not views
                or binding.get('device')!=str(p.device) or binding.get('precision')!=str(p.dtype)
                or binding.get('sampling_mode')!='eval'
                or binding.get('policy_sha256')!=_digest(policy.state_dict())
                or not isinstance(latent,torch.Tensor)
                or latent.shape!=(len(views),model.dimension)
                or latent.device!=p.device or latent.dtype!=p.dtype
                or latent.requires_grad or not torch.isfinite(latent).all()
                or binding.get('latent_sha256')!=_digest(latent)):
            raise ValueError('Fixed reconstruction source/policy/latent binding mismatch')
        model.decoder.teacher_batch(views)
        result=cls.__new__(cls)
        result._views=views;result._latents=latent.detach().clone()
        result._binding=copy.deepcopy(binding)
        result._configuration=copy.deepcopy(binding['configuration'])
        return result


class ProfileReconstructionTrainer:
    """Fixed-draw experimental reconstruction with bound Adam and strict recovery.

    No sampling occurs here. Exclusive checkpoint-file creation is caller-owned.
    State dictionaries are trusted local artifacts, not adversarial signatures.
    """
    def __init__(self,model,batch,*,lr=.001):
        import math
        if (not isinstance(model,ProfileLatentSelfiesDecoder)
                or not isinstance(batch,FixedProfileReconstruction)
                or type(lr) not in (int,float) or not math.isfinite(lr) or lr<=0):
            raise ValueError('Native model, fixed batch and positive finite lr required')
        batch.objective(model)
        self.model=model;self.batch=batch;self.steps=0
        self.optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        self._groups=copy.deepcopy(self.optimizer.state_dict()['param_groups'])
        self._binding=batch.binding();self._runtime=self.runtime()

    def runtime(self):
        p=next(self.model.parameters())
        return dict(device=str(p.device),dtype=str(p.dtype),threads=torch.get_num_threads(),
            deterministic=torch.are_deterministic_algorithms_enabled(),
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32)

    def _check(self):
        if (self.runtime()!=self._runtime or self.batch.binding()!=self._binding
                or self.model.get_extra_state()!=self._binding['configuration']
                or self.optimizer.state_dict()['param_groups']!=self._groups
                or _batch(self.batch._views)!=self._binding['batch_sha256']):
            raise ValueError('Reconstruction source/config/runtime/Adam binding changed')
        self.batch.latents()

    def step(self):
        self._check()
        self.optimizer.zero_grad(set_to_none=True)
        loss=self.batch.objective(self.model)
        if not torch.isfinite(loss):raise ValueError('Nonfinite reconstruction loss')
        loss.backward()
        grads=[p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise ValueError('Finite nonempty reconstruction gradients required')
        self.optimizer.step();self.steps+=1
        return dict(step=self.steps,loss=loss.item())

    def state_dict(self):
        self._check()
        return copy.deepcopy(dict(version=1,binding=self._binding,runtime=self._runtime,
            batch=self.batch.state_dict(),steps=self.steps,model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            modes={n:m.training for n,m in self.model.named_modules()},
            gradients={n:p.grad for n,p in self.model.named_parameters()},
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()
            if next(self.model.parameters()).is_cuda else None))

    def load_state_dict(self,state):
        """Validate candidate weights/slots/modes/gradients/RNG before mutation."""
        from .rfam_family_training import validate_adam_state
        self._check()
        if (not isinstance(state,dict) or state.get('version')!=1
                or state.get('binding')!=self._binding or state.get('runtime')!=self._runtime
                or type(state.get('steps')) is not int or state['steps']<0
                or _digest(state.get('batch'))!=_digest(self.batch.state_dict())
                or state['optimizer']['param_groups']!=self._groups):
            raise ValueError('Reconstruction checkpoint binding mismatch')
        validate_adam_state(self.model,state)
        current=self.model.state_dict()
        if current.keys()!=state['model'].keys():raise ValueError('Model allocation mismatch')
        for name,v in current.items():
            incoming=state['model'][name]
            if isinstance(v,torch.Tensor) and (not isinstance(incoming,torch.Tensor)
                    or incoming.shape!=v.shape or incoming.dtype!=v.dtype):
                raise ValueError('Model shape/precision mismatch')
        params=dict(self.model.named_parameters());modules=dict(self.model.named_modules())
        grads=state['gradients'];modes=state['modes']
        if (grads.keys()!=params.keys() or modes.keys()!=modules.keys()
                or any(type(v) is not bool for v in modes.values())):
            raise ValueError('Gradient/mode allocation mismatch')
        slots=state['optimizer']['state'];ids=self._groups[0]['params']
        if (state['steps']>0 and not slots) or (state['steps']==0 and slots):
            raise ValueError('Incomplete Adam history')
        for (name,p),pid in zip(params.items(),ids):
            g=grads[name]
            if g is not None:
                if (not isinstance(g,torch.Tensor) or g.shape!=p.shape or g.dtype!=p.dtype
                        or not torch.isfinite(g).all() or pid not in slots
                        or slots[pid]['step'].item()!=state['steps']):
                    raise ValueError('Invalid gradient/current Adam history')
        candidate,adam=copy.deepcopy((self.model,self.optimizer))
        candidate.load_state_dict(state['model'],strict=True);adam.load_state_dict(state['optimizer'])
        cpu=torch.Generator();cpu.set_state(state['cpu_rng'].cpu())
        cuda=state['cuda_rng'];device=next(self.model.parameters()).device
        if device.type=='cuda':
            if not isinstance(cuda,list) or len(cuda)!=torch.cuda.device_count():
                raise ValueError('CUDA RNG allocation mismatch')
            for i,rng in enumerate(cuda):torch.Generator(device=f'cuda:{i}').set_state(rng.cpu())
        elif cuda is not None:raise ValueError('Unexpected CUDA RNG')
        self.model.load_state_dict(candidate.state_dict());self.optimizer.load_state_dict(adam.state_dict())
        for n,m in modules.items():m.training=modes[n]
        for n,p in params.items():p.grad=None if grads[n] is None else grads[n].to(p.device).clone()
        self.steps=state['steps'];torch.set_rng_state(cpu.get_state())
        if cuda is not None:torch.cuda.set_rng_state_all([r.cpu() for r in cuda])
