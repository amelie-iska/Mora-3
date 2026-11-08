"""Opt-in shared source reconstruction interface; scheduling is separate.

The legacy interface registers the residual projection and retains fixed draws.
The explicit new primary policy also owns the real shared profile GFN kernels.
Fixed-draw reconstruction alone does not train those kernels; normalized TB has
its own objective/schedule. Source reconstruction is not therapeutic evidence.
Historical cooperative architectures and checkpoints remain unchanged.
"""
from .configuration_snapshot import configuration_snapshot
import torch
from .goa_cooperative import GOACooperativeModel
from .profile_gfn import ProfileGFNPolicy
from .profile_latent_decoder import ProfileLatentSelfiesDecoder
from .profile_reconstruction import FixedProfileReconstruction


class ReconstructionCooperativeModel(GOACooperativeModel):
    def __init__(self,*args,reconstruction_dimension,reconstruction_horizon,
                 reconstruction_hidden=64,reconstruction_source,**kwargs):
        super().__init__(*args,**kwargs)
        new=getattr(self,'parameterization_policy','legacy')=='mup-cooperative-v1'
        # The selected primary arm owns the actual policy. Legacy fixed-draw
        # reference construction keeps its historical external frozen boundary.
        policy=ProfileGFNPolicy(self.profile,dimension=reconstruction_dimension,
            horizon=reconstruction_horizon,hidden=reconstruction_hidden,
            source_binding=reconstruction_source,**(
                dict(parameterization_policy='mup-profile-gfn-v1') if new else {}))
        if new:self.profile_gfn=policy
        self.reconstruction=ProfileLatentSelfiesDecoder(self.decoder,policy,
                                                        share_decoder=True)
        self.configuration=dict(self.configuration,
            architecture='unpaired-native-GOA-fixed-profile-reconstruction-interface-v1',
            reconstruction=self.reconstruction.get_extra_state(),
            reconstruction_ownership='parent-decoder-profile;branch-projection-only-v1')
        if new:self.configuration.update(profile_gfn=policy.get_extra_state(),
            profile_gfn_ownership='registered-real-shared-profile-policy-v1',
            reconstruction_ownership='parent-decoder-profile;branch-projection-plus-registered-profile-policy-v2')

    def get_extra_state(self):
        state=super().get_extra_state()
        state['reconstruction']=self.reconstruction.get_extra_state()
        if getattr(self,'parameterization_policy','legacy')=='mup-cooperative-v1':
            if (self.profile_gfn.profiles is not self.profile
                    or self.reconstruction._policy!=self.profile_gfn.get_extra_state()):
                raise ValueError('registered profile GFN/shared reconstruction configuration mismatch')
            state['profile_gfn']=self.profile_gfn.get_extra_state()
        return state

    def set_extra_state(self,state):
        if state!=self.get_extra_state():
            raise ValueError('Reconstruction cooperative configuration mismatch')

    def reconstruction_objective(self,batch,*,indices=None):
        if not isinstance(batch,FixedProfileReconstruction):
            raise ValueError('Native fixed source reconstruction batch required')
        if indices is None:
            return batch.objective(self.reconstruction)
        return batch.subset_objective(self.reconstruction,indices)


from .goa_cooperative import GOACooperativeTrainer
from .profile_reconstruction import _batch, _digest as _archive_digest
from .annotation_trainer import _digest


class ReconstructionCooperativeTrainer(GOACooperativeTrainer):
    """Experimental fourteenth objective with fixed, authenticated row draws.

    A fresh optimizer is constructed for this new architecture; historical Adam
    adoption and strict fresh-process recovery require separate validation.
    No draw resampling or cross-source observation joins occur here.
    """
    def __init__(self,model,batch,pools,*,reconstruction_batch,
                 reconstruction_batch_size,**kwargs):
        if (not isinstance(model,ReconstructionCooperativeModel)
                or not isinstance(reconstruction_batch,FixedProfileReconstruction)):
            raise ValueError('Native shared reconstruction model and fixed batch required')
        if type(reconstruction_batch_size) is not int or reconstruction_batch_size<=0:
            raise ValueError('Positive integer reconstruction batch size required')
        decoder_batch=kwargs.get('decoder_batch')
        if decoder_batch is None or _batch(decoder_batch.views)!=reconstruction_batch.binding()['batch_sha256']:
            raise ValueError('Exact ordered decoder/reconstruction source batch required')
        self.reconstruction_batch=reconstruction_batch
        self.reconstruction_batch_size=reconstruction_batch_size
        self._validate_reconstruction(model)
        super().__init__(model,batch,pools,**kwargs)
        self.kinds+=('profile_reconstruction',)
        self._reconstruction_policy=reconstruction_batch.binding()
        self._reconstruction_schedule=(self.kinds,reconstruction_batch_size)
        self.binding=_digest(dict(version=1,goa_binding=self.binding,
            reconstruction=self._reconstruction_policy,batch_size=reconstruction_batch_size,
            schedule=self.kinds,policy='equal-active-objective-fixed-reconstruction-v1',
            optimizer_boundary='fresh-Adam-new-shared-projection-no-historical-adoption'))

    def _validate_reconstruction(self,model):
        fixed=self.reconstruction_batch
        if (_batch(fixed._views)!=fixed.binding()['batch_sha256']
                or model.reconstruction.get_extra_state()!=fixed.binding()['configuration']):
            raise ValueError('Fixed reconstruction source/configuration mismatch')
        fixed.latents()  # Validate every archived row before any objective mutates Adam.

    def _check_policy(self):
        super()._check_policy()
        if (self.reconstruction_batch.binding()!=self._reconstruction_policy
                or _batch(self.decoder_batch.views)!=self._reconstruction_policy['batch_sha256']):
            raise ValueError('Fixed reconstruction cooperative policy mismatch')
        if (self.kinds,self.reconstruction_batch_size)!=self._reconstruction_schedule:
            raise ValueError('Fixed reconstruction schedule mismatch')
        self._validate_reconstruction(self.model)

    def additional_objective(self,kind):
        if kind!='profile_reconstruction':return super().additional_objective(kind)
        rows=self.reconstruction_batch.binding()['rows']
        size=min(self.reconstruction_batch_size,rows)
        start=(self.steps//len(self.kinds))*size
        indices=tuple((start+i)%rows for i in range(size))
        return self.model.reconstruction_objective(self.reconstruction_batch,indices=indices),size


    def state_dict(self):
        """Include the exact fixed draws in every new native checkpoint.

        The archive, heterogeneous module modes and pending gradients are
        detached copies. Frozen policy exports remain external dependencies.
        """
        self._check_policy()
        state=super().state_dict()
        state['reconstruction_archive']=self.reconstruction_batch.state_dict()
        state['modes']={n:m.training for n,m in self.model.named_modules()}
        state['gradients']={n:None if p.grad is None else p.grad.detach().clone()
                            for n,p in self.model.named_parameters()}
        return state

    def load_state_dict(self,state):
        # Require byte-equivalent trusted draws before parent model/Adam/RNG
        # restoration. A matching source binding alone cannot recover latents.
        self._check_policy()
        if (not isinstance(state,dict) or 'reconstruction_archive' not in state
                or _archive_digest(state['reconstruction_archive'])
                    !=_archive_digest(self.reconstruction_batch.state_dict())):
            raise ValueError('Fixed reconstruction checkpoint archive mismatch')
        params=dict(self.model.named_parameters())
        modules=dict(self.model.named_modules())
        modes=state.get('modes');grads=state.get('gradients')
        if (not isinstance(modes,dict) or modes.keys()!=modules.keys()
                or any(type(v) is not bool for v in modes.values())
                or not isinstance(grads,dict) or grads.keys()!=params.keys()):
            raise ValueError('Gradient/mode allocation mismatch')
        for name,p in params.items():
            g=grads[name]
            if g is not None and (not isinstance(g,torch.Tensor)
                    or g.shape!=p.shape or g.dtype!=p.dtype
                    or g.layout!=torch.strided or not torch.isfinite(g).all()):
                raise ValueError('Invalid gradient shape/precision/value')
        # Allocate all restored gradients before parent staged validation/commit.
        restored={n:None if grads[n] is None else grads[n].to(p.device).clone()
                  for n,p in params.items()}
        result=super().load_state_dict(state)
        for n,m in modules.items():m.training=modes[n]
        for n,p in params.items():p.grad=restored[n]
        return result


class GeometryReconstructionCooperativeModel(ReconstructionCooperativeModel):
    """Opt-in geometry interface retaining all fourteen independent objectives.

    Reuses the existing image encoder and TokenGT transforms. Frozen discrete
    predictions condition image logits; they do not supply response supervision.
    This architecture requires a new explicit trainer/adoption boundary before
    cooperative scheduling; historical checkpoints are never migrated implicitly.
    """
    def __init__(self, *args, geometry_id_width, **kwargs):
        from .geometry_pixel_fusion import GeometryPixelFusion
        super().__init__(*args, **kwargs)
        geometry_options = {}
        if kwargs.get('parameterization_policy', 'legacy') == 'mup-cooperative-v1':
            geometry_options = dict(parameterization_policy='mup-geometry-v1',
                                    base_width=self.native.config['mup']['base_width'])
        self.geometry_fusion = GeometryPixelFusion(self.native.image,
            id_width=geometry_id_width, **geometry_options)
        self.configuration = dict(self.configuration,
            architecture='unpaired-native-fourteen-objective-geometry-interface-v1',
            geometry=self.geometry_fusion.get_extra_state(),
            geometry_schedule='interface-only-no-adoption',
            geometry_pairing='none-inferred')

    def get_extra_state(self):
        import copy
        return configuration_snapshot(super().get_extra_state())

    def geometry_objective(self, task, *, request, native,
                           admitted_task_sha256, prediction_asset):
        """Exact admitted source task/request/checkpoint checks precede loss.

        Source masks enter only union/affinity targets. The visible image and
        frozen predicted geometry alone condition the native pixel logits.
        Parameter iteration deduplicates shared modules for a single Adam owner.
        """
        loss, _, _ = self.geometry_fusion.training_objective(task, request, native,
            admitted_task_sha256=admitted_task_sha256,
            prediction_asset=prediction_asset)
        return loss


class GeometryReconstructionCooperativeTrainer(ReconstructionCooperativeTrainer):
    """Opt-in fifteenth objective, independent admitted images, fresh shared Adam.

    Historical model/Adam adoption is never implicit. Fixed reconstruction draws
    are supplied unchanged by the caller. Equal cyclic objective updates are an
    experimental schedule, not biological pairing or quality promotion. The
    checkpoint-bound frozen predictor is an external dependency, not Adam-owned.
    """
    def __init__(self, model, batch, pools, *, geometry_tasks, image_requests,
                 admitted_task_sha256s, prediction_assets, native,
                 geometry_batch_size, **kwargs):
        import copy
        if not isinstance(model, GeometryReconstructionCooperativeModel):
            raise ValueError('shared fourteen-objective geometry model required')
        if type(geometry_batch_size) is not int or geometry_batch_size <= 0:
            raise ValueError('positive integer geometry batch size required')
        self.geometry_tasks = tuple(geometry_tasks)
        self.image_requests = tuple(image_requests)
        self.admitted_task_sha256s = tuple(admitted_task_sha256s)
        self.prediction_assets = tuple(prediction_assets)
        self.geometry_native = native
        self.geometry_batch_size = geometry_batch_size
        self._validate_geometry(model)
        super().__init__(model, batch, pools, **kwargs)
        self.kinds += ('geometry',)
        # Parent archive validation now follows this explicitly extended cycle.
        self._reconstruction_schedule = (self.kinds, self.reconstruction_batch_size)
        self._geometry_policy = copy.deepcopy(self._geometry_configuration())
        self._geometry_groups = copy.deepcopy(self.optimizer.state_dict()['param_groups'])
        self.binding = _digest(dict(version=1, reconstruction_binding=self.binding,
            geometry=self._geometry_policy,
            policy='equal-active-objective-independent-geometry-v1',
            optimizer_boundary='fresh-Adam-new-geometry-no-historical-adoption'))
        self._geometry_binding = self.binding

    def _validate_geometry(self, model):
        count = len(self.geometry_tasks)
        if (not count or any(len(v) != count for v in (self.image_requests,
                self.admitted_task_sha256s, self.prediction_assets))
                or any(not isinstance(a, str) or not a.strip() for a in self.prediction_assets)
                or len(set(self.prediction_assets)) != count):
            raise ValueError('one exact request/digest/distinct prediction asset per geometry image required')
        for task, request, digest, asset in zip(self.geometry_tasks, self.image_requests,
                self.admitted_task_sha256s, self.prediction_assets):
            model.geometry_fusion.validate_training_binding(task, request, self.geometry_native,
                admitted_task_sha256=digest, prediction_asset=asset)

    def _geometry_configuration(self):
        import hashlib
        digest = lambda v: hashlib.sha256(v.canonical_bytes()).hexdigest()
        # Bind actual bytes and all independent pools, axes, masks and sizes,
        # not merely the caller's labels or the parent checkpoint receipt.
        return dict(tasks=[digest(t) for t in self.geometry_tasks],
            requests=[digest(r) for r in self.image_requests],
            admitted=self.admitted_task_sha256s, assets=self.prediction_assets,
            native=self.geometry_native.identity,
            geometry=self.model.geometry_fusion.get_extra_state(),
            geometry_batch_size=self.geometry_batch_size,
            native_sources=self.model.native.binding(self.pools, batch_size=self.batch_size),
            batches={name:getattr(self, name+'_batch').binding() for name in
                ('residue', 'profile', 'decoder', 'tox21', 'nucleotide', 'rfam', 'flow')},
            annotation=self.batch.binding(), goa=self._goa_configuration(),
            sizes={name:getattr(self, name) for name in ('flow_batch_size', 'profile_batch_size',
                'decoder_batch_size', 'tox21_batch_size', 'nucleotide_batch_size',
                'rfam_prediction_batch_size', 'rfam_generation_batch_size', 'goa_batch_size',
                'reconstruction_batch_size')}, sigma_min=self.sigma_min,
            schedule=self.kinds, native_kinds=self.native_kinds,
            runtime=self._execution_policy())

    def _check_policy(self):
        super()._check_policy()
        self._validate_geometry(self.model)
        if (self.binding != self._geometry_binding
                or self._geometry_configuration() != self._geometry_policy
                or self.optimizer.state_dict()['param_groups'] != self._geometry_groups):
            raise ValueError('geometry cooperative source/request/schedule/Adam binding mismatch')

    def additional_objective(self, kind):
        if kind != 'geometry':
            return super().additional_objective(kind)
        size = min(self.geometry_batch_size, len(self.geometry_tasks))
        start = (self.steps // len(self.kinds)) * size
        indices = tuple((start+i) % len(self.geometry_tasks) for i in range(size))
        losses = [self.model.geometry_objective(self.geometry_tasks[i],
            request=self.image_requests[i], native=self.geometry_native,
            admitted_task_sha256=self.admitted_task_sha256s[i],
            prediction_asset=self.prediction_assets[i]) for i in indices]
        return torch.stack(losses).mean(), size

    def load_state_dict(self, state):
        # PyTorch permits distinct checkpoint values for names of one shared
        # tensor, silently taking the last value. Reject ambiguity before the
        # parent staged restore can commit model, Adam, RNG or pending gradients.
        self._check_policy()
        if not isinstance(state, dict) or not isinstance(state.get('model'), dict):
            raise ValueError('geometry checkpoint model mapping required')
        if type(state.get('version')) is not int or state['version'] != 1:
            raise ValueError('geometry checkpoint version must be integer 1')
        # Probe RNG payloads independently before inherited staged restoration.
        # Invalid generator bytes raise RuntimeError in PyTorch; expose the same
        # ValueError boundary as other rejected checkpoint fields.
        try:
            rng = state['cpu_rng']
            if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1:
                raise ValueError('invalid CPU RNG tensor')
            torch.Generator().set_state(rng.cpu())
            cuda = state['cuda_rng']
            if next(self.model.parameters()).is_cuda:
                if not isinstance(cuda, (list, tuple)) or len(cuda) != torch.cuda.device_count():
                    raise ValueError('CUDA RNG allocation mismatch')
                for i, value in enumerate(cuda):
                    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
                        raise ValueError('invalid CUDA RNG tensor')
                    torch.Generator(device=f'cuda:{i}').set_state(value.cpu())
            elif cuda is not None:
                raise ValueError('unexpected CUDA RNG')
        except (RuntimeError, TypeError, KeyError) as exc:
            raise ValueError('invalid geometry checkpoint RNG') from exc
        tensors = state['model']
        seen = {}
        entries = list(self.model.named_parameters(remove_duplicate=False))
        entries += list(self.model.named_buffers(remove_duplicate=False))
        for name, tensor in entries:
            value = tensors.get(name)
            if not isinstance(value, torch.Tensor):
                raise ValueError('geometry checkpoint tensor missing')
            if id(tensor) in seen:
                previous = seen[id(tensor)]
                if (value.shape != previous.shape or value.dtype != previous.dtype
                        or value.layout != previous.layout
                        or not torch.equal(value, previous)):
                    raise ValueError('geometry checkpoint shared tensor alias mismatch')
            else:
                seen[id(tensor)] = value
        return super().load_state_dict(state)
