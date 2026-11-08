"""Explicit unpaired multitask reference sharing native TokenGT parameters.

Round-robin objectives, equal updates per nonempty source pool; no inferred
cross-source correspondence or final calibrated mixture. Existing native losses
and admission/target-exclusion boundaries are retained. No discrete mask gradient.
"""
import hashlib
import copy
from .mixed_payload_projection import exact_wire
from collections.abc import Mapping
import torch
from torch import nn
from .morphology import MorphologyInstance, foreground_objective
from .morphology_training import targets
from .instance_affinity import affinity_objective
from .sciplex import SciPlexSharedModel

KINDS = ('image', 'response', 'intervention')


def objective_schedule(pools, *, step, batch_size):
    """Stateless cyclic subsets; absent modalities cause no lookup or update.

    Step counts optimizer updates. Each active pool advances its own cyclic
    batch once per round, preserving all examples, with explicit wraparound.
    batch_size is a positive integer or a complete KINDS-to-positive-size map;
    per-modality sizes change example exposure, not objective update frequency.
    """
    if isinstance(batch_size, Mapping):
        if (set(batch_size) != set(KINDS)
                or any(type(v) is not int or v <= 0 for v in batch_size.values())):
            raise ValueError('positive batch size for each native objective required')
    elif type(batch_size) is not int or batch_size <= 0:
        raise ValueError('positive integer batch size required')
    if (type(step) is not int or step < 0 or set(pools) - set(KINDS)):
        raise ValueError('valid step, batch size and native objective pools required')
    active = [kind for kind in KINDS if len(pools.get(kind, ())) > 0]
    if not active:
        return None
    kind = active[step % len(active)]
    pool = pools[kind]
    size = batch_size[kind] if isinstance(batch_size, Mapping) else batch_size
    start = (step // len(active)) * size
    return kind, tuple(pool[(start + j) % len(pool)] for j in range(min(size, len(pool))))


class CooperativeModel(nn.Module):
    """Experimental shared two-layer backbone, native image/array projections.

    A one-image-node identifier [1] is padded to [1,0], retaining exact rank and
    orthogonality. Response graphs retain two-node identity IDs and directed edge.
    This changes the image branch depth/identifier projection relative to its
    standalone reference; checkpoints/defaults are never silently converted.
    """
    def __init__(self, axis_ids, *, width=16,parameterization_policy='legacy',
                 base_width=None,readout_mult=1.0,attn_mult=1.0,encoder_var=1.0,
                 grayscale_source_envelope=None):
        super().__init__()
        if parameterization_policy not in ('legacy','mup-cooperative-v1'):
            raise ValueError('explicit known cooperative parameterization policy required')
        if parameterization_policy=='legacy' and (base_width is not None or readout_mult!=1 or attn_mult!=1 or encoder_var!=1):
            raise ValueError('legacy cooperative model has no muP declaration')
        if grayscale_source_envelope is not None:
            from .raster_source_envelope import RasterSourceEnvelope
            if type(grayscale_source_envelope) is not RasterSourceEnvelope or parameterization_policy == 'legacy':
                raise ValueError('Explicit opt-in grayscale source envelope required')
            grayscale_source_envelope.binding()
        self.parameterization_policy=parameterization_policy
        self._parameterization_declaration=parameterization_policy
        options={} if parameterization_policy=='legacy' else dict(
            scaling_policy='mup-v1',initialization_policy='mup-transformer-v1',
            parameterization_policy='mup-sciplex-v1',base_width=base_width,
            readout_mult=readout_mult,attn_mult=attn_mult,encoder_var=encoder_var)
        self.response = SciPlexSharedModel(axis_ids, width=width,**options)
        self.image = MorphologyInstance(width,normalization='image-minmax',**(
            {} if parameterization_policy=='legacy' else dict(parameterization_policy='mup-image-v1',
                base_width=base_width,output_mult=readout_mult,attn_mult=attn_mult,encoder_var=encoder_var,
                backbone=self.response.backbone, source_envelope=grayscale_source_envelope)))
        self.image.backbone = self.response.backbone
        self.config = dict(version=1, architecture='native-shared-tokengt-v1',
                           width=width, axis_ids=tuple(axis_ids), attention='vanilla',
                           image_objective='union-bce+local-affinity-bce',
                           response=self.response.config.copy(),
                           identifier_policy='exact-one-node-zero-pad-to-two',
                           shared_layers=2, pairing='none-inferred')
        self._mup_response_configuration=None
        if parameterization_policy!='legacy':
            self.config.update(version=3,image=self.image.get_extra_state(),mup=dict(schema='cooperative-native-parameterization-v2',
                policy=parameterization_policy,base_width=base_width,output_mult=float(readout_mult),
                attn_mult=float(attn_mult),encoder_var=float(encoder_var),
                image_operator_qualification='applied-mup-image-v1'))
            self._mup_response_configuration=copy.deepcopy(self.response.get_extra_state())
            self._mup_configuration=copy.deepcopy(self.config['mup'])

    def validate_parameterization_configuration(self):
        if self.parameterization_policy!=self._parameterization_declaration:
            raise ValueError('cooperative parameterization policy drift')
        if self._mup_response_configuration is None:
            raise ValueError('explicit cooperative native parameterization required')
        self.response.validate_configuration()
        if (not exact_wire(self.response.get_extra_state(),self._mup_response_configuration)
                or not exact_wire(self.config.get('response'),self._mup_response_configuration)
                or not exact_wire(self.config.get('mup'),self._mup_configuration)
                or not exact_wire(self.config.get('image'),self.image.get_extra_state())
                or self.image.backbone is not self.response.backbone):
            raise ValueError('cooperative native/alias/readout configuration drift')

    def get_extra_state(self):
        if self._mup_response_configuration is not None:
            self.validate_parameterization_configuration()
            return copy.deepcopy(self.config)
        return self.config.copy()

    def set_extra_state(self, state):
        if state != self.config:
            raise ValueError('cooperative configuration mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if self._mup_response_configuration is not None:
            # Preflight all descendant policy sidecars before any shared tensor copy.
            current=self.state_dict()
            for key,value in current.items():
                if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+key),value):
                    raise ValueError('cooperative checkpoint child configuration/policy mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def binding(self, pools, *, batch_size):
        objective_schedule(pools, step=0, batch_size=batch_size)
        return dict(config=self.config.copy(), schedule='active-native-cyclic-v1',
                    batch_size=dict(batch_size) if isinstance(batch_size, Mapping) else batch_size,
                    pools={kind: [hashlib.sha256(t.canonical_bytes()).hexdigest()
                                  for t in pools.get(kind, ())] for kind in KINDS})

    def objective(self, kind, tasks):
        """Selected native objective only; no zero-filled absent targets."""
        if kind not in KINDS or not tasks:
            raise ValueError('nonempty selected native objective required')
        if kind != 'image':
            if any(t.input_metadata()['task'] != kind for t in tasks):
                raise ValueError('selected objective/task mask mismatch')
            return self.response.loss(tasks)
        terms = []
        for task in tasks:
            from mora3.data.public_image_admission import PublicImageAdmission
            if isinstance(task, PublicImageAdmission):
                task.require_training()
                z = self.image(task.native_input())
                labels = task.instances(device=z.device)
            else:
                z = self.image(task, for_training=True)
                labels = targets(task, z.device)
            terms.append(foreground_objective(z[0], labels)
                         + affinity_objective(z[1:], labels))
        return torch.stack(terms).mean()


class GeometryCooperativeModel(CooperativeModel):
    """Joint native heads plus opt-in frozen-geometry image conditioning.

    Enabled/disabled modes own identical parameters. Disabled mode still executes
    the bound frozen bridge for admission checks, then uses native pixel loss;
    it is a quality control, not a compute-matched speed baseline. No source join.
    """
    def __init__(self, axis_ids, *, width=16, id_width=1024, geometry_enabled=True,
                 parameterization_policy='legacy',base_width=None,readout_mult=1.0,attn_mult=1.0,encoder_var=1.0,
                 grayscale_source_envelope=None):
        from .geometry_pixel_fusion import GeometryPixelFusion
        if type(geometry_enabled) is not bool:
            raise ValueError('explicit boolean geometry mode required')
        super().__init__(axis_ids,width=width,parameterization_policy=parameterization_policy,
            base_width=base_width,readout_mult=readout_mult,attn_mult=attn_mult,encoder_var=encoder_var,
            grayscale_source_envelope=grayscale_source_envelope)
        self.fusion = GeometryPixelFusion(self.image,id_width=id_width,**(
            {} if parameterization_policy=='legacy' else dict(parameterization_policy='mup-geometry-v1',base_width=base_width)))
        self.config = dict(self.config, architecture='joint-native-geometry-v1',
                           geometry=self.fusion.config.copy(),
                           geometry_enabled=geometry_enabled)

    def geometry_objective(self, task, **binding):
        loss, _, _ = self.fusion.training_objective(task, **binding)
        if self.config['geometry_enabled']:
            return loss
        return self.objective('image', [task])

    def geometry_binding(self, pools, *, batch_size, image_requests, native):
        """Bind ordered image requests/checkpoint in addition to all task bytes.

        Per-task source/context and admission checks remain in the objective.
        This receipt does not confer data rights or imply paired supervision.
        """
        from .segmentation_inference import NativeRasterSegmentation
        images = pools.get('image', ())
        if (len(images) != len(image_requests)
                or not isinstance(native, NativeRasterSegmentation)
                or any(r.head_checkpoint != native.identity for r in image_requests)):
            raise ValueError('one matching native request per image task required')
        result = self.binding(pools, batch_size=batch_size)
        parameter = next(self.parameters())
        result.update(native_checkpoint=native.identity,
            image_requests=[hashlib.sha256(r.canonical_bytes()).hexdigest()
                            for r in image_requests],
            device=str(parameter.device), dtype=str(parameter.dtype),
            deterministic=torch.are_deterministic_algorithms_enabled())
        return result


class GeometryCooperativeTrainer:
    """Bound cyclic native tasks with all-head Adam/RNG continuation.

    Pools remain separate source observations; equal objective updates are an
    explicit experimental schedule, not a calibrated biological sampling mix.
    Frozen predictions are recomputed and never used as observed supervision.
    """
    def __init__(self, model, pools, *, image_requests, native, batch_size,
                 prediction_asset, learning_rate=.001):
        import copy
        import math
        if (not isinstance(model, GeometryCooperativeModel)
                or not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError('joint geometry model and positive finite learning rate required')
        if model.parameterization_policy!='legacy':
            raise ValueError('new native parameterization requires complete cooperative MuAdamW family; legacy Adam trainer unavailable')
        self.pools = {k: tuple(v) for k, v in pools.items()}
        if objective_schedule(self.pools, step=0, batch_size=batch_size) is None:
            raise ValueError('at least one admitted objective required')
        for kind, tasks in self.pools.items():
            for task in tasks:
                if kind == 'image':
                    task.require_training()
                else:
                    from mora3.data.sciplex_admission import ObservationalSciPlexTask
                    if not isinstance(task, ObservationalSciPlexTask):
                        raise ValueError('admitted observational response task required')
                if kind != 'image' and task.input_metadata()['task'] != kind:
                    raise ValueError('selected objective/task mask mismatch')
        self.model, self.native = model, native
        self.requests = tuple(image_requests)
        self.batch_size = dict(batch_size) if isinstance(batch_size, Mapping) else batch_size
        self.prediction_asset = prediction_asset
        self.binding = copy.deepcopy(model.geometry_binding(self.pools,
            batch_size=batch_size, image_requests=self.requests, native=native))
        self.binding.update(trainer='joint-geometry-cyclic-v1', optimizer='Adam',
                            learning_rate=learning_rate)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.steps = 0

    def step(self):
        kind, tasks = objective_schedule(self.pools, step=self.steps,
                                         batch_size=self.batch_size)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if kind == 'image':
            images = self.pools['image']
            terms = []
            for task in tasks:
                index = next(i for i,t in enumerate(images) if t is task)
                terms.append(self.model.geometry_objective(task,
                    request=self.requests[index], native=self.native,
                    admitted_task_sha256=self.binding['pools']['image'][index],
                    prediction_asset=self.prediction_asset))
            loss = torch.stack(terms).mean()
        else:
            loss = self.model.objective(kind, tasks)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite joint objective')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in self.model.parameters()):
            raise ValueError('nonfinite joint gradient')
        self.optimizer.step()
        self.steps += 1
        return dict(step=self.steps, kind=kind, examples=len(tasks), loss=float(loss.detach()))

    def snapshot(self):
        import copy
        device = next(self.model.parameters()).device
        return copy.deepcopy(dict(binding=self.binding, steps=self.steps,
            model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device)
            if device.type == 'cuda' else None))

    def restore(self, state):
        """Validate detached candidates before committing model/Adam/RNG state.

        Historical snapshot layout is retained. Modes and pending gradients are
        not represented by this format and are a separate recovery boundary.
        """
        import copy
        from .rfam_family_training import validate_adam_state
        try:
            if (not isinstance(state, dict) or set(state) != {
                    'binding', 'steps', 'model', 'optimizer', 'cpu_rng', 'cuda_rng'}
                    or state['binding'] != self.binding):
                raise ValueError('joint trainer data/request/configuration mismatch')
            if type(state['steps']) is not int or state['steps'] < 0:
                raise ValueError('invalid joint trainer step')
            expected = self.optimizer.state_dict()['param_groups']
            groups = state['optimizer']['param_groups']
            if len(groups) != len(expected) or any(a != b for a,b in zip(groups,expected)):
                raise ValueError('joint Adam configuration/allocation mismatch')
            validate_adam_state(self.model, state)
            current = self.model.state_dict()
            if set(current) != set(state['model']):
                raise ValueError('joint model allocation mismatch')
            for key, value in current.items():
                candidate = state['model'][key]
                if isinstance(value, torch.Tensor) and (
                        not isinstance(candidate, torch.Tensor)
                        or candidate.shape != value.shape or candidate.dtype != value.dtype):
                    raise ValueError('joint model tensor mismatch')
            staged_model, staged_optimizer = copy.deepcopy((self.model, self.optimizer))
            staged_model.load_state_dict(copy.deepcopy(state['model']))
            staged_optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
            rng = state['cpu_rng'].cpu().clone()
            torch.Generator().set_state(rng)
            device = next(self.model.parameters()).device
            cuda = state['cuda_rng']
            if device.type == 'cuda':
                if cuda is None:
                    raise ValueError('missing joint CUDA RNG')
                cuda = cuda.cpu().clone()
                torch.Generator(device=device).set_state(cuda)
            elif cuda is not None:
                raise ValueError('unexpected joint CUDA RNG')
        except (KeyError, TypeError, RuntimeError, AttributeError, IndexError) as exc:
            raise ValueError('malformed joint trainer snapshot') from exc
        self.model.load_state_dict(staged_model.state_dict())
        self.optimizer.load_state_dict(staged_optimizer.state_dict())
        self.steps = state['steps']
        torch.set_rng_state(rng)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, device)


class StrictGeometryCooperativeTrainer(GeometryCooperativeTrainer):
    """Opt-in version 2 state; never implicitly adopts legacy snapshots.

    Binds current ordered sources, requests, numerical runtime and output asset.
    The frozen native predictor remains an external source-bound dependency.
    """
    def __init__(self, *args, **kwargs):
        import copy
        super().__init__(*args, **kwargs)
        self._strict_binding = copy.deepcopy(self.binding)
        self._strict_runtime = self._runtime()
        self._strict_asset = self.prediction_asset
        self._strict_groups = copy.deepcopy(self.optimizer.state_dict()['param_groups'])

    def _runtime(self):
        return dict(threads=torch.get_num_threads(),
            deterministic=torch.are_deterministic_algorithms_enabled(),
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32)

    def _check(self):
        current = self.model.geometry_binding(self.pools, batch_size=self.batch_size,
            image_requests=self.requests, native=self.native)
        current.update(trainer='joint-geometry-cyclic-v1', optimizer='Adam',
                       learning_rate=self._strict_binding['learning_rate'])
        if (current != self._strict_binding or self.binding != self._strict_binding
                or self._runtime() != self._strict_runtime
                or self.prediction_asset != self._strict_asset
                or self.optimizer.state_dict()['param_groups'] != self._strict_groups):
            raise ValueError('strict geometry live source/request/runtime binding changed')

    def step(self):
        self._check()
        return super().step()

    def snapshot(self):
        import copy
        self._check()
        state = super().snapshot()
        state.update(version=2, runtime=copy.deepcopy(self._strict_runtime),
            prediction_asset=self._strict_asset,
            modes={n:m.training for n,m in self.model.named_modules()},
            gradients={n:None if p.grad is None else p.grad.detach().clone()
                       for n,p in self.model.named_parameters()})
        return state

    def restore(self, state):
        """Allocate validated gradients before inherited transactional commit."""
        self._check()
        extra = {'version', 'runtime', 'prediction_asset', 'modes', 'gradients'}
        base = {'binding', 'steps', 'model', 'optimizer', 'cpu_rng', 'cuda_rng'}
        if (not isinstance(state, dict) or set(state) != base | extra
                or type(state['version']) is not int or state['version'] != 2
                or state['runtime'] != self._strict_runtime
                or state['prediction_asset'] != self._strict_asset):
            raise ValueError('strict geometry snapshot version/runtime mismatch')
        modules = dict(self.model.named_modules())
        params = dict(self.model.named_parameters())
        modes, gradients = state['modes'], state['gradients']
        if (not isinstance(modes, dict) or modes.keys() != modules.keys()
                or any(type(v) is not bool for v in modes.values())
                or not isinstance(gradients, dict) or gradients.keys() != params.keys()):
            raise ValueError('strict geometry modes/gradient allocation mismatch')
        staged = {}
        for name, p in params.items():
            g = gradients[name]
            if g is not None and (not isinstance(g, torch.Tensor)
                    or g.layout != torch.strided or g.shape != p.shape
                    or g.dtype != p.dtype or not torch.isfinite(g).all()):
                raise ValueError('strict geometry invalid pending gradient')
            staged[name] = None if g is None else g.detach().to(p.device).clone()
        # Parent validates model/Adam/RNG in detached candidates before mutation.
        super().restore({k:state[k] for k in base})
        for name, m in modules.items():
            m.training = modes[name]
        for name, p in params.items():
            p.grad = staged[name]
