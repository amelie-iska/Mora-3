"""Opt-in frozen native geometry -> shared context -> pixel logits adaptation.

Inference/gradient bridge only, without training admission. Hard predictions are
recomputed on the supplied artifact/pixels using a trusted frozen checkpoint.
Its discrete geometry is conditioning, never independent biological validation.
"""
import hashlib
import copy
from .mixed_payload_projection import exact_wire
from torch import nn
from mora3.data.segmentation_image import SegmentationImageInput
from mora3.data.segmentation_shared import project_native_geometry
from .geometry_conditioning import GeometryConditioner
from .morphology import MorphologyInstance
from .segmentation_inference import NativeRasterSegmentation


class GeometryPixelFusion(nn.Module):
    def __init__(self, image, *, id_width,parameterization_policy='legacy',base_width=None):
        super().__init__()
        if not isinstance(image, MorphologyInstance):
            raise ValueError('native instance pixel model required')
        if parameterization_policy not in ('legacy','mup-geometry-v1'):
            raise ValueError('known explicit geometry fusion parameterization required')
        if parameterization_policy!='legacy' and getattr(image,'parameterization_policy','legacy')!='mup-image-v1':
            raise ValueError('explicit applied image owner required for new geometry fusion')
        self.image = image
        self.geometry = GeometryConditioner(image.backbone,id_width=id_width,
            parameterization_policy=parameterization_policy,base_width=base_width)
        self.config = dict(version=1, architecture='native-additive-geometry-pixel-v1',
                           geometry=self.geometry.config.copy(),
                           geometry_origin='frozen-native-checkpoint',
                           training_admission=False)
        self._mup_configuration=None
        if parameterization_policy!='legacy':
            self.config.update(version=2,image=image.get_extra_state(),
                parameterization_policy=parameterization_policy)
            self._mup_configuration=copy.deepcopy(self.config)

    def get_extra_state(self):
        if self._mup_configuration is not None:
            if (not exact_wire(self.config,self._mup_configuration)
                    or not exact_wire(self.config['geometry'],self.geometry.get_extra_state())
                    or not exact_wire(self.config['image'],self.image.get_extra_state())):
                raise ValueError('geometry fusion/image configuration drift')
            return copy.deepcopy(self.config)
        return self.config.copy()

    def set_extra_state(self, state):
        if state != self.config:
            raise ValueError('geometry pixel fusion configuration mismatch')

    def forward(self, image, request, native, *, prediction_asset):
        if not isinstance(image, SegmentationImageInput):
            raise ValueError('image-only inference input required')
        visible = [a for a in request.artifacts
                   if a.kind == 'image' and a.role in ('condition', 'desired')]
        if visible != [image.artifact]:
            raise ValueError('exactly one bound visible image required')
        if (not isinstance(native, NativeRasterSegmentation)
                or request.head_checkpoint != native.identity
                or dict(request.route()).get(image.artifact.artifact_id) != 'native_segmentation'):
            raise ValueError('matching available native checkpoint required')
        key = image.artifact.artifact_id
        predictions = native.infer(request, {key: image},
            prediction_assets={key: prediction_asset},
            expected_pixel_sha256={key: hashlib.sha256(image.pixels).hexdigest()})
        projection = project_native_geometry(request, predictions,
            provenance=image.input_graph().provenance,
            requested_targets=('refined_segmentation',))
        context = self.geometry(projection['input_graph'], projection['task'])
        logits = self.image._logits(image, external_context=context)
        # Caller can persist the actual masks at the declared immutable assets.
        projection['native_predictions'] = predictions
        return logits, projection

    def training_objective(self, task, request, native, *, admitted_task_sha256,
                           prediction_asset):
        """Explicit source-train diagnostic binding; caller owns rights admission.

        The complete admitted task digest binds targets as well as pixels. Only
        source-observed condition images may enter this objective. Frozen native
        geometry is recomputed from pixels; observed mask topology is used solely
        by the existing union/affinity losses. This does not admit a new source or
        provide paired response labels. Held-out tasks remain inference-only.
        """
        from .morphology import foreground_objective
        from .instance_affinity import affinity_objective
        image = self.validate_training_binding(task, request, native,
            admitted_task_sha256=admitted_task_sha256, prediction_asset=prediction_asset)
        logits, audit = self(image, request, native, prediction_asset=prediction_asset)
        labels = task.instances(device=logits.device)
        loss = (foreground_objective(logits[0], labels)
                + affinity_objective(logits[1:], labels))
        audit['training_binding'] = dict(task_sha256=admitted_task_sha256,
            native_checkpoint=native.identity, split=task.split,
            objective='union-bce+local-affinity-bce', response_supervision=False)
        return loss, logits, audit

    def validate_training_binding(self, task, request, native, *,
                                  admitted_task_sha256, prediction_asset):
        """Validate without inference, RNG draws, gradients or optimizer changes."""
        from mora3.data.raster_morphology import RasterMorphologyTask
        from mora3.data.cooperative_segmentation import SegmentationRequest
        if (not isinstance(request, SegmentationRequest)
                or not isinstance(native, NativeRasterSegmentation)
                or request.head_checkpoint != native.identity
                or not isinstance(prediction_asset, str) or not prediction_asset.strip()):
            raise ValueError('matching native request/checkpoint and prediction asset required')
        if (not isinstance(task, RasterMorphologyTask)
                or task.direction != 'image_to_mask'
                or hashlib.sha256(task.canonical_bytes()).hexdigest() != admitted_task_sha256):
            raise ValueError('exact admitted source image_to_mask task required')
        task.require_training()
        visible = [a for a in request.artifacts
                   if a.kind == 'image' and a.role in ('condition', 'desired')]
        if len(visible) != 1:
            raise ValueError('sole observed condition image required')
        a = visible[0]
        if (a.evidence != 'source_observed' or a.role != 'condition'
                or dict(request.route()).get(a.artifact_id) != 'native_segmentation'
                or request.object_type != 'whole_cell' or a.channels != ('brightfield',)
                or a.source_id != task.source + ':' + task.revision
                or a.parent_id != task.parent_group
                or a.asset.rsplit('#', 1)[-1] != task.source_id
                or a.spatial_shape != (task.height, task.width)
                or a.pixel_size_um != task.pixel_size_um
                or a.reference_frame != 'source_pixel_yx'
                or dict(a.context) != dict(species=task.species, strain=task.strain,
                                          perturbation=None, expression=None)):
            raise ValueError('observed training image source/context binding mismatch')
        return SegmentationImageInput(a, task.image_dtype, bytes.fromhex(task.image_hex))


class GeometryPixelTrainer:
    """One admitted image diagnostic, with immutable full-state continuation.

    Native masks are recomputed, never differentiated. Snapshot binding includes
    the full request and admitted task, frozen checkpoint, model configuration,
    optimizer and execution dtype/device. It grants no new source eligibility.
    """
    def __init__(self, model, task, request, native, *, admitted_task_sha256,
                 prediction_asset, learning_rate=.001):
        import math
        import torch
        if (not isinstance(model, GeometryPixelFusion)
                or not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError('fusion model and positive finite learning rate required')
        if getattr(model.image,'parameterization_policy','legacy')!='legacy':
            raise ValueError('new geometry owner requires complete MuAdamW family; legacy Adam trainer unavailable')
        task.require_training()
        if hashlib.sha256(task.canonical_bytes()).hexdigest() != admitted_task_sha256:
            raise ValueError('admitted task digest mismatch')
        self.model, self.task, self.request, self.native = model, task, request, native
        self.admitted_task_sha256 = admitted_task_sha256
        self.prediction_asset = prediction_asset
        parameter = next(model.parameters())
        self.binding = dict(version=1, task_sha256=admitted_task_sha256,
            request_sha256=hashlib.sha256(request.canonical_bytes()).hexdigest(),
            native_checkpoint=native.identity, config=model.config.copy(),
            image_config=model.image.get_extra_state(),
            device=str(parameter.device), dtype=str(parameter.dtype),
            optimizer='Adam', learning_rate=learning_rate,
            deterministic=torch.are_deterministic_algorithms_enabled(),
            objective='union-bce+local-affinity-bce', schedule='single-admitted-image-v1')
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.steps = 0

    def step(self):
        import torch
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss, _, _ = self.model.training_objective(self.task, self.request, self.native,
            admitted_task_sha256=self.admitted_task_sha256,
            prediction_asset=self.prediction_asset)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite fusion objective')
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        return float(loss.detach())

    def snapshot(self):
        import copy
        import torch
        device = next(self.model.parameters()).device
        return copy.deepcopy(dict(binding=self.binding, steps=self.steps,
            model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device)
            if device.type == 'cuda' else None))

    def restore(self, state):
        import copy
        import torch
        if state['binding'] != self.binding:
            raise ValueError('fusion trainer data/request/configuration mismatch')
        if type(state['steps']) is not int or state['steps'] < 0:
            raise ValueError('invalid fusion trainer step')
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        self.steps = state['steps']
        torch.set_rng_state(state['cpu_rng'].cpu())
        if state['cuda_rng'] is not None:
            torch.cuda.set_rng_state(state['cuda_rng'].cpu(), next(self.model.parameters()).device)
