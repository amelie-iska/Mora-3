"""Complete ordered overlapping-mask evaluation; no corpus training admission.

Reuses the mask-only bipartite BCE adaptation, preserving source overlap/order.
Only source-observed visible-image context enters predictors; no caller graphs,
mask-derived geometry or desired biological context can supervise these targets.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import torch
from mora3.data.public_image_masks import PublicImageMasks
from .color_input import color_tensor
from .color_conditioning import ConditionedColorInstance
from .color_semantic_inference import (
    _require_source_image, image_condition_graph, selected_image_condition_graph)
from .instance_affinity import overlapping_set_objective


@dataclass(frozen=True)
class ColorInstanceTask:
    source: PublicImageMasks
    expected_source_sha256: str

    def __post_init__(self):
        if not isinstance(self.source, PublicImageMasks):
            raise ValueError('typed source image/mask record required')
        if self.source.binding_sha256 != self.expected_source_sha256:
            raise ValueError('source binding mismatch')
        _require_source_image(self.source.image)
        if self.source.encoding != 'ordered_binary_instances':
            raise ValueError('native ordered binary instance encoding required')

    def input_pixels(self):
        """Independent source pixels; masks and target measurements excluded."""
        return self.source.image_array()

    def require_training(self):
        if self.source.split != 'train':
            raise ValueError('held-out source cannot train')
        raise ValueError('native ordered-instance corpus admission unavailable')

    def resource_preflight(self, *, slots, element_bytes, max_output_bytes,
                           slot_policy='bounded-max64-v1', pixel_block_size=None):
        """Complete source counts only; no pixel/target decoding or allocation."""
        from .instance_resources import instance_resource_preflight
        height, width = self.source.image.spatial_shape
        report = instance_resource_preflight(instances=len(self.source.masks_encoded),
            slots=slots, height=height, width=width, element_bytes=element_bytes,
            max_output_bytes=max_output_bytes, slot_policy=slot_policy,
            **({} if pixel_block_size is None else dict(pixel_block_size=pixel_block_size)))
        return dict(report, source_sha256=self.expected_source_sha256)

    def evaluate_conditioned(self, model, *, context_fields=None,
                             expected_image_sha256=None, pixel_block_size=None):
        """Differentiable complete-source loss; caller owns modes and autograd.

        Reject insufficient slots before neural execution or mask decoding.
        Optional pixel blocking affects only detached assignment-cost reduction;
        complete logits, targets and differentiable BCE remain resident.
        The serialized head slot policy also selects the assignment capacity.
        Output-byte refusal precedes decoding; it does not bound peak memory.
        No truncation, semantic union, optimization or checkpoint changes.
        """
        if pixel_block_size is not None and (type(pixel_block_size) is not int
                                             or pixel_block_size < 1):
            raise ValueError('pixel block size must be a positive integer')
        if type(model) is not ConditionedColorInstance:
            raise ValueError('conditioned native color instance model required')
        n = len(self.source.masks_encoded)
        if n > model.color.head.slots:
            raise ValueError('complete source instance set exceeds slot capacity')
        head = model.color.head
        height, width = self.source.image.spatial_shape
        element_bytes = head.projection.weight.element_size()
        if head.slots * height * width * element_bytes > head.max_output_bytes:
            raise ValueError('instance output-byte policy exceeded')
        # The serialized head is the explicit capacity opt-in. Keep the legacy
        # loss call unchanged and never infer capacity from source target count.
        loss_policy = (dict(slot_policy=head.slot_policy)
                       if head.slot_policy == 'extended-max512-v2' else {})
        pixels = self.input_pixels()
        color_tensor(pixels, alpha_policy=model.color.alpha_policy,
                     input_policy=model.color.input_policy)
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        if context_fields is None:
            if expected_image_sha256 is not None:
                raise ValueError('image binding requires selected context fields')
            graph, task = image_condition_graph(self.source.image, pixel_sha256=digest)
        else:
            graph, task = selected_image_condition_graph(self.source.image,
                pixel_sha256=digest, context_fields=context_fields,
                expected_image_sha256=expected_image_sha256)
        logits = model(pixels, graph, task)
        masks = self.source.target_arrays()
        target = torch.from_numpy(np.stack(masks) if masks else
            np.empty((0, *pixels.shape[:2]), dtype=np.bool_)).to(logits.device)
        return dict(loss=overlapping_set_objective(
                        logits, target, pixel_block_size=pixel_block_size,
                        **loss_policy),
                    pixel_block_size=pixel_block_size,
                    source_sha256=self.expected_source_sha256,
                    objective='shared-source-observed-ordered-instance-set-bce-v1',
                    instances=n, slots=model.color.head.slots,
                    context_fields=None if context_fields is None else tuple(sorted(context_fields)),
                    split=self.source.split, training_admission=False)
