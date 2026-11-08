"""Bound source-native nuclear semantic objective; no corpus admission granted.

Evaluation keeps the differentiable reference loss available for future native
training admission. A source train split alone never establishes that admission.
Masks are decoded only as objective targets and never passed to the network.
"""
from dataclasses import dataclass
import numpy as np
import torch
from mora3.data.public_image_masks import PublicImageMasks
from .color_input import color_tensor
from .color_semantic import ColorNuclearSemantic, nuclear_semantic_objective
from .color_semantic_inference import _require_source_image


@dataclass(frozen=True)
class ColorSemanticTask:
    source: PublicImageMasks
    expected_source_sha256: str

    def __post_init__(self):
        if not isinstance(self.source, PublicImageMasks):
            raise ValueError('typed source image/mask record required')
        if self.source.binding_sha256 != self.expected_source_sha256:
            raise ValueError('source binding mismatch')
        _require_source_image(self.source.image)
        if self.source.encoding != 'semantic_binary':
            raise ValueError('native binary nuclear semantic encoding required')
        color_tensor(self.source.image_array(), alpha_policy='require-opaque')

    def input_pixels(self):
        """Independent native pixels; no masks, IDs or target geometry."""
        return self.source.image_array()

    def require_training(self):
        if self.source.split != 'train':
            raise ValueError('held-out source cannot train')
        raise ValueError('native nuclear-semantic corpus admission unavailable')

    def evaluate(self, model):
        """Differentiable source-bound loss; caller controls modes and gradients."""
        if type(model) is not ColorNuclearSemantic:
            raise ValueError('unconditioned native nuclear semantic model required')
        logits = model(self.input_pixels())
        target = torch.from_numpy(np.array(self.source.target_arrays()[0], copy=True)).to(logits.device)
        return dict(loss=nuclear_semantic_objective(logits, target),
                    source_sha256=self.expected_source_sha256,
                    objective='native-nuclear-semantic-bce-v1',
                    split=self.source.split, training_admission=False)

    def evaluate_conditioned(self, model, *, context_fields=None, expected_image_sha256=None):
        """Source-observed shared graph objective; targets remain loss-only.

        Caller-supplied graphs and desired context are deliberately absent:
        observed source masks cannot supervise hypothetical desired biology.
        Caller retains control of module modes, autograd and optimization.
        """
        import hashlib
        from .color_conditioning import ConditionedColorSemantic
        from .color_semantic_inference import image_condition_graph, selected_image_condition_graph
        if type(model) is not ConditionedColorSemantic:
            raise ValueError('conditioned native nuclear semantic model required')
        pixels = self.input_pixels()
        graph, task = image_condition_graph(self.source.image,
            pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
        if context_fields is not None:
            graph, task = selected_image_condition_graph(self.source.image,
                pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                context_fields=context_fields,
                expected_image_sha256=expected_image_sha256)
        elif expected_image_sha256 is not None:
            raise ValueError('image binding requires selected context fields')
        logits = model(pixels, graph, task)
        target = torch.from_numpy(np.array(self.source.target_arrays()[0], copy=True)).to(logits.device)
        return dict(loss=nuclear_semantic_objective(logits, target),
                    source_sha256=self.expected_source_sha256,
                    objective='shared-source-observed-native-nuclear-semantic-bce-v1',
                    context_fields=None if context_fields is None else tuple(sorted(context_fields)),
                    split=self.source.split, training_admission=False)
