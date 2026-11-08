"""Complete grayscale image-to-mask objective on the exact shared pixel model."""

import torch
from torch.nn import functional as F

from mora3.data.deepbacs_native import DeepBacsNativeSource
from .morphology import MorphologyInstance


def grayscale_integer_label_loss(logits, labels):
    """Native non-overlapping integer labels induce foreground and local affinity."""
    if (
        not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 3
        or not logits.is_floating_point() or not torch.isfinite(logits).all()
        or not isinstance(labels, torch.Tensor) or labels.dtype != torch.long or labels.ndim != 2
        or labels.shape != logits.shape[1:] or not labels.numel() or labels.device != logits.device
        or (labels < 0).any()
    ):
        raise ValueError('Aligned native integer labels and finite full foreground/affinity logits required')
    # All nonborder edges are observed in a non-overlapping integer mask.
    foreground = labels > 0
    right = foreground[:, :-1] & (labels[:, :-1] == labels[:, 1:])
    down = foreground[:-1] & (labels[:-1] == labels[1:])
    foreground_loss = F.binary_cross_entropy_with_logits(logits[0], foreground.to(logits.dtype))
    edge_loss = F.binary_cross_entropy_with_logits(logits[1, :, :-1], right.to(logits.dtype), reduction='sum')
    edge_loss = edge_loss + F.binary_cross_entropy_with_logits(logits[2, :-1], down.to(logits.dtype), reduction='sum')
    return foreground_loss + edge_loss / max(right.numel() + down.numel(), 1)


def native_grayscale_segmentation_objective(model, source, batch, *, split, return_logits=False):
    if (
        type(model) is not MorphologyInstance or type(source) is not DeepBacsNativeSource
        or model.source_envelope != source.envelope or type(return_logits) is not bool
    ):
        raise ValueError('Exact shared grayscale model, source envelope and reporting policy required')
    if split == 'test' and torch.is_grad_enabled():
        raise ValueError('Development-exposed original test is evaluation only')
    model.get_extra_state()
    packets = source.packets(batch, split=split, direction='image_to_mask')
    outputs, losses = [], []
    for packet in packets:
        logits = model(packet.image)
        losses.append(grayscale_integer_label_loss(logits, packet.labels(device=logits.device)))
        if return_logits:
            outputs.append(logits)
    loss = torch.stack(losses).mean()
    return dict(loss=loss, logits=tuple(outputs)) if return_logits else loss
