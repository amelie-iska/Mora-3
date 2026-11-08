"""Observed-coordinate MSE for native LINCS axes; no normalization or imputation.

Endpoint-local loss only: CP and L1000 have distinct source scales and are never
pooled into an implicit common assay. Missing targets contribute no gradients.
"""
import torch


def native_profile_loss(prediction, target, observed_mask, *, modality):
    n = {'CellPainting': 1783, 'L1000': 978}.get(modality)
    if n is None or any(not isinstance(x, torch.Tensor) for x in (prediction,target,observed_mask)):
        raise ValueError('native modality and tensor arrays required')
    if prediction.shape != (n,) or target.shape != (n,) or observed_mask.shape != (n,):
        raise ValueError('complete native feature axis required')
    if (not prediction.is_floating_point() or target.dtype != prediction.dtype
            or observed_mask.dtype != torch.bool or target.device != prediction.device
            or observed_mask.device != prediction.device):
        raise ValueError('matching floating precision/device and boolean observation mask required')
    if not observed_mask.any() or not torch.isfinite(prediction).all() or not torch.isfinite(target[observed_mask]).all():
        raise ValueError('finite prediction and nonempty finite observed targets required')
    return (prediction[observed_mask] - target[observed_mask]).square().mean()
