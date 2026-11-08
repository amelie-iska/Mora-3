"""Full-coordinate transformed-response diagnostics; no heldout admission implied.

Energy score uses Euclidean cell-vector distance and the empirical V-statistic:
E||X-y|| - .5 E||X-X'||, including diagonal pairs. Negative samples remain
unchanged in primary scores; clamp-zero is a separately named sensitivity audit.
"""
import torch


@torch.no_grad()
def response_quality(samples, target):
    if (not isinstance(samples,torch.Tensor) or not isinstance(target,torch.Tensor)
            or samples.ndim!=3 or target.ndim!=2 or not samples.numel()
            or samples.shape[1:]!=target.shape or not samples.is_floating_point()
            or samples.dtype!=target.dtype or samples.device!=target.device
            or not torch.isfinite(samples).all() or not torch.isfinite(target).all()
            or (target<0).any()):
        raise ValueError('finite complete draws/cells/genes and nonnegative transformed targets required')
    # Accumulate pair distances without a draws-squared/cells/genes allocation.
    pair=target.new_zeros(())
    for a in samples:
        for b in samples:
            pair+=torch.linalg.vector_norm(a-b,dim=-1).mean()
    energy=torch.linalg.vector_norm(samples-target,dim=-1).mean()-pair/(2*len(samples)**2)
    return dict(sample_mse=float((samples-target).square().mean()),
        ensemble_mean_mse=float((samples.mean(0)-target).square().mean()),
        energy_score=float(energy),negative_fraction=float((samples<0).double().mean()),
        clamp_zero_sample_mse=float((samples.clamp_min(0)-target).square().mean()),
        coordinates=target.numel(),cells=len(target),genes=target.shape[1],draws=len(samples))
