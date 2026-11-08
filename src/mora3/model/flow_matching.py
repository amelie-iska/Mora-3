"""Gaussian conditional flow reference (Lipman et al., 2210.02747v2).

Operates in continuous transformed data space, not count space. No clipping,
normalization, random sampling, task admission or biological context is implicit.
"""
import math
import torch


def _pair(a, b):
    if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
            or a.ndim != 2 or not a.numel() or a.shape != b.shape
            or not a.is_floating_point() or a.dtype != b.dtype or a.device != b.device):
        raise ValueError('matching nonempty floating batch/feature tensors required')


def gaussian_path(noise, target, time, *, sigma_min=0.01):
    """Eq. 22/23 sample and pathwise derivative; terminal residual sigma*noise.

    Target is supervision only, detached from autograd. The noisy state is an
    intentional training input; it is never a clean conditioning observation.
    """
    _gaussian_inputs(noise, target, time, sigma_min)
    t = time[:, None]
    y = target.detach()
    return (1 - (1 - sigma_min) * t) * noise + t * y, y - (1 - sigma_min) * noise


def _gaussian_inputs(noise, target, time, sigma_min):
    _pair(noise, target)
    if (not isinstance(time, torch.Tensor) or time.shape != (noise.shape[0],)
            or time.device != noise.device or time.dtype != noise.dtype
            or not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any()
            or not isinstance(sigma_min, (int, float)) or not math.isfinite(sigma_min)
            or not 0 <= sigma_min < 1
            or not torch.isfinite(noise).all() or not torch.isfinite(target).all()):
        raise ValueError('finite inputs, batch times in [0,1], sigma in [0,1) required')


def gaussian_conditional_velocity(state, target, time, *, sigma_min=0.01):
    """Eq. 21 target-aware oracle: (y-(1-sigma)*x)/(1-(1-sigma)*t).

    For equation/capacity tests only: requires the clean target, so it must NOT
    serve as a biological context encoder or a target-free learned sampler.
    Targets are detached; state/time derivatives are retained. The sigma=0,
    t=1 singularity is rejected even though the pathwise derivative exists.
    """
    _gaussian_inputs(state, target, time, sigma_min)
    scale = 1 - (1 - sigma_min) * time[:, None]
    if (scale <= 0).any():
        raise ValueError('conditional field requires positive Gaussian scale')
    return (target.detach() - (1 - sigma_min) * state) / scale


def velocity_loss(prediction, velocity, available):
    """Mean over explicitly available coordinates; empty supervision is inactive.

    Masked values are never used in arithmetic, including masked NaNs. Caller
    must supply admitted missingness; zero is never inferred to mean missing.
    """
    _pair(prediction, velocity)
    if (not isinstance(available, torch.Tensor) or available.dtype != torch.bool
            or available.shape != prediction.shape or available.device != prediction.device):
        raise ValueError('exact boolean availability mask required')
    if not available.any():
        return None
    p, u = prediction[available], velocity.detach()[available]
    if not torch.isfinite(p).all() or not torch.isfinite(u).all():
        raise ValueError('available coordinates must be finite')
    return (p - u).square().mean()


@torch.no_grad()
def euler_flow(field, initial, *, steps):
    """Fixed-step Euler reference from t=0 to 1; no hidden target argument.

    Context must be bound by caller in field(state, batch_time). This is an
    approximate ODE solver, not an adaptive or quality-selected sampler.
    """
    _pair(initial, initial)
    if type(steps) is not int or steps < 1 or not torch.isfinite(initial).all():
        raise ValueError('positive integer steps and finite initial state required')
    x = initial.clone()
    for i in range(steps):
        t = x.new_full((x.shape[0],), i / steps)
        v = field(x, t)
        _pair(x, v)
        if not torch.isfinite(v).all():
            raise ValueError('finite velocity required')
        x = x + v / steps
        if not torch.isfinite(x).all():
            raise ValueError('nonfinite integrated state')
    return x
