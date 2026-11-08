"""Continuous GFN density/TB primitives, arXiv:2301.12594v1 §3.5.

Time-layered Euclidean states use Lebesgue measure; origin/sink use Dirac mass.
The caller owns the normalized policy, valid DAG, trajectories and reward measure.
No sampler, convergence guarantee, reward clipping or implicit detach is provided.
"""
import math
import torch
from torch.nn import functional as F


def _finite(x):
    if (not isinstance(x,torch.Tensor) or not x.is_floating_point()
            or x.numel()==0 or not torch.isfinite(x).all()):
        raise ValueError('Require nonempty finite floating tensor')


def _match(x,y):
    _finite(y)
    if x.dtype!=y.dtype or x.device!=y.device or x.shape!=y.shape:
        raise ValueError('Shape, dtype and device must match')


def diagonal_gaussian_log_density(value, mean, log_std):
    """Normalized diagonal Gaussian log density, sum over last event axis.

    No broadcasting or clipping. Arbitrary leading batch/time axes are retained.
    Negative log densities are not required: densities may exceed one.
    """
    _finite(value)
    if value.ndim<1:raise ValueError('Require event axis')
    _match(value,mean);_match(value,log_std)
    scale=log_std.exp()
    if not torch.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Unrepresentable Gaussian scale')
    result=(-.5*((value-mean)/scale).square()-log_std-.5*math.log(2*math.pi)).sum(-1)
    _finite(result)
    return result


def stop_log_probabilities(logits):
    """Return (log p_stop, log p_continue) for finite Bernoulli logits.

    For each optional-stop state, multiply continuous move density by p_continue.
    Forced final stopping uses log p_stop=0, with no learned stop decision there.
    Origin continuation is forced; its transition has no Bernoulli factor.
    """
    _finite(logits)
    return F.logsigmoid(logits),F.logsigmoid(-logits)


def trajectory_balance(log_z, log_forward, log_backward, log_stop, log_reward, mask):
    """Mean squared log-ratio for complete time-layered paths (§3.5).

    [B,T] forward entries already include continuation mass where applicable.
    Backward entries align with those moves; first must be zero (Dirac origin).
    [B] stop mass is separate and appears exactly once. No backward sink factor.
    Bool mask is a nonempty prefix per path. Finite padded values are ignored;
    NaN/inf padding is rejected. Reward is a positive finite density supplied in
    log space; no biomedical validity is implied. Gradients propagate to all
    inputs: caller must detach fixed rewards and sampled states as appropriate.
    """
    _finite(log_forward)
    if log_forward.ndim!=2:raise ValueError('Require [batch,time] densities')
    _match(log_forward,log_backward)
    if (not isinstance(mask,torch.Tensor) or mask.dtype!=torch.bool
            or mask.shape!=log_forward.shape or mask.device!=log_forward.device
            or not mask[:,0].all() or (mask[:,1:] & ~mask[:,:-1]).any()):
        raise ValueError('Require nonempty prefix mask for every path')
    reference=log_forward[:,0]
    _match(reference,log_stop);_match(reference,log_reward)
    _finite(log_z)
    if log_z.ndim!=0 or log_z.dtype!=reference.dtype or log_z.device!=reference.device:
        raise ValueError('Require matching scalar log partition')
    if (log_stop>0).any() or (log_backward[:,0]!=0).any():
        raise ValueError('Stop log mass must be <=0; backward origin log mass must be zero')
    residual=log_z+torch.where(mask,log_forward-log_backward,0).sum(-1)+log_stop-log_reward
    loss=residual.square().mean()
    _finite(residual);_finite(loss)
    return dict(residual=residual,loss=loss)
