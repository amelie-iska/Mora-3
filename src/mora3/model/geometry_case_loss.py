"""Equal-active-case free-atom velocity reduction, separate from generic CFM.

Caller-provided masks/required flags are applicability declarations, not admission
proof. Call preflight before constructing paths or drawing noise. No coordinates,
frames, paths, random draws, model calls or task admission are supplied here.
"""
import torch


def active_geometry_cases(free_atoms, required_available):
    """Return active ordered indices; reject empty supervision before path work.

    Each mask covers a complete case atom axis. A free atom contributes all three
    Cartesian coordinates. Missing-required and all-fixed cases are inactive.
    """
    if (type(free_atoms) not in (list, tuple) or not free_atoms
            or type(required_available) not in (list, tuple)
            or len(free_atoms) != len(required_available)
            or any(type(x) is not bool for x in required_available)):
        raise ValueError('ordered nonempty masks and exact required booleans needed')
    for mask in free_atoms:
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                or mask.ndim != 1 or not mask.numel()):
            raise ValueError('complete nonempty boolean atom masks required')
    active = tuple(i for i, (mask, ready) in enumerate(zip(free_atoms, required_available))
                   if ready and bool(mask.any()))
    if not active:
        raise ValueError('no active geometry cases')
    return active


def geometry_case_velocity_loss(predictions, velocities, free_atoms, required_available):
    """Mean_i mean_(free atom,xyz) (prediction - detached velocity)^2.

    No arithmetic touches inactive cases or fixed atoms, including masked NaNs.
    Inactive case arrays are opaque. Active tensors must share dtype/device.
    Nonfinite active values, overflowed residuals and malformed axes refuse.
    """
    active = active_geometry_cases(free_atoms, required_available)
    if (type(predictions) not in (list, tuple) or type(velocities) not in (list, tuple)
            or len(predictions) != len(free_atoms) or len(velocities) != len(free_atoms)):
        raise ValueError('complete ordered case arrays required')
    selected = []
    signature = None
    for i in active:
        p, u, mask = predictions[i], velocities[i], free_atoms[i]
        if (not isinstance(p, torch.Tensor) or not isinstance(u, torch.Tensor)
                or p.shape != (mask.numel(), 3) or u.shape != p.shape
                or not p.is_floating_point() or p.dtype != u.dtype
                or p.device != u.device or p.device != mask.device):
            raise ValueError('matching floating full-atom xyz tensors required')
        current = (p.dtype, p.device)
        if signature is not None and signature != current:
            raise ValueError('active cases must share dtype and device')
        signature = current
        a, b = p[mask], u.detach()[mask]
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            raise ValueError('active free coordinates must be finite')
        selected.append((a, b))
    losses = [(a-b).square().mean() for a,b in selected]
    loss = torch.stack(losses).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError('nonfinite velocity residual loss')
    return loss
