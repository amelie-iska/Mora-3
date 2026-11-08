"""GraphCG paper reference: arXiv:2401.17123v1 Eqs 9–12,24.

Linear editing only. Explicit views/conditions keep sampling and RNG ownership
with the trainer. These objectives alone do not establish semantic factors.
See planning/research/primary-graphcg.md for reductions and upstream differences.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def _matrix(x):
    if (not isinstance(x, torch.Tensor) or x.ndim != 2 or min(x.shape) < 1
            or not x.is_floating_point() or not torch.isfinite(x).all()):
        raise ValueError('Require a finite nonempty floating matrix')


def _same(reference, other):
    _matrix(other)
    if (other.shape != reference.shape or other.dtype != reference.dtype
            or other.device != reference.device):
        raise ValueError('Shape, dtype and device must agree')


class LinearConceptEditor(nn.Module):
    """Eq24 fixed basis -> affine -> unit direction -> additive latent edit.

    K<=D is a conservative capacity policy, not a guarantee of independence.
    Caller owns basis randomness. Basis is cloned and persisted as a buffer.
    """
    def __init__(self, basis):
        super().__init__()
        _matrix(basis)
        if basis.shape[0] > basis.shape[1]:
            raise ValueError('Independent-axis budget exceeds latent dimension')
        self.register_buffer('basis', basis.detach().clone())
        width = basis.shape[1]
        self.projection = nn.Linear(width, width, device=basis.device, dtype=basis.dtype)

    def directions(self):
        raw = self.projection(self.basis)
        norms = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
        if not torch.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError('Directions must have finite nonzero norms')
        return raw / norms

    def forward(self, z, indices, alpha):
        _matrix(z)
        if (z.shape[1] != self.basis.shape[1] or z.device != self.basis.device
                or z.dtype != self.basis.dtype):
            raise ValueError('Latents must match editor width, dtype and device')
        if (indices.shape != (z.shape[0],) or indices.dtype != torch.long
                or indices.device != z.device or (indices < 0).any()
                or (indices >= self.basis.shape[0]).any()):
            raise ValueError('Require one valid direction index per latent')
        if (alpha.shape != (z.shape[0],) or alpha.dtype != z.dtype
                or alpha.device != z.device or not torch.isfinite(alpha).all()):
            raise ValueError('Require one finite step size per latent')
        return z + alpha[:, None] * self.directions()[indices]


def symmetric_nce(u_positive, v_positive, u_negative, v_negative):
    """Eq9 sum of both conditional losses, averaged over pairs; raw dot energy.

    Each negative is the same base view edited at the alternate condition.
    No cosine normalization, temperature or half factor is implicit here.
    """
    _matrix(u_positive)
    for x in (v_positive, u_negative, v_negative):
        _same(u_positive, x)
    positive = (u_positive * v_positive).sum(-1)
    negative_u = (u_negative * v_positive).sum(-1)
    negative_v = (v_negative * u_positive).sum(-1)
    scores = torch.stack((positive, negative_u, negative_v))
    if not torch.isfinite(scores).all():
        raise ValueError('Dot product overflow')
    return (2 * F.softplus(-positive) + F.softplus(negative_u)
            + F.softplus(negative_v)).mean()


def graphcg_objective(editor, u, v, indices, alpha, negative_indices,
                      beta, *, coefficients=(1., 0., 0.)):
    """Eq11 with Algorithm1 negatives (both index and step must differ).

    Similarity averages distinct unordered raw direction dots; sparsity averages
    per-direction L1 norms. These explicit reductions differ from upstream code.
    No implicit detach: freeze encoder in the caller for GraphCG's fixed-DGM use.
    """
    if (len(coefficients) != 3 or any(not math.isfinite(c) or c < 0 for c in coefficients)):
        raise ValueError('Require three finite nonnegative coefficients')
    _matrix(u)
    _same(u, v)
    up = editor(u, indices, alpha)
    vp = editor(v, indices, alpha)
    un = editor(u, negative_indices, beta)
    vn = editor(v, negative_indices, beta)
    if (indices == negative_indices).any() or (alpha == beta).any():
        raise ValueError('Negative direction and step must both differ')
    nce = symmetric_nce(up, vp, un, vn)
    d = editor.directions()
    row, col = torch.triu_indices(len(d), len(d), offset=1, device=d.device)
    # One-direction editing is valid; no pairwise regularization exists then.
    similarity = (d[row] * d[col]).sum(-1).mean() if len(row) else d.sum() * 0
    sparsity = d.abs().sum(-1).mean()
    total = coefficients[0]*nce + coefficients[1]*similarity + coefficients[2]*sparsity
    return dict(total=total, nce=nce, similarity=similarity, sparsity=sparsity)


@torch.no_grad()
def direction_diagnostics(directions):
    """Numerical capacity/orthogonality evidence, never semantic validation."""
    _matrix(directions)
    norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    if not torch.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError('Require finite nonzero directions')
    d = directions / norms
    row, col = torch.triu_indices(len(d), len(d), offset=1, device=d.device)
    cosine = (d[row] * d[col]).sum(-1)
    return dict(rank=int(torch.linalg.matrix_rank(d)),
                singular_values=torch.linalg.svdvals(d).cpu().tolist(),
                max_abs_cosine=float(cosine.abs().max()) if len(row) else 0.,
                off_diagonal_rms=float(cosine.square().mean().sqrt()) if len(row) else 0.,
                directions=len(d), latent_width=d.shape[1])
