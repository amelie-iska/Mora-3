"""Dense finite tropical reference; arXiv:2505.17190v1 §3 core equations.

The explicitly named code wrapper follows e3c12f3e's symmetric default,
not the paper's differing valuation/projection conventions. No compiled backend.
Dense intermediates cost O(batch * heads * queries * keys * head_width).
Extrema use first-index subgradients; ties are nondifferentiable mathematically.
"""
import torch
from torch import nn


def maxplus(a, b):
    """(..., M, R) tropical (..., R, N), with broadcast batch dimensions."""
    if (a.ndim < 2 or b.ndim < 2 or a.shape[-1] != b.shape[-2]
            or min(*a.shape, *b.shape) <= 0 or not a.is_floating_point()
            or a.dtype != b.dtype or a.device != b.device
            or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
        raise ValueError('Max-plus requires compatible nonempty finite floating tensors')
    return (a.unsqueeze(-1) + b.unsqueeze(-3)).max(dim=-2).values


def tropical_context(q, k, v, valid):
    """Finite Q/K/V [B,H,N,D], True=valid keys [B,K]; output [B,H,Q,Dv].

    Negative Hilbert range followed by max-plus aggregation, with no softmax.
    Query and key counts may differ. All-masked batches fail closed.
    """
    _validate_context(q, k, v, valid)
    difference = q.unsqueeze(-2) - k.unsqueeze(-3)
    scores = -(difference.max(-1).values - difference.min(-1).values)
    scores = scores.masked_fill(~valid[:,None,None,:], -torch.inf)
    return (scores.unsqueeze(-1) + v.unsqueeze(-3)).max(-2).values


def _validate_context(q, k, v, valid):
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4
            or q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3]
            or q.shape[-1] != k.shape[-1] or min(*q.shape,*k.shape,*v.shape) <= 0
            or not q.is_floating_point() or any(t.dtype != q.dtype or t.device != q.device for t in (k,v))
            or valid.dtype != torch.bool or valid.device != q.device
            or valid.shape != (k.shape[0],k.shape[2]) or not valid.any(-1).all()
            or any(not torch.isfinite(t).all() for t in (q,k,v))):
        raise ValueError('Invalid finite tropical tensors or key mask')


def tropical_context_blockwise(q, k, v, valid, *, block_size=32):
    """Exact full-context tiling, first-index ties; first-order autograd only.

    Saves Q/K and one winning key per output coordinate, not pairwise scores.
    Forward workspace is quadratic in block_size, not full context length.
    Backward recomputes extrema only for winning pairs. No context truncation.
    """
    if type(block_size) is not int or block_size <= 0:
        raise ValueError('block_size must be a positive integer')
    _validate_context(q, k, v, valid)
    return _BlockwiseContext.apply(q, k, v, valid, block_size)


class _BlockwiseContext(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, valid, block_size):
        shape = (*q.shape[:3], v.shape[-1])
        result = q.new_empty(shape)
        winners = torch.empty(shape, dtype=torch.long, device=q.device)
        for i in range(0, q.shape[2], block_size):
            qi = q[:, :, i:i+block_size]
            best = q.new_full((*qi.shape[:3], v.shape[-1]), -torch.inf)
            winner = torch.zeros_like(best, dtype=torch.long)
            for j in range(0, k.shape[2], block_size):
                diff = qi.unsqueeze(-2) - k[:, :, j:j+block_size].unsqueeze(-3)
                score = -(diff.max(-1).values - diff.min(-1).values)
                score.masked_fill_(~valid[:, None, None, j:j+block_size], -torch.inf)
                candidate, local = (score.unsqueeze(-1) + v[:, :, None, j:j+block_size]).max(-2)
                take = candidate > best  # Strict: earlier global key wins ties.
                winner = torch.where(take, local + j, winner)
                best = torch.where(take, candidate, best)
            result[:, :, i:i+block_size] = best
            winners[:, :, i:i+block_size] = winner
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite blockwise tropical context; arithmetic overflow')
        ctx.save_for_backward(q, k, winners)
        ctx.value_shape = v.shape
        ctx.block_size = block_size
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        q, k, winners = ctx.saved_tensors
        dq = torch.zeros_like(q)
        # Allocate flat key-gradient accumulator directly for noncontiguous inputs.
        dk_flat = k.new_zeros((*k.shape[:2], k.shape[2]*k.shape[3]))
        dv = grad.new_zeros(ctx.value_shape)
        b, h, _, d = q.shape
        for i in range(0, q.shape[2], ctx.block_size):
            index = winners[:, :, i:i+ctx.block_size]
            g = grad[:, :, i:i+ctx.block_size]
            count = index.shape[2]
            selected = k.gather(2, index.reshape(b,h,-1,1).expand(-1,-1,-1,d))
            diff = q[:, :, i:i+count, None] - selected.reshape(b,h,count,-1,d)
            hi, lo = diff.max(-1).indices, diff.min(-1).indices
            qpart = q.new_zeros((b,h,count,d))
            qpart.scatter_add_(-1, hi, -g)
            qpart.scatter_add_(-1, lo, g)
            dq[:, :, i:i+count] = qpart
            dk_flat.scatter_add_(-1, (index*d+hi).flatten(2), g.flatten(2))
            dk_flat.scatter_add_(-1, (index*d+lo).flatten(2), -g.flatten(2))
            dv.scatter_add_(2, index, g)
        dk = dk_flat.reshape(k.shape)
        return dq, dk, dv, None, None


class TropicalCodeAttention(nn.Module):
    """Pinned-code variant: log1p(ReLU), shared head-local max-plus QKV,
    symmetric Hilbert scores, max-plus values, expm1 and bias-free output.

    Normalization off, projection on. This is not literal paper Eq. 2:
    finite transform avoids log(0)/infinity subtraction. No clipping added;
    extreme exponent overflow raises rather than silently altering the method.
    """
    def __init__(self, width, heads, *, block_size=None):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError('Positive width must be divisible by heads')
        if block_size is not None and (type(block_size) is not int or block_size <= 0):
            raise ValueError('block_size must be a positive integer')
        self.block_size = block_size
        self.heads, self.head_width = heads, width // heads
        self.output = nn.Linear(width,width,bias=False)
        self.query = nn.Parameter(torch.randn(self.head_width,self.head_width))
        self.key = nn.Parameter(torch.randn(self.head_width,self.head_width))
        self.value = nn.Parameter(torch.randn(self.head_width,self.head_width))

    def forward(self, x, valid):
        if (x.ndim != 3 or x.shape[-1] != self.heads*self.head_width
                or x.dtype != self.query.dtype or x.device != self.query.device):
            raise ValueError('Invalid input dimensions, dtype or device')
        b,n,d = x.shape
        z = x.relu().log1p().reshape(b,n,self.heads,self.head_width).transpose(1,2)
        q,k,v = (maxplus(z,w.T) for w in (self.query,self.key,self.value))
        c = (tropical_context(q,k,v,valid) if self.block_size is None else
             tropical_context_blockwise(q,k,v,valid,block_size=self.block_size))
        c = c.transpose(1,2).reshape(b,n,d).expm1()
        result = self.output(c)
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite tropical output; exponent overflow is unsupported')
        return result
