"""Bounded sparse activation for continuous-latent FoT (inference selection).

2412.09078v1 Eq3 specifies top-m scored nodes. Setting min_distance=0
implements that rule with explicit first-index ties and eligible candidates.
Positive distance is a Mora-3 adaptation: greedy score-ordered Euclidean
separation, not a paper method, semantic validator or GFN sampling kernel.
Coordinates/score calibration and per-tree invocation belong to the caller.
"""
import math
import torch


@torch.no_grad()
def select_branches(states, scores, budget, *, min_distance=0., eligible=None):
    """Return up to budget original indices, ranked by score then input index.

    Reject invalid numeric inputs even when ineligible. Distance is inclusive;
    no padding, duplicate replacement, hidden normalization or source mutation.
    Selection is nondifferentiable; gathering original states retains gradients.
    Work O(N log N + N*K*D), auxiliary storage O(N + D), K<=budget.
    """
    if (states.ndim!=2 or states.shape[1]==0 or not states.is_floating_point()
        or scores.shape!=(len(states),) or scores.dtype!=states.dtype
        or scores.device!=states.device):
        raise ValueError('Require floating [N,D] states and matching [N] scores')
    if not torch.isfinite(states).all() or not torch.isfinite(scores).all():
        raise ValueError('Nonfinite candidate or score')
    if type(budget) is not int or budget<=0:
        raise ValueError('Require positive integer budget')
    if isinstance(min_distance,bool) or not isinstance(min_distance,(int,float)) or not math.isfinite(min_distance) or min_distance<0:
        raise ValueError('Require finite nonnegative distance')
    if eligible is None:
        eligible=torch.ones(len(states),dtype=torch.bool,device=states.device)
    if eligible.shape!=(len(states),) or eligible.dtype!=torch.bool or eligible.device!=states.device:
        raise ValueError('Require matching boolean eligibility')
    order=torch.argsort(scores,descending=True,stable=True)
    selected=[]
    # Float64 distance arithmetic avoids overflow at ordinary FP32 magnitudes.
    # Coordinate magnitudes whose differences overflow are rejected, not clipped.
    for i in order.tolist():
        if not eligible[i].item():continue
        accept=True
        for j in selected:
            distance=torch.linalg.vector_norm(states[i].double()-states[j].double())
            if not torch.isfinite(distance):
                raise ValueError('Unrepresentable distance')
            if distance.item()<min_distance:
                accept=False;break
        if accept:selected.append(i)
        if len(selected)==budget:break
    return torch.tensor(selected,dtype=torch.long,device=states.device)
