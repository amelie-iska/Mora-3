"""Read-only coarse GOA diagnostics, stratified by source evidence code.

Strata overlap when several evidence codes support one observed term. Pooled
counts deduplicate those terms. These are assertion diagnostics, not calibrated
biological truth or evidence of experimental validation. No admission is granted.
"""
import torch
from torch.nn import functional as F
from .uniprot_go import _GOA_EVIDENCE, goa_targets


@torch.no_grad()
def evaluate_goa_evidence(logits, assertion_rows, axis_ids, *, subjects):
    """Evaluate exact-subject source observations without changing model state.

    Uses the existing coarse objective's form/context/ND exclusions. Global
    polarity conflicts refuse before evidence splitting. Empty strata return
    None, never a misleading zero error. Unknown logits may be nonfinite.
    Inputs must be independently source-verified; this pure function does not
    authenticate raw assertions, select train/test splits or run a predictor.
    """
    axis = tuple(axis_ids)
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
            or not logits.is_floating_point()
            or logits.shape != (len(assertion_rows), len(axis))):
        raise ValueError('floating batch-by-axis logits required')
    y, observed = goa_targets(assertion_rows, axis, subjects=subjects,
                             device=logits.device, dtype=logits.dtype)

    def metrics(targets, mask):
        z, t = logits[mask], targets[mask]
        if not torch.isfinite(z).all():
            raise ValueError('finite observed GOA logits required')
        count = z.numel()
        positive = int(t.sum().item())
        return dict(observed=count, positive=positive, negative=count-positive,
                    bce=F.binary_cross_entropy_with_logits(z,t).item() if count else None,
                    brier=((z.sigmoid()-t)**2).mean().item() if count else None)

    pooled = metrics(y, observed)
    strata = {}
    for code in sorted(_GOA_EVIDENCE):
        selected = [[a for a in row if a.get('goEvidence') == code]
                    for row in assertion_rows]
        target, mask = goa_targets(selected, axis, subjects=subjects,
                                  device=logits.device, dtype=logits.dtype)
        strata[code] = metrics(target, mask)
    return dict(policy='exact-subject-coarse-GOA-evidence-evaluation-v1',
                pooled=pooled, by_evidence=strata,
                unknown_is_negative=False, strata_are_disjoint=False)
