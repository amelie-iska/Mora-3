"""Train-only response diagnostics; fitted means are NOT held-out predictors.

For squared error, the empirical conditional mean minimizes loss over all
functions of the observed input on this finite sample. It is a diagnostic lower
bound, not an estimate of irreducible biological noise or generalization error.
"""
import torch


def response_quality(prediction, target, observed_keys):
    """Compare deterministic full-axis predictions to fitted empirical means.

    Keys must encode only the complete observed model input, excluding targets,
    cell IDs and hidden metadata. Caller owns that contract. Reductions use CPU
    float64 to keep diagnostic calculations independent of training precision.
    """
    if (prediction.ndim!=2 or prediction.shape!=target.shape or
        not prediction.is_floating_point() or not target.is_floating_point() or
        not prediction.numel() or len(observed_keys)!=len(target) or
        not torch.isfinite(prediction).all() or not torch.isfinite(target).all()):
        raise ValueError('finite nonempty matching floating matrices and keys required')
    p=prediction.detach().cpu().double();y=target.detach().cpu().double()
    groups={}
    for i,key in enumerate(observed_keys):groups.setdefault(key,[]).append(i)
    fitted=torch.empty_like(y)
    for indices in groups.values():
        if not torch.equal(p[indices],p[indices[0]].expand(len(indices),-1)):
            raise ValueError('equal observed inputs require deterministic predictions')
        fitted[indices]=y[indices].mean(0)
    mse=(p-y).square().mean().item()
    constant=(y-y.mean(0)).square().mean().item()
    lower=(y-fitted).square().mean().item()
    excess=(p-fitted).square().mean().item()
    return dict(prediction_mse=mse,gene_mean_mse=constant,context_mean_mse=lower,
        excess_over_context_mean_mse=excess,decomposition_residual=mse-lower-excess,
        groups=len(groups),singleton_groups=sum(len(v)==1 for v in groups.values()),
        training_gate_passed=mse<constant)
