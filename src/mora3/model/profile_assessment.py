"""Full source-axis profile diagnostics, with target-free prediction boundary.

Per-record feature correlation is descriptive across heterogeneous source units;
it is not calibrated biological accuracy. No heldout eligibility is inferred.
"""
import torch
from .molecule_profile import SharedMoleculeProfileModel


def profile_metrics(prediction, target):
    """Float64 CPU metrics; constant rows have undefined Pearson correlation."""
    if (prediction.ndim != 2 or prediction.shape != target.shape
            or not all(prediction.shape) or not torch.isfinite(prediction).all()
            or not torch.isfinite(target).all()):
        raise ValueError('nonempty aligned finite full-axis arrays required')
    z=prediction.detach().double().cpu();y=target.detach().double().cpu()
    error=z-y
    a=z-z.mean(1,keepdim=True);b=y-y.mean(1,keepdim=True)
    denom=a.norm(dim=1)*b.norm(dim=1);defined=denom>0
    correlation=(a[defined]*b[defined]).sum(1)/denom[defined]
    return dict(records=len(y),coordinates_per_record=y.shape[1],
        mse=float(error.square().mean()),mae=float(error.abs().mean()),
        zero_baseline_mse=float(y.square().mean()),
        mean_record_pearson=float(correlation.mean()) if defined.any() else None,
        pearson_defined_records=int(defined.sum()))


def predict_profiles(model, views):
    """Run no-grad inference and restore heterogeneous modes and CPU/CUDA RNG.

    This boundary never reads targets/provenance; ordered axes remain validated.
    Parameters, gradients and optimizer are untouched. No model copy or update.
    """
    if not isinstance(model,SharedMoleculeProfileModel):
        raise ValueError('native profile model required')
    modes=tuple((m,m.training) for m in model.modules())
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices),torch.no_grad():
            model.eval()
            return tuple(z.detach().cpu().clone() for z in model(tuple(views)))
    finally:
        for m,training in modes:m.training=training


def assess_profiles(model, views):
    """Read targets only after all target-free predictions; no train/holdout claim."""
    views=tuple(views)
    predictions=predict_profiles(model,views)
    groups={}
    for z,v in zip(predictions,views):
        target=v.targets()['source_profile_array']
        if tuple(target['axis_ids'])!=v.axis:
            raise ValueError('ordered target source axis mismatch')
        groups.setdefault(v.modality,[]).append((z,torch.tensor(target['values'],dtype=torch.float64)))
    return dict(predictions=predictions,modalities={m:profile_metrics(
        torch.stack([z for z,y in rows]),torch.stack([y for z,y in rows])) for m,rows in groups.items()})
