"""Masked source-call assessment; no calibration or biological validation implied."""
import torch


def _validate(values, mask):
    if (values.ndim != 2 or values.shape != mask.shape or mask.dtype != torch.bool
            or not torch.isfinite(values[mask]).all()
            or not ((values[mask] == 0) | (values[mask] == 1)).all()):
        raise ValueError('binary observed calls and same-shape boolean mask required')


def fit_call_prior(values, mask):
    """Jeffreys-smoothed train-only per-axis log odds; absent axes use 0.5."""
    _validate(values, mask)
    y=values.detach().double().cpu();m=mask.detach().cpu()
    positives=torch.where(m,y,0.).sum(0)
    return torch.log((positives+.5)/(m.sum(0)-positives+.5)).tolist()


def _metrics(z,y):
    n=len(y);p=int(y.sum());negative=n-p
    result=dict(observed=n,positive=p,inactive=negative,bce=None,brier=None,
                accuracy=None,roc_auc=None,roc_auc_undefined=None)
    if n:
        result.update(bce=float((torch.nn.functional.softplus(z)-z*y).mean()),
            brier=float(((z.sigmoid()-y)**2).mean()),
            accuracy=float(((z>=0)==y.bool()).double().mean()))
    if not p or not negative:
        result['roc_auc_undefined']='requires_both_classes'
    else:
        # Mann-Whitney statistic with average ranks for exact score ties.
        order=torch.argsort(z);scores=z[order];labels=y[order]
        _,counts=torch.unique_consecutive(scores,return_counts=True)
        end=counts.cumsum(0).double();rank=end-(counts.double()-1)/2
        ranks=torch.repeat_interleave(rank,counts)
        result['roc_auc']=float((ranks[labels.bool()].sum()-p*(p+1)/2)/(p*negative))
    return result


def masked_call_metrics(logits, values, mask):
    """Threshold zero logits; exclude unknowns before all math and report support."""
    _validate(values,mask)
    if logits.shape!=values.shape or not torch.isfinite(logits[mask]).all():
        raise ValueError('finite observed logits and matching shape required')
    z=logits.detach().double().cpu();y=values.detach().double().cpu();m=mask.detach().cpu()
    axes=[_metrics(z[:,j][m[:,j]],y[:,j][m[:,j]]) for j in range(y.shape[1])]
    aucs=[r['roc_auc'] for r in axes if r['roc_auc'] is not None]
    return dict(axes=axes,micro=_metrics(z[m],y[m]),
                macro_roc_auc=sum(aucs)/len(aucs) if aucs else None,
                macro_roc_auc_axes=len(aucs))
