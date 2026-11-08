"""Observed within-assay goal-conditioned pair inference; no new intervention claims."""
import math
import torch

from mora3.data.lincx_fetch_native_rows import LinCxNativeFetchSource,PANELS
from .lincx_fetch_model import SharedLinCxFetchModel


def observed_fetch_pair_goals(model,source,rows):
    """Use one forward score per known pair and an inverse likelihood over the same panel.

    The desired assay value is a query outside the variant graph. Equal reported
    scores are jointly positive and never force one arbitrary original pair to
    be uniquely correct. There are no cross-panel competitors or invented variants.
    """
    if type(model) is not SharedLinCxFetchModel or type(source) is not LinCxNativeFetchSource:
        raise ValueError('exact source-bound FETCH model and original group pool required')
    rows=tuple(rows);source.require_rows(rows)
    panel_members={panel:tuple(i for i,r in enumerate(rows) if r.panel==panel) for panel in PANELS}
    if any(len(x)==1 for x in panel_members.values()):
        raise ValueError('at least two observed alternatives required for each included assay panel')
    result=model.reference_objective(source,rows)
    prior_mean=result['mean'];prior_scale=result['scale']
    terms=[];reports=[]
    for panel,indices in panel_members.items():
        if not indices:continue
        p=list(indices);mean=prior_mean[p];scale=prior_scale[p]
        desired_axis=tuple(dict.fromkeys(rows[i].reported_mean_lexeme for i in indices))
        desired=torch.tensor(tuple(map(float,desired_axis)),device=mean.device,dtype=mean.dtype)
        known=[rows[i].reported_mean_lexeme for i in indices]
        positive=torch.tensor([[value==target for target in known]for value in desired_axis],device=mean.device,dtype=torch.bool)
        score=-.5*((desired[:,None]-mean)/scale).square()-scale.log()-.5*math.log(2*math.pi)
        # The uniform prior over explicit alternatives cancels in the log ratio.
        obs=torch.logsumexp(score.masked_fill(~positive,-torch.inf),dim=1)
        normalizer=torch.logsumexp(score,dim=1)
        nll=normalizer-obs
        if not torch.isfinite(nll).all():raise ValueError('finite observed goal-conditioned loss required')
        terms.append(nll)
        reports.append(dict(panel=panel,desired_native_mean_lexemes=desired_axis,
            source_alternative_indices=indices,observed_positive_mask=positive,nll=nll))
    return dict(forward=result,observed_goal_inverse_loss=torch.cat(terms).mean(),native_goal_panel_records=tuple(reports),
        assay_only=True,training_admission=False)
