"""Source-bound train-only residue supervision with native full-state Adam.

Reuses the tested transactional restore and finite-gradient machinery of the
annotation trainer; it grants no new corpus admission or generation eligibility.
"""
from dataclasses import dataclass
import torch
from mora3.data.uniprot_residue import residue_batch
from mora3.model.annotation_trainer import AnnotationTrainer, _digest


@dataclass(frozen=True)
class ResidueTrainingBatch:
    views: tuple
    feature_axis: tuple
    applicable: tuple

    def __post_init__(self):
        for name in ('views','feature_axis','applicable'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        residue_batch(self.views,feature_axis=self.feature_axis)
        if len(self.applicable)!=len(self.views) or any(type(x) is not bool for x in self.applicable):
            raise ValueError('Boolean per-row applicability required')
        for v in self.views:
            p=v.provenance()
            if p.get('split')!='train' or p.get('license')!='CC-BY-4.0':
                raise ValueError('source train-only rights scope required')

    def binding(self):
        return _digest(dict(residue=residue_batch(self.views,feature_axis=self.feature_axis).binding_sha256,
                            applicable=self.applicable))

    def applicability(self,device):
        return torch.tensor(self.applicable,dtype=torch.bool,device=device)


class ResidueTrainer(AnnotationTrainer):
    def __init__(self,model,batch,*,lr=.003):
        from mora3.model.residue_objective import SharedResidueModel
        if not isinstance(model,SharedResidueModel) or not isinstance(batch,ResidueTrainingBatch):
            raise ValueError('native shared residue model and training batch required')
        if batch.feature_axis!=model.head.feature_axis:
            raise ValueError('residue feature axis mismatch')
        self.model=model;self.batch=batch;self.steps=0
        self.optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        self.binding=_digest(dict(kind='native-residue-trainer-v1',batch=batch.binding(),
            configuration=model.get_extra_state(),optimizer='Adam',lr=lr,
            betas=[.9,.999],eps=1e-8,weight_decay=0,
            device=str(next(model.parameters()).device),dtype=str(next(model.parameters()).dtype)))
