"""Fixed source-admitted Tox21 prediction batch and transactional Adam state.

Reader admission is required upstream. This binding is integrity protection, not
new source admission, cooperative scheduling or molecular generation training.
"""
from dataclasses import dataclass
import hashlib
import math
import torch
from mora3.data.tox21 import ASSAYS, FORWARD, Tox21View
from .tox21 import SharedTox21Model
from .annotation_trainer import AnnotationTrainer, _digest


@dataclass(frozen=True)
class Tox21Batch:
    views: tuple

    def __post_init__(self):
        object.__setattr__(self, 'views', tuple(self.views))
        if not self.views or any(not isinstance(v, Tox21View) for v in self.views):
            raise ValueError('nonempty native Tox21 views required')
        for v in self.views:
            p = v.provenance()
            a = p.get('admission', {})
            if (v.direction != FORWARD or v.axis != ASSAYS or p.get('split') != 'train'
                or not p.get('effective_groups')
                or a.get('license') != 'US-government-public-domainoriginalTox21data;officialNCATSproviderstatement'
                or a.get('all12assayaxes_retained') is not True
                or a.get('physical_dose_time_missing') is not True
                or a.get('measuredcallannotation_prioronly') is not True):
                raise ValueError('native prediction train admission required')

    def binding(self):
        return _digest(dict(version=1, rows=[dict(direction=v.direction,
            axis=v.axis, values=v.values, observed_mask=v.observed_mask,
            reader_binding=v.binding_sha256,
            input=hashlib.sha256(v.input_bytes).hexdigest(),
            target=hashlib.sha256(v.target_bytes).hexdigest(),
            provenance=hashlib.sha256(v.provenance_bytes).hexdigest()) for v in self.views]))


class Tox21Trainer(AnnotationTrainer):
    """Single full-batch prediction objective; inherited transactional restore."""
    def __init__(self, model, batch, *, lr=.001):
        if not isinstance(model, SharedTox21Model) or not isinstance(batch, Tox21Batch):
            raise ValueError('native Tox21 model and batch required')
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError('finite positive learning rate required')
        model._validate(batch.views)
        self.model = model
        self.batch = batch
        self.steps = 0
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.binding = _digest(dict(kind='native-Tox21-prediction-trainer-v1',
            batch=batch.binding(), configuration=model.get_extra_state(),
            optimizer='Adam', lr=lr, betas=[.9,.999], eps=1e-8, weight_decay=0,
            schedule='fixed-full-batch-observed-call-mean',
            device=str(next(model.parameters()).device),
            dtype=str(next(model.parameters()).dtype)))

    def step(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.model.objective(self.model(self.batch.views), self.batch.views)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite Tox21 loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in self.model.parameters()):
            raise ValueError('nonfinite Tox21 gradient')
        self.optimizer.step()
        self.steps += 1
        return {'total': float(loss.detach())}
