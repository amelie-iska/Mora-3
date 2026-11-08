"""Opt-in native LINCS objective provider; no optimizer or checkpoint ownership.

Fixed cyclic endpoint sampling with caller-declared positive endpoint weights.
Weights are experimental scale choices, not calibrated cross-assay equivalence.
Reverse views remain latent-only and are excluded from supervised scheduling.
"""
import copy
import math
from .lincs_model import SharedLincsModel
from .annotation_trainer import _digest


class LincsSchedule:
    def __init__(self, model, views, *, weights, batch_size):
        if not isinstance(model, SharedLincsModel):
            raise ValueError('native shared LINCS model required')
        if (type(weights) is not dict or set(weights) != set(model.axes)
                or any(type(w) not in (int,float) or not math.isfinite(w) or w <= 0
                       for w in weights.values())):
            raise ValueError('explicit finite positive weights for both native endpoints required')
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('positive integer batch size required')
        views = model._validate(views)
        seen=set()
        for v in views:
            p=v.provenance()
            if (v.direction != 'molecule_to_profile' or p.get('split') != 'train'
                    or p.get('component_verified') is not True
                    or not p.get('effective_groups') or p.get('physical_dose_valid') is not False
                    or p.get('image_available') is not False or p.get('mask_available') is not False
                    or p.get('task_id') in seen):
                raise ValueError('unique component-qualified source-coordinate train prediction views required')
            seen.add(p['task_id'])
        self.model=model
        self._weights=tuple((m,float(weights[m])) for m in ('CellPainting','L1000'))
        self.batch_size=batch_size
        self.pools={m:tuple(v for v in views if v.modality==m) for m,_ in self._weights}
        if any(not p for p in self.pools.values()):
            raise ValueError('both native endpoint pools required')
        self._binding=_digest(dict(version=1,views=[v.binding_sha256 for v in views],
            configuration=model.get_extra_state(),weights=dict(self._weights),batch_size=batch_size,
            schedule='alternating-endpoints-cyclic-rows-v1',objective='weight-times-record-mean-observed-MSE'))

    @property
    def weights(self):
        return dict(self._weights)

    def state_dict(self):
        return copy.deepcopy(dict(version=1,binding=self._binding,weights=self.weights,
            batch_size=self.batch_size,schedule='alternating-endpoints-cyclic-rows-v1'))

    def validate_state(self,state):
        # Canonical digest distinguishes bool/int aliases and rejects malformed payloads.
        if _digest(state) != _digest(self.state_dict()):
            raise ValueError('native source schedule binding mismatch')

    def select(self,step):
        if type(step) is not int or step < 0:
            raise ValueError('nonnegative integer native schedule step required')
        modality=self._weights[step % 2][0]
        pool=self.pools[modality]; size=min(self.batch_size,len(pool))
        start=(step // 2)*size
        return modality,tuple(pool[(start+i)%len(pool)] for i in range(size))

    def objective(self,step):
        modality,views=self.select(step)
        loss=self.model.objective(self.model(views),views)*self.weights[modality]
        if not math.isfinite(float(loss.detach())):
            raise ValueError('nonfinite native LINCS objective')
        return loss,len(views)
