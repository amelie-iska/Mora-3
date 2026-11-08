"""Opt-in native LINCS endpoints in one shared cooperative optimizer.

Independent source pools receive explicit weighted prediction updates; this is
neither calibrated cross-assay equivalence nor paired image/expression evidence.
Legacy 977-axis heads and historical checkpoint formats are unchanged.
"""
import copy
from .reconstruction_cooperative import (GeometryReconstructionCooperativeModel,
                                        GeometryReconstructionCooperativeTrainer)
from .lincs_model import SharedLincsModel
from .lincs_schedule import LincsSchedule
from .annotation_trainer import _digest


class LincsCooperativeModel(GeometryReconstructionCooperativeModel):
    def __init__(self, *args, lincs_axes, **kwargs):
        super().__init__(*args, **kwargs)
        self.lincs = SharedLincsModel(self.annotation,lincs_axes,**(dict(parameterization_policy='mup-lincs-v1') if self.parameterization_policy!='legacy' else {}))
        self.configuration = dict(self.configuration,
            architecture='unpaired-native-seventeen-objective-LINCS-v1',
            lincs=self.lincs.get_extra_state(),
            lincs_pairing='independent-source-observations-no-image-join')


class LincsCooperativeTrainer(GeometryReconstructionCooperativeTrainer):
    """Fresh Adam owns native heads and all shared parameters exactly once.

    Each cooperative cycle visits CP then L1000 once. Native row cursors advance
    once per cycle. The fixed reconstruction archive is preserved. Strict parent
    preflight covers aliases, Adam, runtime, modes, gradients and device RNG.
    """
    def __init__(self, model, batch, pools, *, lincs_views, lincs_weights,
                 lincs_batch_size, **kwargs):
        if not isinstance(model, LincsCooperativeModel):
            raise ValueError('native LINCS cooperative model required')
        self.lincs_views = tuple(lincs_views)
        self.lincs_weights = copy.deepcopy(lincs_weights)
        self.lincs_batch_size = lincs_batch_size
        self.lincs_schedule = LincsSchedule(model.lincs, self.lincs_views,
            weights=lincs_weights, batch_size=lincs_batch_size)
        super().__init__(model, batch, pools, **kwargs)
        self.kinds += ('lincs_CellPainting', 'lincs_L1000')
        self._reconstruction_schedule = (self.kinds, self.reconstruction_batch_size)
        self._geometry_policy = copy.deepcopy(self._geometry_configuration())
        self._lincs_policy = self.lincs_schedule.state_dict()
        self.binding = _digest(dict(version=1, geometry_binding=self.binding,
            lincs=self._lincs_policy, schedule=self.kinds,
            policy='independent-weighted-native-endpoints-once-per-cycle-v1',
            optimizer_boundary='fresh-Adam-native-LINCS-no-historical-adoption'))
        self._geometry_binding = self.binding

    def _check_policy(self):
        super()._check_policy()
        if (self.model.lincs.shared is not self.model.annotation
                or self.lincs_schedule.model is not self.model.lincs):
            raise ValueError('native LINCS shared model ownership changed')
        current = LincsSchedule(self.model.lincs, self.lincs_views,
            weights=self.lincs_weights, batch_size=self.lincs_batch_size)
        current.validate_state(self._lincs_policy)
        self.lincs_schedule.validate_state(self._lincs_policy)
        # A schedule's public pools are live objects; bind them too, not just
        # the cached receipt created by its constructor.
        for modality in current.pools:
            if self.lincs_schedule.pools[modality] != current.pools[modality]:
                raise ValueError('native LINCS live endpoint pool changed')

    def additional_objective(self, kind):
        if kind not in ('lincs_CellPainting', 'lincs_L1000'):
            return super().additional_objective(kind)
        endpoint = 0 if kind == 'lincs_CellPainting' else 1
        return self.lincs_schedule.objective((self.steps // len(self.kinds))*2 + endpoint)

    def state_dict(self):
        state = super().state_dict()
        state['lincs_schedule'] = copy.deepcopy(self._lincs_policy)
        return state

    def load_state_dict(self, state):
        self._check_policy()
        if not isinstance(state, dict) or 'lincs_schedule' not in state:
            raise ValueError('native LINCS checkpoint source schedule required')
        self.lincs_schedule.validate_state(state['lincs_schedule'])
        return super().load_state_dict(state)
