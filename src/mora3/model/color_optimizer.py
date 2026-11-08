"""Fresh all-head Adam allocation and fail-closed native color update boundary.

Allocation is not training admission. No update execution or checkpoint adoption
is provided. Exact zero-step allocation tensors may be restored in memory;
native semantic corpus admission and a separately tested execution/
recovery interface remain required. Existing parent Adam state is never adopted.
"""
import copy
import math
import torch
from .color_schedule import ColorSemanticSchedule


class ColorOptimizerBoundary:
    def __init__(self, schedule, *, learning_rate):
        if type(schedule) is not ColorSemanticSchedule:
            raise ValueError('native color provider required')
        if (type(learning_rate) not in (int, float)
                or not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError('explicit positive finite learning rate required')
        self.schedule = schedule
        self.model = schedule.model
        self._schedule = schedule
        self._model = self.model
        self._policy = copy.deepcopy(schedule.state_dict())
        self._parameters = tuple(self.model.parameters())
        self.optimizer = torch.optim.Adam(self._parameters, lr=float(learning_rate))
        self._optimizer = self.optimizer
        self._groups = copy.deepcopy(self.optimizer.state_dict()['param_groups'])
        self.steps = 0

    def preflight(self):
        """Read-only binding/allocation check; never clears pending gradients."""
        if (self.schedule is not self._schedule or self.model is not self._model
                or self.schedule.model is not self.model):
            raise ValueError('native color provider/model ownership changed')
        self.schedule.validate_state(self._policy)
        if type(self.steps) is not int or self.steps != 0:
            raise ValueError('allocation-only boundary requires zero exact steps')
        parameters = tuple(self.model.parameters())
        allocated = tuple(p for g in self.optimizer.param_groups for p in g['params'])
        if (self.optimizer is not self._optimizer
                or tuple(map(id, parameters)) != tuple(map(id, self._parameters))
                or tuple(map(id, allocated)) != tuple(map(id, parameters))
                or len(set(map(id, allocated))) != len(allocated)
                or self.optimizer.state
                or self.optimizer.state_dict()['param_groups'] != self._groups):
            raise ValueError('native color optimizer allocation/configuration binding changed')
        return dict(version=1, boundary='fresh-all-head-Adam-native-color-allocation-v1',
            provider_binding=self._policy['binding'],optimizer='Adam',
            optimizer_groups=copy.deepcopy(self._groups),
            parameter_names=[name for name,_ in self.model.named_parameters()],
            parameter_count=sum(p.numel() for p in parameters),
            optimizer_updates=0,training_admission=False,execution_supported=False,
            historical_optimizer_adoption=False)

    def step(self):
        """Fail before mode/RNG/gradient/Adam mutation, including unselected rows."""
        self.preflight()
        for task in self.schedule.tasks:
            task.require_training()
        # Future source admission alone cannot enable an untested update path.
        raise ValueError('native color update execution/recovery interface unavailable')

    def _parameter_aliases(self):
        groups = {}
        for name, parameter in self.model.named_parameters(remove_duplicate=False):
            groups.setdefault(id(parameter), []).append(name)
        return list(groups.values())

    def allocation_state_dict(self):
        """Detached in-memory allocation snapshot; not an executable checkpoint."""
        self.preflight()
        return copy.deepcopy(dict(
            format='native-color-allocation-state-v1', steps=0,
            configuration=self.model.get_extra_state(),
            provider=self.schedule.state_dict(),
            parameter_aliases=self._parameter_aliases(),
            model=self.model.state_dict(), optimizer=self.optimizer.state_dict()))

    def validate_allocation_state(self, state):
        """Validate without loading tensors, adopting Adam state or granting use."""
        self.preflight()
        from .profile_reconstruction import _digest
        expected = self.allocation_state_dict()
        if type(state) is not dict or set(state) != set(expected):
            raise ValueError('strict native color allocation state schema required')
        if type(state['steps']) is not int or state['steps'] != 0:
            raise ValueError('allocation state requires exact zero steps')
        for key in ('format', 'configuration', 'provider', 'parameter_aliases', 'optimizer'):
            if _digest(state[key]) != _digest(expected[key]):
                raise ValueError('native color allocation state binding mismatch: '+key)
        actual, reference = state['model'], expected['model']
        if not isinstance(actual, dict) or set(actual) != set(reference):
            raise ValueError('native color model state keys mismatch')
        for name, ref in reference.items():
            value = actual[name]
            if isinstance(ref, torch.Tensor):
                if (type(value) is not torch.Tensor or value.shape != ref.shape
                        or value.dtype != ref.dtype or value.device != ref.device
                        or value.layout != ref.layout or value.requires_grad
                        or not torch.isfinite(value).all()):
                    raise ValueError('native color tensor state mismatch: '+name)
            elif _digest(value) != _digest(ref):
                raise ValueError('native color extra state mismatch: '+name)
        for aliases in expected['parameter_aliases']:
            if any(not torch.equal(actual[aliases[0]], actual[name]) for name in aliases[1:]):
                raise ValueError('native color shared parameter aliases disagree')

    def load_allocation_state_dict(self, state):
        """Restore exact zero-step tensor values after complete binding validation.

        This is in-memory allocation restoration, not checkpoint migration or
        training recovery. Parameter objects, modes, gradients and RNG survive.
        No model hooks or extra-state setters execute during restoration.
        """
        candidate = copy.deepcopy(state)
        self.validate_allocation_state(candidate)
        live = self.model.state_dict()
        tensors = {name: value for name, value in live.items()
                   if isinstance(value, torch.Tensor)}
        previous = {name: value.detach().clone() for name, value in tensors.items()}
        with torch.no_grad():
            try:
                for name, value in tensors.items():
                    value.copy_(candidate['model'][name])
            except Exception:
                for name, value in tensors.items():
                    value.copy_(previous[name])
                raise
