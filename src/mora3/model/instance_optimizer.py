"""Fresh all-head Adam allocation for the native overlapping-instance provider.

This boundary cannot execute updates, adopt historical Adam/checkpoint state or
admit sources. Preflight is read-only, preserving pending gradients and modes.
"""
import copy
import math
import torch
from .instance_schedule import InstanceSchedule
from .annotation_trainer import _digest


class InstanceOptimizerBoundary:
    def __init__(self, schedule, *, learning_rate):
        if type(schedule) is not InstanceSchedule:
            raise ValueError('native instance provider required')
        if (type(learning_rate) not in (int, float)
                or not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError('explicit positive finite learning rate required')
        self.schedule = self._schedule = schedule
        self.model = self._model = schedule.model
        self._policy = copy.deepcopy(schedule.state_dict())
        self._parameters = tuple(self.model.parameters())
        self._modules = self._module_ownership()
        self._names = tuple(name for name, _ in self.model.named_parameters())
        self._extra = self._extra_configuration()
        self.optimizer = torch.optim.Adam(self._parameters, lr=float(learning_rate))
        self._optimizer = self.optimizer
        self._groups = copy.deepcopy(self.optimizer.state_dict()['param_groups'])
        self.steps = 0

    def _module_ownership(self):
        # Include shared paths, not only the first registered occurrence.
        return tuple((name, id(module)) for name, module in
                     self.model.named_modules(remove_duplicate=False))

    def _extra_configuration(self):
        return _digest({name: module.get_extra_state()
                        for name, module in self.model.named_modules()
                        if type(module).get_extra_state is not torch.nn.Module.get_extra_state})

    def preflight(self):
        """Validate allocation and source/provider identity without neural calls."""
        if (self.schedule is not self._schedule or self.model is not self._model
                or self.schedule.model is not self.model):
            raise ValueError('native instance provider/model ownership changed')
        self.schedule.validate_state(self._policy)
        if (self._module_ownership() != self._modules
                or self._extra_configuration() != self._extra):
            raise ValueError('native instance module ownership/configuration changed')
        if type(self.steps) is not int or self.steps != 0:
            raise ValueError('allocation-only boundary requires zero exact steps')
        parameters = tuple(self.model.parameters())
        allocated = tuple(p for g in self.optimizer.param_groups for p in g['params'])
        if (self.optimizer is not self._optimizer
                or tuple(map(id, parameters)) != tuple(map(id, self._parameters))
                or tuple(map(id, allocated)) != tuple(map(id, parameters))
                or tuple(n for n, _ in self.model.named_parameters()) != self._names
                or len(set(map(id, allocated))) != len(allocated)
                or self.optimizer.state
                or self.optimizer.state_dict()['param_groups'] != self._groups):
            raise ValueError('native instance optimizer allocation/configuration changed')
        return dict(version=1, boundary='fresh-all-head-Adam-native-instance-allocation-v1',
            provider_binding=self._policy['binding'], optimizer='Adam',
            optimizer_groups=copy.deepcopy(self._groups),
            parameter_names=list(self._names),
            parameter_count=sum(p.numel() for p in parameters),
            optimizer_updates=0, training_admission=False, execution_supported=False,
            historical_optimizer_adoption=False)

    def step(self):
        """Refuse before forward, RNG, modes, gradients or optimizer mutation."""
        self.preflight()
        for task in self.schedule.tasks:
            task.require_training()
        raise ValueError('native instance update execution/recovery interface unavailable')

    def _parameter_aliases(self):
        groups = {}
        for name, parameter in self.model.named_parameters(remove_duplicate=False):
            groups.setdefault(id(parameter), []).append(name)
        return list(groups.values())

    def allocation_state_dict(self):
        """Detached in-memory allocation snapshot; not an executable checkpoint."""
        self.preflight()
        return copy.deepcopy(dict(
            format='native-instance-allocation-state-v1', steps=0,
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
            raise ValueError('strict native instance allocation state schema required')
        if type(state['steps']) is not int or state['steps'] != 0:
            raise ValueError('allocation state requires exact zero steps')
        for key in ('format', 'configuration', 'provider', 'parameter_aliases', 'optimizer'):
            if _digest(state[key]) != _digest(expected[key]):
                raise ValueError('native instance allocation state binding mismatch: '+key)
        actual, reference = state['model'], expected['model']
        if not isinstance(actual, dict) or set(actual) != set(reference):
            raise ValueError('native instance model state keys mismatch')
        for name, ref in reference.items():
            value = actual[name]
            if isinstance(ref, torch.Tensor):
                if (type(value) is not torch.Tensor or value.shape != ref.shape
                        or value.dtype != ref.dtype or value.device != ref.device
                        or value.layout != ref.layout or value.requires_grad
                        or not torch.isfinite(value).all()):
                    raise ValueError('native instance tensor state mismatch: '+name)
            elif _digest(value) != _digest(ref):
                raise ValueError('native instance extra state mismatch: '+name)
        for aliases in expected['parameter_aliases']:
            if any(not torch.equal(actual[aliases[0]], actual[name]) for name in aliases[1:]):
                raise ValueError('native instance shared parameter aliases disagree')


    def load_allocation_state_dict(self, state):
        """Restore validated zero-step tensors transactionally in memory.

        Parameter identities/aliases, pending gradients, modes and RNG survive.
        No model loader hooks, extra-state setters or Adam adoption execute.
        This does not grant source admission or training recovery support.
        """
        candidate = copy.deepcopy(state)
        self.validate_allocation_state(candidate)
        tensors = {name: value for name, value in self.model.state_dict().items()
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
