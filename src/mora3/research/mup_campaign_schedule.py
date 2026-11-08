"""Exact common warmup/cosine learning rate for declared MuAdamW owners."""

import math

from mora3.model.mixed_payload_projection import exact_wire
from mora3.model.mu_optimizer import MuAdamW


SCHEMA = 'Mora3.NativeMuWarmupCosine/v1'


def _warmup_updates(total_updates):
    if type(total_updates) is not int or total_updates < 10 or total_updates % 10:
        raise ValueError('Exact 10 percent warmup requires a positive horizon divisible by ten')
    return total_updates // 10


def warmup_cosine_factor(total_updates, completed_updates):
    """Return the factor for the next update, or zero at the terminal cursor."""
    warmup = _warmup_updates(total_updates)
    if type(completed_updates) is not int or not 0 <= completed_updates <= total_updates:
        raise ValueError('Exact completed update within the declared schedule horizon required')
    if completed_updates == total_updates:
        return 0.
    next_update = completed_updates + 1
    if next_update <= warmup:
        return next_update / warmup
    progress = (next_update - warmup) / (total_updates - warmup + 1)
    return math.cos((math.pi / 2.) * progress) ** 2


class NativeMuWarmupCosine:
    """One common LR factor with an exact owner-bound, primitive checkpoint.

    Scheduler position counts completed optimizer updates. The initial state
    sets the first update to one warmup increment. The update engine advances
    the scheduler once after each successful optimizer step.
    """

    def __init__(self, optimizer, *, total_updates):
        warmup = _warmup_updates(total_updates)
        if type(optimizer) is not MuAdamW:
            raise ValueError('Exact native MuAdamW owner required')
        optimizer._validate_live()
        groups = optimizer.param_groups
        base_lr = optimizer._mup_metadata['base_lr']
        initial_lrs = tuple(group['lr'] for group in groups)
        if any('initial_lr' in group or initial != base_lr * group['mup_lr_multiplier']
               for group, initial in zip(groups, initial_lrs, strict=True)):
            raise ValueError('Fresh native optimizer with exact initial rates required for cosine schedule')
        self.optimizer = optimizer
        self._owner = (optimizer,)
        self._specification = (total_updates, warmup, initial_lrs)
        self._completed_updates = 0
        for group, initial in zip(groups, initial_lrs, strict=True):
            group['initial_lr'] = initial
        self._apply_rates(0)

    @property
    def last_epoch(self):
        return self._completed_updates

    @property
    def _step_count(self):
        return self._completed_updates + 1

    def _rates(self, completed_updates):
        total, _, initial_lrs = self._specification
        factor = warmup_cosine_factor(total, completed_updates)
        return tuple(rate * factor for rate in initial_lrs)

    def _state(self, completed_updates):
        total, warmup, initial_lrs = self._specification
        return dict(schema=SCHEMA, total_updates=total, warmup_updates=warmup,
                    completed_updates=completed_updates, initial_lrs=initial_lrs,
                    last_lr=self._rates(completed_updates))

    def validate_state(self, state, completed_updates, optimizer):
        """Validate policy, cursor and exact group rates before any owner mutation."""
        if self.optimizer is not self._owner[0] or type(optimizer) is not MuAdamW:
            raise ValueError('Exact cosine schedule optimizer ownership required')
        expected = self._state(completed_updates)
        if not exact_wire(state, expected):
            raise ValueError('Exact cosine schedule policy, position and rate state required')
        groups = optimizer.param_groups
        if (
            not exact_wire(tuple(group.get('initial_lr') for group in groups), expected['initial_lrs'])
            or not exact_wire(tuple(group.get('lr') for group in groups), expected['last_lr'])
        ):
            raise ValueError('Optimizer rates disagree with the exact cosine schedule')

    def state_dict(self):
        state = self._state(self._completed_updates)
        self.validate_state(state, self._completed_updates, self.optimizer)
        return state

    def load_state_dict(self, state):
        # Recovery must stage/restore the optimizer rates first, exactly as
        # during engine recovery. Loading does not rewrite an optimizer group.
        position = state.get('completed_updates') if type(state) is dict else None
        self.validate_state(state, position, self.optimizer)
        self._completed_updates = position

    def get_last_lr(self):
        return list(self.state_dict()['last_lr'])

    def require_step_available(self, completed_updates):
        if (type(completed_updates) is not int or completed_updates != self._completed_updates
                or completed_updates >= self._specification[0]):
            raise ValueError('Completed update disagrees with the finite cosine schedule horizon')

    def _apply_rates(self, completed_updates):
        for group, rate in zip(self.optimizer.param_groups, self._rates(completed_updates), strict=True):
            group['lr'] = rate

    def step(self):
        self.require_step_available(self._completed_updates)
        self.state_dict()
        position = self._completed_updates + 1
        self._apply_rates(position)
        self._completed_updates = position
