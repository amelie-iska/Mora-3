"""Fail-closed native same-run optimizer clock recovery; no optimizer steps in tests."""
import copy

import pytest
import torch

from test_primary_training import engine, equal


@pytest.mark.parametrize("forged_step", [0, 1, 17])
def test_checkpoint_rejects_optimizer_moments_before_any_native_update(forged_step):
    trainer = engine()
    before = trainer.checkpoint_state()
    assert before["steps"] == 0 and before["optimizer"]["state"] == {}
    forged = copy.deepcopy(before)
    parameter = trainer.optimizer.param_groups[0]["params"][0]
    forged["optimizer"]["state"][0] = {
        "step": torch.tensor(float(forged_step), dtype=torch.float32),
        "exp_avg": torch.zeros_like(parameter.detach()),
        "exp_avg_sq": torch.zeros_like(parameter.detach()),
    }
    with pytest.raises(ValueError, match="optimizer.*step|clock"):
        trainer.restore_state(forged)
    equal(trainer.checkpoint_state(), before)


def test_future_optimizer_moment_cannot_be_replayed_as_older_checkpoint():
    trainer = engine()
    before = trainer.checkpoint_state()
    forged = copy.deepcopy(before)
    forged["steps"] = 2
    forged["scheduler"]["last_epoch"] = 2
    forged["scheduler"]["_step_count"] = 3
    # A malformed schedule must already refuse; clock guard independently
    # rejects the impossible Adam step without mutating live owners.
    parameter = trainer.optimizer.param_groups[0]["params"][0]
    forged["optimizer"]["state"][0] = {
        "step": torch.tensor(3.0, dtype=torch.float32),
        "exp_avg": torch.zeros_like(parameter.detach()),
        "exp_avg_sq": torch.zeros_like(parameter.detach()),
    }
    with pytest.raises(ValueError, match="optimizer.*step|clock"):
        trainer.restore_state(forged)
    equal(trainer.checkpoint_state(), before)


def test_valid_zero_update_checkpoint_restores_exact_owner_and_cursor_state():
    trainer = engine()
    state = trainer.checkpoint_state()
    trainer.restore_state(copy.deepcopy(state))
    equal(trainer.checkpoint_state(), state)
    assert trainer.steps == 0 and trainer.optimizer.state == {}
