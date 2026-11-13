from pathlib import Path
"""Exact independent native GFN sampler RNG recovery; zero optimizer updates."""
import copy
import json
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_primary_training import equal
from test_mu_optimizer import target, optimizer
from mora3.research.primary_training import NativeMuUpdateEngine


def qualified_fixture(seed=173):
    model, roles = target()
    opt = optimizer(model, roles)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=.8)
    sampler = torch.Generator(device='cpu').manual_seed(seed)
    trainer = NativeMuUpdateEngine(model, opt, binding_sha256='a'*64,
        task_cycle=('one', 'two', 'one'), source_cursors={'one':b'0', 'two':b'0'},
        scheduler=scheduler, sampler_generator=sampler)
    return trainer, sampler


def draw(sampler):
    return torch.randn((4, 6), generator=sampler)


def test_private_sampler_cursor_is_checkpointed_and_restored(tmp_path):
    trainer, sampler = qualified_fixture()
    draw(sampler)
    saved = trainer.checkpoint_state()
    assert saved['schema'] == 'Mora3.NativeMuUpdateState/v4'
    assert saved['sampler_rng']['device'] == 'cpu'
    assert saved['steps'] == 0 and saved['optimizer']['state'] == {}
    expected = draw(sampler)
    path = tmp_path/'sampler.pt'
    trainer.save_checkpoint(path)
    # Saving a new checkpoint after the draw must faithfully record the newer
    # sampler cursor. The earlier snapshot remains immutable.
    restored, other = qualified_fixture(seed=2)
    restored.restore_state(copy.deepcopy(saved))
    assert torch.equal(draw(other), expected)
    assert random.getstate() == saved['rng']['python']
    assert np.random.get_state()[1].tolist() == saved['rng']['numpy'][1]
    assert torch.equal(torch.get_rng_state(), saved['rng']['torch_cpu'])
    assert restored.steps == 0 and restored.source_cursors == trainer.source_cursors
    loaded, third = qualified_fixture(seed=3)
    loaded.load_checkpoint(path)
    assert torch.equal(draw(third), draw(sampler))


@pytest.mark.parametrize('corrupt', ['state_short', 'state_dtype', 'device', 'missing', 'rogue'])
def test_sampler_corruption_refuses_before_owner_or_rng_mutation(corrupt):
    trainer, sampler = qualified_fixture()
    before = trainer.checkpoint_state()
    malformed = copy.deepcopy(before)
    if corrupt == 'state_short':
        malformed['sampler_rng']['state'] = torch.tensor([1, 2], dtype=torch.uint8)
    elif corrupt == 'state_dtype':
        malformed['sampler_rng']['state'] = malformed['sampler_rng']['state'].to(torch.int64)
    elif corrupt == 'device':
        malformed['sampler_rng']['device'] = 'cuda:0'
    elif corrupt == 'missing':
        del malformed['sampler_rng']
    else:
        malformed['sampler_rng']['unexpected'] = True
    with pytest.raises((ValueError, RuntimeError)):
        trainer.restore_state(malformed)
    equal(trainer.checkpoint_state(), before)
    assert torch.equal(sampler.get_state(), before['sampler_rng']['state'])


def test_replaced_sampler_owner_refuses_even_if_state_matches():
    trainer, sampler = qualified_fixture()
    saved = trainer.checkpoint_state()
    trainer.sampler_generator = torch.Generator().set_state(sampler.get_state())
    with pytest.raises(ValueError, match='sampler|generator'):
        trainer.restore_state(copy.deepcopy(saved))
    with pytest.raises(ValueError, match='sampler|generator'):
        trainer.checkpoint_state()


def test_private_sampler_checkpoint_reloads_in_a_separate_python_process(tmp_path):
    trainer, sampler = qualified_fixture(seed=227)
    draw(sampler)
    checkpoint = tmp_path / 'actual-private-generator-checkpoint.pt'
    trainer.save_checkpoint(checkpoint)
    expected = draw(sampler).tolist()
    # The child has no access to the current trainer, global RNG, or sampler
    # object. It must reconstruct a fresh owner using the persisted bytes.
    code = """import json,sys
from test_campaign26_sampler_rng_recovery import qualified_fixture,draw
trainer, sampler = qualified_fixture(seed=999)
trainer.load_checkpoint(sys.argv[1])
print(json.dumps(dict(draw=draw(sampler).tolist(),steps=trainer.steps,
    task=trainer.next_task,cursors={k:v.decode() for k,v in trainer.source_cursors.items()})))
"""
    finished = subprocess.run([sys.executable, '-B', '-c', code, str(checkpoint)],
        env={'PATH':'/usr/bin:/bin', 'PYTHONPATH':':'.join(str(Path(__file__).resolve().parents[2]/part) for part in ('src','tests/model','tests/research')),
             'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
             'OPENBLAS_NUM_THREADS':'1'}, capture_output=True,text=True,check=True,timeout=60)
    result = json.loads(finished.stdout)
    assert result == dict(draw=expected,steps=0,task='one',cursors={'one':'0','two':'0'})


def test_sampler_cursor_coexists_with_real_native_mup_cosine_policy_without_updates():
    from mora3.research.mup_campaign_schedule import NativeMuWarmupCosine
    model, roles = target()
    opt = optimizer(model, roles)
    scheduler = NativeMuWarmupCosine(opt, total_updates=30)
    sampler = torch.Generator(device='cpu').manual_seed(427)
    trainer = NativeMuUpdateEngine(model, opt, binding_sha256='b'*64,
        task_cycle=('one', 'two'), source_cursors={'one':b'0', 'two':b'0'},
        scheduler=scheduler, gradient_clip_max_norm=.6, sampler_generator=sampler)
    draw(sampler)
    checkpoint = trainer.checkpoint_state()
    assert checkpoint['schema'] == 'Mora3.NativeMuUpdateState/v4'
    assert checkpoint['update_policy']['gradient_clip_max_norm'] == .6
    expected = draw(sampler)
    trainer.restore_state(copy.deepcopy(checkpoint))
    assert torch.equal(draw(sampler), expected)
    assert trainer.steps == 0 and trainer.optimizer.state == {}
    assert trainer.scheduler.last_epoch == 0
