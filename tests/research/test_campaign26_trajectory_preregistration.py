from pathlib import Path
"""Test-only predeclared native complete-path GFN source/method freeze; ZERO updates."""
import copy

import pytest
import torch

from test_mu_optimizer import target, optimizer
from mora3.research.primary_training import NativeMuUpdateEngine


def declaration():
    return dict(
        schema='Mora3.NativePathspaceBinding/v1',
        trial_id='test-only-proxy-01',
        task_key='trajectory',
        source_snapshot_sha256='a'*64,
        policy_module_path='core.parent.parent.profile_gfn',
        history_module_path='core.parent.parent.geometry_fusion.geometry',
        prefix_binding_sha256='b'*64,
        reward_binding_sha256='c'*64,
        sampling_method='native-unpruned-history-conditioned-forced-horizon',
        terminal_object='entire-ordered-reasoning-trajectory')


def engine(binding=None, sampler=True):
    model, roles=target()
    opt=optimizer(model, roles)
    return NativeMuUpdateEngine(
        model, opt, binding_sha256='d'*64,
        task_cycle=('trajectory', 'other'),
        source_cursors={'trajectory':b'0', 'other':b'0'},
        sampler_generator=torch.Generator().manual_seed(391) if sampler else None,
        trajectory_objective_binding=binding)


def test_predeclared_guide_prefix_and_method_survive_exact_checkpoint_readback():
    frozen=declaration()
    trainer=engine(frozen)
    checkpoint=trainer.checkpoint_state()
    assert checkpoint['schema']=='Mora3.NativeMuUpdateState/v5'
    assert checkpoint['trajectory_objective_binding']==frozen
    assert checkpoint['steps']==0 and checkpoint['optimizer']['state']=={}
    trainer.require_trajectory_objective_binding(**frozen)
    restored=engine(frozen)
    restored.restore_state(copy.deepcopy(checkpoint))
    assert restored.checkpoint_state()['trajectory_objective_binding']==frozen
    assert restored.steps==0 and not restored.optimizer.state


@pytest.mark.parametrize('changed', [
    'trial_id','task_key','source_snapshot_sha256','policy_module_path',
    'history_module_path','prefix_binding_sha256','reward_binding_sha256',
    'sampling_method','terminal_object'])
def test_gfn_objective_cannot_change_between_updates_or_recovery(changed):
    trainer=engine(declaration())
    baseline=trainer.checkpoint_state()
    wrong=declaration()
    if changed.endswith('_sha256'):
        wrong[changed]='f'*64
    elif changed=='task_key':
        wrong[changed]='other'
    else:
        wrong[changed]=wrong[changed]+'-other'
    with pytest.raises(ValueError,match='frozen|pathspace|trajectory|binding'):
        trainer.require_trajectory_objective_binding(**wrong)
    checkpoint=copy.deepcopy(baseline)
    checkpoint['trajectory_objective_binding'][changed]=wrong[changed]
    with pytest.raises(ValueError,match='frozen|pathspace|trajectory|binding'):
        trainer.restore_state(checkpoint)
    assert trainer.checkpoint_state()['trajectory_objective_binding']==baseline['trajectory_objective_binding']
    assert trainer.steps==0 and not trainer.optimizer.state


def test_unbound_sampler_or_mutated_live_binding_is_not_an_authority():
    with pytest.raises(ValueError,match='sampler|generator'):
        engine(declaration(),sampler=False)
    trainer=engine()
    with pytest.raises(ValueError,match='frozen|pathspace|trajectory|binding'):
        trainer.require_trajectory_objective_binding(**declaration())
    bound=engine(declaration())
    before=bound.checkpoint_state()
    bound._trajectory_objective_binding['reward_binding_sha256']='f'*64
    with pytest.raises(ValueError,match='frozen|pathspace|trajectory|binding'):
        bound.checkpoint_state()
    assert bound.steps==0 and not bound.optimizer.state

def test_frozen_native_trajectory_checkpoint_reloads_in_fresh_process(tmp_path):
    import json
    import subprocess
    import sys

    trainer=engine(declaration())
    checkpoint=tmp_path/'trajectory-preregistered-native.pt'
    trainer.save_checkpoint(checkpoint)
    program="""import json,sys
from test_campaign26_trajectory_preregistration import engine,declaration
model=engine(declaration())
model.load_checkpoint(sys.argv[1])
model.require_trajectory_objective_binding(**declaration())
state=model.checkpoint_state()
print(json.dumps(dict(schema=state['schema'],steps=model.steps,
    frozen=state['trajectory_objective_binding']==declaration(),
    task=model.next_task,optimizer_empty=not bool(model.optimizer.state))))
"""
    child=subprocess.run([sys.executable,'-B','-c',program,str(checkpoint)],
        env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':'',
             'PYTHONPATH':':'.join(str(Path(__file__).resolve().parents[2]/part) for part in ('src','tests/model','tests/research')),
             'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
        text=True,capture_output=True,check=True,timeout=60)
    assert json.loads(child.stdout)==dict(schema='Mora3.NativeMuUpdateState/v5',
        steps=0,frozen=True,task='trajectory',optimizer_empty=True)
