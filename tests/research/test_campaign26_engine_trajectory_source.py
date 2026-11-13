"""Test-only native-owner GFN sampler-to-engine binding; no optimizer updates."""
from hashlib import sha256
import json

import pytest
import torch

from test_full_owner import actual_core_assets, cases
from test_full_owner_mup import AXES, members
from test_molecule_profile_model import view
from mora3.model.full_owner import SourceBoundOwner
from mora3.model.full_owner_mup import full_owner_mup_parameter_roles, full_owner_mup_width_family
from mora3.model.mu_optimizer import MuAdamW
from mora3.reasoning.trajectory_gfn import PathSpaceReward
from mora3.reasoning.trajectory_history import NativePrefixContext
from mora3.research.primary_training import NativeMuUpdateEngine, _trajectory_freeze_wire


def _test_only_declaration(history, reward_sha, trial_id, task_key):
    # Test-only synthetic binding. Real campaign uses preregistered immutable owner inputs.
    return dict(schema='Mora3.NativePathspaceBinding/v1',
        trial_id=trial_id,task_key=task_key,source_snapshot_sha256='a'*64,
        policy_module_path='core.parent.parent.profile_gfn',
        history_module_path='core.parent.parent.geometry_fusion.geometry',
        prefix_binding_sha256=sha256(json.dumps(history.binding(),sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        reward_binding_sha256=reward_sha,
        sampling_method='native-unpruned-history-conditioned-forced-horizon',
        terminal_object='entire-ordered-reasoning-trajectory')

import mora3.reasoning.trajectory_gfn as implementation


def test_registered_native_owner_engine_consumes_only_its_saved_sampler(tmp_path, actual_core_assets, monkeypatch):
    assert hasattr(implementation, 'native_engine_pathspace_loss'), 'Native engine has no owned on-policy GFN objective entry'
    model_family=members(tmp_path,actual_core_assets)
    owner=model_family[-1]
    family=full_owner_mup_width_family(*model_family,**AXES)
    optimizer=MuAdamW(owner,family,full_owner_mup_parameter_roles(owner,**AXES),
        base_lr=.001,weight_decay=0.,decay_aliases=())
    generator=torch.Generator(device='cpu').manual_seed(4821)
    policy=owner.core.parent.parent.profile_gfn
    geometry=owner.core.parent.parent.geometry_fusion.geometry
    history=NativePrefixContext(policy,geometry,source_context_sha256='a'*64,coefficient=.2)
    reference=torch.tensor([[[0.,0.],[1.,0.],[1.,1.],[2.,1.]]],dtype=next(policy.parameters()).dtype)
    reward=PathSpaceReward(reference,source_sha256='b'*64,encoder_sha256='c'*64,
        alignment_sha256='d'*64,group_policy_sha256='e'*64,
        guide_groups=(('guide:1',),),query_groups=(('query:2',),),scale=1.,
        metric_weight=.2,topology_weight=.1,evidence_kind='synthetic_fixture')
    digest=sha256(json.dumps(reward.binding(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    options=dict(policy_module_path='core.parent.parent.profile_gfn',
        history_module_path='core.parent.parent.geometry_fusion.geometry',
        history_context=history,views=(view('profile_to_molecule'),),reward=reward,
        reward_binding_sha256=digest,trial_id='test-only-nonadmitted',
        task_key='trajectory',cursor_before=b'source-start')
    engine=NativeMuUpdateEngine(owner,optimizer,binding_sha256='f'*64,
        task_cycle=('trajectory',),source_cursors={'trajectory':b'source-start'},
        sampler_generator=generator,trajectory_objective_binding=
            _test_only_declaration(history,digest,options['trial_id'],'trajectory'))
    before=generator.get_state().clone()
    checkpoint=engine.checkpoint_state()
    # The actual source-bound owner refuses before RNG sampling or target scoring.
    with pytest.raises(ValueError):implementation.native_engine_pathspace_loss(engine,**options)
    assert torch.equal(generator.get_state(),before) and engine.steps==0 and not optimizer.state
    # Synthetic ONLY: exercise the adapter with admission stubbed inside test scope.
    calls=[]
    monkeypatch.setattr(SourceBoundOwner,'require_training_admission',lambda self,**kwargs:calls.append(kwargs))
    with pytest.raises(ValueError,match='reward|guide|binding'):
        implementation.native_engine_pathspace_loss(engine,**(options|{'reward_binding_sha256':'0'*64}))
    assert torch.equal(generator.get_state(),before)
    with pytest.raises(ValueError,match='cursor|task'):
        implementation.native_engine_pathspace_loss(engine,**(options|{'cursor_before':b'other'}))
    assert torch.equal(generator.get_state(),before)
    batch,out=implementation.native_engine_pathspace_loss(engine,**options)
    assert len(calls)>=2 and any(x.get('campaign_sha256')=='f'*64 for x in calls)
    assert batch.sampler=='native-unpruned-history-conditioned-forced-horizon'
    assert out['terminal_object']=='entire-ordered-reasoning-trajectory'
    assert out['optimizer_updates']==0 and out['loss'].requires_grad and torch.isfinite(out['loss'])
    grad=torch.autograd.grad(out['loss'],policy.forward_head.weight)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    assert not torch.equal(generator.get_state(),before)
    assert engine.steps==0 and engine.source_cursors['trajectory']==b'source-start' and not optimizer.state
    assert all(p.grad is None for p in owner.parameters())
    # The generator is checkpointed by the actual MuAdamW owner; reloading a
    # pre-sampling checkpoint produces exactly the same full native paths.
    engine.restore_state(checkpoint)
    second,again=implementation.native_engine_pathspace_loss(engine,**options)
    torch.testing.assert_close(batch.paths['states'],second.paths['states'],rtol=0,atol=0)
    torch.testing.assert_close(out['loss'],again['loss'],rtol=0,atol=0)
    engine.sampler_generator=torch.Generator().set_state(generator.get_state())
    with pytest.raises(ValueError,match='sampler|generator'):
        implementation.native_engine_pathspace_loss(engine,**options)
    assert engine.steps==0 and not optimizer.state


def test_non_native_engine_is_never_a_sampler_authority():
    assert hasattr(implementation,'native_engine_pathspace_loss')
    with pytest.raises(ValueError,match='engine'):
        implementation.native_engine_pathspace_loss(object(),policy_module_path='non-native',
            history_module_path='other',history_context=None,views=(),reward=None,
            reward_binding_sha256='a'*64,trial_id='test',task_key='test',cursor_before=b'0')


def test_pure_native_pathspace_engine_boundary_test_only_registered_shell(monkeypatch):
    """Synthetic integration of real profile/TokenGT/GFN kernels, NOT an admitted owner.

    The missing original LINCS fixture above prevents genuine full-owner
    execution; this shell checks the new interface with no invented corpus pin,
    optimizer, training record, or full-model qualification.
    """
    from torch import nn
    from test_trajectory_history_conditioning import setup as history_setup
    policy, geometry, history, views = history_setup()
    owner = object.__new__(SourceBoundOwner)
    nn.Module.__init__(owner)
    owner.core = nn.Module()
    owner.core.parent = nn.Module()
    owner.core.parent.parent = nn.Module()
    owner.core.parent.parent.profile_gfn = policy
    owner.core.parent.parent.geometry_fusion = nn.Module()
    owner.core.parent.parent.geometry_fusion.geometry = geometry
    owner.get_extra_state = lambda: {'source_snapshot_sha256': 'a'*64}
    calls=[]
    owner.require_training_admission = lambda **scope: calls.append(scope)
    engine = object.__new__(NativeMuUpdateEngine)
    engine.model=owner
    engine.task_cycle=('trajectory',)
    engine.source_cursors={'trajectory':b'original-start'}
    engine.steps=0
    engine.binding_sha256='f'*64
    engine.sampler_generator=torch.Generator().manual_seed(731)
    engine._sampler_owner=(engine.sampler_generator,)
    engine._sampler_device='cpu'
    reference=torch.tensor([[[0.,0.],[1.,0.],[1.,1.],[2.,1.]]],dtype=next(policy.parameters()).dtype)
    guide=PathSpaceReward(reference,source_sha256='b'*64,encoder_sha256='c'*64,
        alignment_sha256='d'*64,group_policy_sha256='e'*64,
        guide_groups=(('source:train:1',),),query_groups=(('source:query:2',),),
        scale=1.,metric_weight=.2,topology_weight=.1,evidence_kind='synthetic_fixture')
    guide_sha=sha256(json.dumps(guide.binding(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    params=dict(policy_module_path='core.parent.parent.profile_gfn',
        history_module_path='core.parent.parent.geometry_fusion.geometry',
        history_context=history,views=views,reward=guide,reward_binding_sha256=guide_sha,
        trial_id='test-only',task_key='trajectory',cursor_before=b'original-start')
    # The shell deliberately bypasses the true optimizer constructor and remains test-only.
    engine._trajectory_objective_binding=_test_only_declaration(
        history,guide_sha,'test-only','trajectory')
    engine._trajectory_frozen_wire=_trajectory_freeze_wire(engine._trajectory_objective_binding)
    before=engine.sampler_generator.get_state().clone()
    with pytest.raises(ValueError,match='reward|guide|binding'):
        implementation.native_engine_pathspace_loss(engine,**(params|{'reward_binding_sha256':'0'*64}))
    assert torch.equal(engine.sampler_generator.get_state(),before)
    with pytest.raises(ValueError,match='cursor|task'):
        implementation.native_engine_pathspace_loss(engine,**(params|{'cursor_before':b'different'}))
    assert torch.equal(engine.sampler_generator.get_state(),before)
    with monkeypatch.context() as patch:
        def fail_after_draw(*a,**k):raise ValueError('synthetic scoring failed after drawing')
        patch.setattr(implementation,'native_pathspace_objective',fail_after_draw)
        with pytest.raises(ValueError,match='after drawing'):
            implementation.native_engine_pathspace_loss(engine,**params)
    assert torch.equal(engine.sampler_generator.get_state(),before)
    batch,result=implementation.native_engine_pathspace_loss(engine,**params)
    # A self-consistent *new* guide with a matching *new* hash must still be
    # rejected: only the original per-trial predeclared guide can draw.
    other_guide=PathSpaceReward(reference,source_sha256='b'*64,encoder_sha256='c'*64,
        alignment_sha256='d'*64,group_policy_sha256='e'*64,
        guide_groups=(('source:train:1',),),query_groups=(('source:query:2',),),
        scale=1.,metric_weight=.4,topology_weight=.1,evidence_kind='synthetic_fixture')
    other_sha=sha256(json.dumps(other_guide.binding(),sort_keys=True,separators=(',',':'),
        allow_nan=False).encode()).hexdigest()
    frozen_sampler=engine.sampler_generator.get_state().clone()
    with pytest.raises(ValueError,match='frozen|trajectory|binding'):
        implementation.native_engine_pathspace_loss(engine,**(
            params|{'reward':other_guide,'reward_binding_sha256':other_sha}))
    with pytest.raises(ValueError,match='frozen|trajectory|binding'):
        implementation.native_engine_pathspace_loss(engine,**(
            params|{'trial_id':'test-only-different-trial'}))
    assert torch.equal(frozen_sampler,engine.sampler_generator.get_state())
    assert batch.sampler=='native-unpruned-history-conditioned-forced-horizon'
    assert batch.source_context_sha256=='a'*64
    assert result['terminal_object']=='entire-ordered-reasoning-trajectory'
    assert torch.isfinite(result['loss']) and result['loss'].requires_grad
    assert result['optimizer_updates']==0 and result['training_admission']is False
    assert any(c.get('campaign_sha256')=='f'*64 for c in calls)
    assert engine.steps==0 and engine.source_cursors['trajectory']==b'original-start'
    sampled=engine.sampler_generator.get_state().clone()
    assert not torch.equal(sampled,before)
    engine.sampler_generator.set_state(before)
    replay,loss=implementation.native_engine_pathspace_loss(engine,**params)
    torch.testing.assert_close(batch.paths['states'],replay.paths['states'],atol=0,rtol=0)
    torch.testing.assert_close(result['loss'],loss['loss'],atol=0,rtol=0)
    engine.sampler_generator=torch.Generator().set_state(sampled)
    with pytest.raises(ValueError,match='sampler|generator'):
        implementation.native_engine_pathspace_loss(engine,**params)
    assert engine.steps==0
