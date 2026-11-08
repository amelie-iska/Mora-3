import copy
import math

import pytest
import torch

from mora3.model.mu_optimizer import MuAdamW
from mora3.model.parameterization import Axis, ParameterRole, parameter_ledger, width_family
from mora3.research.mup_campaign_schedule import NativeMuWarmupCosine, warmup_cosine_factor
from mora3.research.primary_training import NativeMuUpdateEngine
from test_mu_optimizer import fixture, optimizer, target
from test_primary_training import advance, equal, seed


def engine(*, updates=30, clip=.75):
    model, roles = target()
    opt = optimizer(model, roles)
    scheduler = NativeMuWarmupCosine(opt, total_updates=updates)
    return NativeMuUpdateEngine(
        model, opt, binding_sha256='a'*64, task_cycle=('one', 'two'),
        source_cursors={'one': b'0', 'two': b'0'}, scheduler=scheduler, gradient_clip_max_norm=clip,
    )


def test_exact_warmup_cosine_coordinates_and_common_native_lr_ratios():
    e = engine()
    base = [.01 if group['mup_aliases'] == ('hidden',) else .02 for group in e.optimizer.param_groups]
    for completed in range(31):
        nxt = completed + 1
        if nxt <= 3:
            expected = nxt / 3
        else:
            expected = math.cos(math.pi / 2 * (nxt-3)/28) ** 2
        assert warmup_cosine_factor(30, completed) == pytest.approx(expected, abs=1e-15)
        assert e.scheduler.last_epoch == completed
        assert e.scheduler.get_last_lr() == pytest.approx([p * expected for p in base], abs=1e-17)
        if completed < 30:
            advance(e)
    assert e.scheduler.get_last_lr() == [0.] * len(base)
    before = e.checkpoint_state()
    with pytest.raises(ValueError, match='horizon'):
        e.update(e.next_task, e.model.input.sum(), cursor_before=e.source_cursors[e.next_task], cursor_after=b'excess')
    equal(e.checkpoint_state(), before)
    for coordinates in ((29, 0), (True, 0), (30, True), (30, 31), (30, -1)):
        with pytest.raises(ValueError):
            warmup_cosine_factor(*coordinates)


def test_global_clip_matches_independent_gradient_equation():
    seed()
    e = engine()
    active = (e.model.input, e.model.hidden, e.model.output)
    loss = sum((p * (i+2)).sum() for i, p in enumerate(active))
    raw_norm = math.sqrt(sum(p.numel() * (i+2)**2 for i, p in enumerate(active)))
    expected = min(.75 / (raw_norm + 1e-6), 1.)
    observed = []
    original = e.optimizer.step
    def inspect():
        observed.extend(p.grad.detach().clone() for p in active)
        return original()
    e.optimizer.step = inspect
    stats = e.update('one', loss, cursor_before=b'0', cursor_after=b'1')
    for i, gradient in enumerate(observed):
        torch.testing.assert_close(gradient, torch.full_like(gradient, (i+2)*expected), atol=1e-14, rtol=1e-14)
    assert stats['gradient_norm_before_clip'] == pytest.approx(raw_norm)
    assert e.steps == 1 and all(p.grad is None for p in e.model.parameters())


def test_checkpoint_cosine_clip_continuation_matches_exact_complete_rng_and_optimizer(tmp_path):
    seed()
    reference = engine()
    expected = [advance(reference) for _ in range(9)]
    seed()
    first = engine()
    actual = [advance(first) for _ in range(2)]
    path = tmp_path/'cosine.pt'
    first.save_checkpoint(path)
    resumed = engine()
    torch.rand(7)
    resumed.load_checkpoint(path)
    actual.extend(advance(resumed) for _ in range(7))
    assert actual == expected
    equal(resumed.checkpoint_state(), reference.checkpoint_state())
    assert resumed.checkpoint_state()['schema'] == 'Mora3.NativeMuUpdateState/v3'
    assert resumed.scheduler.optimizer is resumed.optimizer


@pytest.mark.parametrize('tamper', ('clip', 'clock', 'schedule', 'all_rates'))
def test_bad_schedule_or_clip_restore_refuses_before_live_mutation(tamper):
    seed()
    e = engine()
    advance(e)
    before = e.checkpoint_state()
    bad = copy.deepcopy(before)
    bad['model']['input'].add_(10)
    if tamper == 'clip':
        bad['update_policy']['gradient_clip_max_norm'] = 2.
    elif tamper == 'clock':
        bad['scheduler']['completed_updates'] += 1
    elif tamper == 'schedule':
        bad['scheduler']['total_updates'] += 10
    else:
        bad['scheduler']['last_lr'] = tuple(v * .8 for v in bad['scheduler']['last_lr'])
        for group in bad['optimizer']['param_groups']:
            group['lr'] *= .8
        bad['optimizer']['mora3_mu_adamw']['scheduled_master_lr'] *= .8
    with pytest.raises(ValueError, match='schedule|policy'):
        e.restore_state(bad)
    equal(e.checkpoint_state(), before)


def test_schedule_and_clip_policy_preflights_invalid_or_already_owned_setup():
    m, roles = target()
    opt = optimizer(m, roles)
    before = opt.state_dict()
    with pytest.raises(ValueError):
        NativeMuWarmupCosine(opt, total_updates=15)
    equal(opt.state_dict(),before)
    schedule = NativeMuWarmupCosine(opt, total_updates=10)
    scheduled = opt.state_dict()
    with pytest.raises(ValueError):
        NativeMuWarmupCosine(opt, total_updates=10)
    equal(opt.state_dict(),scheduled)
    for norm in (False, float('inf'), 0., -1.):
        with pytest.raises(ValueError):
            NativeMuUpdateEngine(m,opt,binding_sha256='a'*64,task_cycle=('x',),
                source_cursors={'x':b'0'},scheduler=schedule,gradient_clip_max_norm=norm)
    equal(opt.state_dict(),scheduled)


def test_cosine_profile_pending_checkpoint_and_private_rng_resume(tmp_path):
    from mora3.model.profile_reward_stage import ProfileRewardStageController
    from test_frozen_profile_reward import native

    rows = []
    for width, second in ((4,6), (6,12), (8,18)):
        model, roles = fixture(width, second)
        model.template, _ = native()
        assert tuple(dict(model.template.named_parameters())) == ('latent_projection.weight',)
        roles['template.latent_projection.weight'] = ParameterRole('input',
            (Axis('finite','d'*64),Axis('finite','e'*64)))
        rows.append((model,roles))
    family = width_family(*(parameter_ledger(model,roles) for model,roles in rows))
    model,roles = rows[-1]
    opt = MuAdamW(model,family,roles,base_lr=.02,weight_decay=0.,decay_aliases=())
    scheduler = NativeMuWarmupCosine(opt,total_updates=260)
    controller = ProfileRewardStageController(model.template,source_group_manifest_sha256='a'*64,private_seed=99)
    e = NativeMuUpdateEngine(model,opt,binding_sha256='b'*64,task_cycle=('p',),source_cursors={'p':b'0'},
                             scheduler=scheduler,gradient_clip_max_norm=.5,profile_stage_controller=controller)
    parameter = model.template.latent_projection.weight
    for step in range(250):
        e.update('p',parameter.square().mean(),cursor_before=str(step).encode(),cursor_after=str(step+1).encode())
    path = tmp_path/'pending.pt'
    e.save_checkpoint(path)
    state = e.checkpoint_state()
    assert state['schema']=='Mora3.NativeMuUpdateState/v3'
    assert state['profile_stage']['state']['phase']=='refresh_pending'
    expected_private = torch.randn(3,generator=controller.generator)
    with pytest.raises(ValueError,match='refresh'):
        e.update('p',parameter.square().mean(),cursor_before=b'250',cursor_after=b'251')
    e.prepare_profile_stage()
    e.update('p',parameter.square().mean(),cursor_before=b'250',cursor_after=b'251')
    after = parameter.detach().clone()
    e.load_checkpoint(path)
    assert e.steps==250 and e.scheduler.last_epoch==250
    assert e.checkpoint_state()['profile_stage']['state']['phase']=='refresh_pending'
    assert torch.equal(torch.randn(3,generator=controller.generator),expected_private)
    e.prepare_profile_stage()
    e.update('p',parameter.square().mean(),cursor_before=b'250',cursor_after=b'251')
    torch.testing.assert_close(parameter,after,rtol=0,atol=0)


def test_checkpoint_requires_schedule_position_and_live_policy_to_match_engine():
    e = engine()
    e.scheduler.step()
    with pytest.raises(ValueError,match='schedule'):
        e.checkpoint_state()
    assert e.steps == 0 and not e.optimizer.state
    e = engine()
    before = e.optimizer.state_dict()
    e._gradient_clip_max_norm = 99.
    with pytest.raises(ValueError,match='policy'):
        e.update('one',e.model.input.sum(),cursor_before=b'0',cursor_after=b'1')
    assert e.steps == 0 and all(p.grad is None for p in e.model.parameters())
    equal(e.optimizer.state_dict(),before)
