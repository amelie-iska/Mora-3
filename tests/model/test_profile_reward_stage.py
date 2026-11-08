from dataclasses import replace
import copy
import math

import pytest
import torch

from mora3.model.frozen_profile_reward import ProfileObservedMolecules
from mora3.model.profile_gfn import ProfileGFNPolicy
from mora3.model.profile_reward_stage import ProfileRewardStageController
from test_frozen_profile_reward import native
from test_full_owner import actual_core_assets,cases
from test_full_owner_mup import AXES,members


def test_stage_barrier_exact_frozen_checkpoint_and_private_rng(tmp_path):
    model,view=native();controller=ProfileRewardStageController(model,source_group_manifest_sha256='a'*64,private_seed=108)
    group=ProfileObservedMolecules(view.input_bytes,('CCO','CCO','CCCCO'),'a'*64)
    z=torch.zeros(1,32,dtype=torch.float64);old=controller.reference.log_reward((view,),z,(group,),optimizer_update=249)
    with torch.no_grad():model.decoder.output.bias.add_(torch.arange(model.decoder.codec.size,dtype=torch.float64)*2)
    pending=controller.checkpoint(250);assert pending['phase']=='refresh_pending'
    with pytest.raises(ValueError,match='refresh'):controller.require_current(250)
    with pytest.raises(ValueError,match='stale'):controller.prepare(251)
    saved=tmp_path/'stage.pt';torch.save(pending,saved)
    expected_private=torch.randn(4,generator=controller.generator)
    rng=torch.get_rng_state().clone()
    controller.prepare(250)
    assert controller.checkpoint(250)['phase']=='current' and controller.reference.stage_optimizer_update==250
    refreshed=controller.reference.log_reward((view,),z,(group,),optimizer_update=250)
    assert not torch.equal(old,refreshed)
    staged=controller.stage_restore(torch.load(saved,weights_only=True),steps=250,staged_template=model)
    controller.apply_staged_restore(staged)
    assert controller.checkpoint(250)['phase']=='refresh_pending'
    torch.testing.assert_close(controller.reference.log_reward((view,),z,(group,),optimizer_update=249),old,rtol=0,atol=0)
    assert torch.equal(torch.randn(4,generator=controller.generator),expected_private)
    assert torch.equal(rng,torch.get_rng_state())


def test_zero_latent_gauge_preserves_relative_reward_and_has_live_policy_gradient():
    model,view=native();p=ProfileGFNPolicy(model.decoder.profiles,dimension=32,horizon=8,hidden=64,
        source_binding={'scope':'synthetic-unit-no-source-admission'},parameterization_policy='mup-profile-gfn-v1').double().eval()
    c=ProfileRewardStageController(model,source_group_manifest_sha256='a'*64,private_seed=106)
    group=ProfileObservedMolecules(view.input_bytes,('CCO','CCCCO'),'a'*64)
    rng=torch.get_rng_state().clone();private=c.generator.get_state().clone()
    with pytest.raises(ValueError):c.gauged_objective(p,(view,),(replace(group,smiles=()),),optimizer_update=0)
    assert torch.equal(c.generator.get_state(),private)
    result=c.gauged_objective(p,(view,),(group,),optimizer_update=0)
    zero=torch.zeros((1,32),dtype=torch.float64)
    source_offset=c.reference.log_reward((view,),zero,(group,),optimizer_update=0)+16*math.log(2*math.pi)
    torch.testing.assert_close(result['log_reward'],result['raw_log_reward']-source_offset,rtol=0,atol=0)
    torch.testing.assert_close(result['estimated_raw_log_partition'],result['log_z']+source_offset,rtol=0,atol=0)
    assert not result['reward_gauge_offset'].requires_grad and torch.equal(torch.get_rng_state(),rng)
    loss=result['loss'];g=torch.autograd.grad(loss,p.normalizer.weight)[0]
    assert torch.isfinite(g).all() and torch.count_nonzero(g)>0
    assert all(q.grad is None and not q.requires_grad for q in c.reference.parameters())


def test_native_update_checkpoint_contains_independent_stage_and_preflights_corruption(tmp_path,actual_core_assets):
    from mora3.model.full_owner_mup import full_owner_mup_parameter_roles,full_owner_mup_width_family
    from mora3.model.mu_optimizer import MuAdamW
    from mora3.research.primary_training import NativeMuUpdateEngine
    assets=replace(actual_core_assets,reconstruction_dimension=32,reconstruction_horizon=8,reconstruction_hidden=64)
    models=members(tmp_path,assets);model=models[0];template=model.core.parent.parent.reconstruction
    roles=full_owner_mup_parameter_roles(model,**AXES);family=full_owner_mup_width_family(*models,**AXES)
    opt=MuAdamW(model,family,roles,base_lr=0.001,weight_decay=0.,decay_aliases=(),member='base')
    controller=ProfileRewardStageController(template,source_group_manifest_sha256='a'*64,private_seed=412)
    engine=NativeMuUpdateEngine(model,opt,binding_sha256='b'*64,task_cycle=('synthetic',),
        source_cursors={'synthetic':b'0'},profile_stage_controller=controller)
    assert engine.prepare_profile_stage() is controller.reference
    tensor=template.latent_projection.weight
    original=controller.reference.latent_projection.weight.detach().clone()
    engine.update('synthetic',tensor.square().mean(),cursor_before=b'0',cursor_after=b'1')
    assert not torch.equal(tensor,original) and torch.equal(controller.reference.latent_projection.weight,original)
    path=tmp_path/'engine.pt';engine.save_checkpoint(path)
    saved=torch.load(path,weights_only=True)
    assert saved['schema']=='Mora3.NativeMuUpdateState/v2' and saved['profile_stage']['state']['phase']=='current'
    expected_private=torch.randn(5,generator=controller.generator)
    engine.update('synthetic',tensor.square().mean(),cursor_before=b'1',cursor_after=b'2')
    changed=tensor.detach().clone();rng=torch.get_rng_state().clone();private=controller.generator.get_state().clone()
    bad=copy.deepcopy(saved);bad['profile_stage']['state']['reference']['state']['output.weight'][0,0]+=1
    with pytest.raises(ValueError,match='digest'):engine.restore_state(bad)
    assert engine.steps==2 and torch.equal(tensor,changed) and torch.equal(controller.generator.get_state(),private)
    assert torch.equal(torch.get_rng_state(),rng)
    engine.load_checkpoint(path)
    assert engine.steps==1 and engine.source_cursors=={'synthetic':b'1'}
    assert torch.equal(controller.reference.latent_projection.weight,original)
    assert torch.equal(torch.randn(5,generator=controller.generator),expected_private)
    engine.update('synthetic',tensor.square().mean(),cursor_before=b'1',cursor_after=b'2')
    torch.testing.assert_close(tensor,changed,rtol=0,atol=0)
