"""Real native engine binding/cursor/checkpoint tests, with zero optimizer steps."""
import copy
import json
from dataclasses import replace
import pytest
import torch
from test_full_owner import actual_core_assets,cases
from test_hybrid_forest_teacher import setup_owner
from test_forest_flow import problem,SOURCE,COORD
from test_forest_flow_owner import AXES
from mora3.model.full_owner import SourceBoundOwner
from mora3.model.full_owner_mup import full_owner_mup_parameter_roles,full_owner_mup_width_family
from mora3.model.mu_optimizer import MuAdamW
from mora3.model.forest_training import ForestBatchIdentity
from mora3.research.primary_training import NativeMuUpdateEngine
from mora3.research.hybrid_forest_routes import (HybridForestCondition,HybridForestTrainingRoute,
    native_engine_hybrid_forest_loss,run_native_hybrid_forest_update)


def native(tmp_path,assets):
    models=[];teachers=[];rewards=[]
    for width in (8,12,16):
        p=tmp_path/str(width);p.mkdir()
        model,teacher,reward=setup_owner(p,assets,width=width)
        models.append(model);teachers.append(teacher);rewards.append(reward)
    owner=models[0];teacher=teachers[0];reward=rewards[0];r,t=problem()
    identity=ForestBatchIdentity(SOURCE,COORD,'1'*64,'2'*64,'f'*64,'test-only',
        ('original-row',),(17,),(('source-family',),),'train','synthetic_fixture','3'*64)
    condition=HybridForestCondition(r,t,('r',),torch.zeros(1,2,dtype=torch.float32),identity)
    route=HybridForestTrainingRoute(teacher,reward,(condition,),task_key='hybrid',trial_id='test-only',candidates=4,max_passes=2)
    family=full_owner_mup_width_family(*models,**AXES)
    optimizer=MuAdamW(owner,family,full_owner_mup_parameter_roles(owner,**AXES),base_lr=.001,
        weight_decay=0.,decay_aliases=(),member='base')
    engine=NativeMuUpdateEngine(owner,optimizer,binding_sha256='f'*64,task_cycle=('hybrid',),
        source_cursors={'hybrid':route.initial_cursor()},sampler_generator=torch.Generator().manual_seed(81),
        forest_objective_binding=route.binding())
    return engine,route,condition


def test_actual_native_admission_is_checked_before_route_input_or_randomness(tmp_path,actual_core_assets):
    engine,route,_=native(tmp_path,actual_core_assets);before=engine.sampler_generator.get_state().clone()
    with pytest.raises(ValueError,match='not full corpus'):
        native_engine_hybrid_forest_loss(engine,'untrusted route',trial_id='test-only')
    assert torch.equal(before,engine.sampler_generator.get_state())and engine.steps==0


def test_native_checkpoint_replays_real_loss_gradients_and_cursors(tmp_path,actual_core_assets,monkeypatch):
    engine,route,_=native(tmp_path,actual_core_assets);state=engine.checkpoint_state()
    calls=[]
    monkeypatch.setattr(SourceBoundOwner,'require_training_admission',lambda self,**kw:calls.append(kw))
    a=native_engine_hybrid_forest_loss(engine,route,trial_id='test-only')
    params=(route.teacher.topology.action_readout.weight,route.teacher.policy.forward_head.weight,
            route.teacher.topology.field.readout.weight)
    grad_a=torch.autograd.grad(a['loss'],params);after=engine.sampler_generator.get_state().clone()
    assert all(torch.isfinite(g).all()and g.abs().sum()>0 for g in grad_a)
    engine.restore_state(state)
    b=native_engine_hybrid_forest_loss(engine,route,trial_id='test-only')
    grad_b=torch.autograd.grad(b['loss'],params)
    torch.testing.assert_close(a['loss'],b['loss'],atol=0,rtol=0)
    for x,y in zip(grad_a,grad_b):torch.testing.assert_close(x,y,atol=0,rtol=0)
    assert torch.equal(after,engine.sampler_generator.get_state())
    assert engine.source_cursors['hybrid']==route.initial_cursor()and not engine.optimizer.state and engine.steps==0
    assert a['original_ordinals']==(17,)and a['original_source_rows']==1
    assert calls and calls[0]==dict(campaign_sha256='f'*64,trial_id='test-only')


def test_failure_restores_global_and_owned_rng_without_cursor_commit(tmp_path,actual_core_assets,monkeypatch):
    import mora3.research.hybrid_forest_routes as implementation
    engine,route,_=native(tmp_path,actual_core_assets)
    monkeypatch.setattr(SourceBoundOwner,'require_training_admission',lambda self,**kw:None)
    owned=engine.sampler_generator.get_state().clone();global_before=torch.get_rng_state().clone()
    def broken(*a,**k):
        torch.rand(3);torch.rand(3,generator=engine.sampler_generator)
        raise ValueError('injected pre-update failure')
    monkeypatch.setattr(implementation,'hybrid_forest_objective',broken)
    with pytest.raises(ValueError,match='pre-update'):native_engine_hybrid_forest_loss(engine,route,trial_id='test-only')
    assert torch.equal(owned,engine.sampler_generator.get_state())and torch.equal(global_before,torch.get_rng_state())
    assert engine.source_cursors['hybrid']==route.initial_cursor()and engine.steps==0


def test_update_route_hands_real_gradient_to_existing_engine_only(tmp_path,actual_core_assets,monkeypatch):
    engine,route,_=native(tmp_path,actual_core_assets);seen=[]
    monkeypatch.setattr(SourceBoundOwner,'require_training_admission',lambda self,**kw:None)
    def stop_before_optimizer(task,loss,**kw):
        seen.append((task,loss,kw));raise RuntimeError('intercept native update')
    monkeypatch.setattr(engine,'update',stop_before_optimizer)
    with pytest.raises(RuntimeError,match='intercept native update'):
        run_native_hybrid_forest_update(engine,route,trial_id='test-only')
    assert len(seen)==1 and seen[0][0]=='hybrid'and seen[0][1].requires_grad
    assert seen[0][2]['cursor_before']==route.initial_cursor()and seen[0][2]['cursor_after']!=route.initial_cursor()
    assert not engine.optimizer.state and engine.steps==0


@pytest.mark.parametrize('fault',('split','encoder','export','duplicate','weights','cursor'))
def test_source_and_frozen_protocol_refusals(tmp_path,actual_core_assets,fault):
    engine,route,c=native(tmp_path,actual_core_assets)
    if fault=='split':c=replace(c,identity=replace(c.identity,split='test'))
    elif fault=='encoder':c=replace(c,identity=replace(c.identity,encoder_sha256='9'*64))
    elif fault=='export':c=replace(c,identity=replace(c.identity,export_sha256='9'*64))
    elif fault=='weights':
        route.student_weight=2.
        with pytest.raises(ValueError):route.binding()
        return
    elif fault=='cursor':
        saved=engine.checkpoint_state();saved['source_cursors']['hybrid']=b'{}'
        before=engine.model.forest_flow.readout.weight.detach().clone()
        saved['model']['forest_flow.readout.weight'].add_(5.)
        with pytest.raises(ValueError):engine.restore_state(saved)
        assert torch.equal(before,engine.model.forest_flow.readout.weight)
        return
    with pytest.raises(ValueError):
        HybridForestTrainingRoute(route.teacher,route.reward,(c,c)if fault=='duplicate'else(c,),
            task_key='hybrid',trial_id='test-only',candidates=4)
