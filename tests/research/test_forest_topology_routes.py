"""Versioned topology objective recovery; synthetic labels, zero optimizer steps."""
from dataclasses import replace
import copy
import json
import pytest
import torch
from test_full_owner import actual_core_assets,cases
from test_forest_flow_owner import decoder_assets,AXES
from test_forest_training_routes import cast32
from test_forest_topology_training import enabled
from mora3.model.full_owner import construct_primary_owner,PrimaryConfiguration,SourceBoundOwner
from mora3.model.full_owner_mup import full_owner_mup_width_family,full_owner_mup_parameter_roles
from mora3.model.forest_quality import ForestQualitySpec
from mora3.model.forest_analogy import ForestAnalogySpec
from mora3.model.forest_topology import ForestTopologySpec
from mora3.reasoning.forest_topology import TopologyBounds
from mora3.model.forest_token_decoder import prepare_forest_token_targets
from mora3.model.mu_optimizer import MuAdamW
from mora3.research.primary_training import NativeMuUpdateEngine,_forest_freeze_wire
from mora3.research.forest_training_routes import ForestTrainingRoute,heads_from_owner,native_engine_forest_loss


def native_topology(tmp_path,assets):
    tmp_path.mkdir(parents=True,exist_ok=True)
    source=replace(decoder_assets(tmp_path,assets),forest_quality_spec=ForestQualitySpec(('order','constraint'),'f'*64,(1.,2.)),
        forest_analogy_spec=ForestAnalogySpec(4,'d'*64,max_memory=16),
        forest_topology_spec=ForestTopologySpec(TopologyBounds(8,7,7),'b'*64))
    owners=[construct_primary_owner(PrimaryConfiguration(w,8,'a'*64),source)for w in(8,12,16)]
    h=heads_from_owner(owners[0]);_,spec,batch=enabled();batch=cast32(batch)
    target,mask=prepare_forest_token_targets(h.decoder.axis,(batch.template,)*2,
        ((None,b'abc',b'hi',b'x',None),)*2,max_slots=8)
    batch=replace(batch,tokens=replace(batch.tokens,values=target,available=mask))
    spec=replace(spec,heads_sha256=h.sha256)
    route=ForestTrainingRoute(h,spec,(batch,),task_key='forest',trial_id='test-only-topology',max_passes=2)
    family=full_owner_mup_width_family(*owners,**AXES)
    opt=MuAdamW(owners[0],family,full_owner_mup_parameter_roles(owners[0],**AXES),
        base_lr=.001,weight_decay=0.,decay_aliases=(),member='base')
    engine=NativeMuUpdateEngine(owners[0],opt,binding_sha256='f'*64,task_cycle=('forest',),
        source_cursors={'forest':route.initial_cursor()},sampler_generator=torch.Generator().manual_seed(731),
        forest_objective_binding=route.binding())
    return engine,route


def test_new_route_versions_only_match_their_own_semantics():
    h,s,b=enabled();r=ForestTrainingRoute(h,s,(b,),task_key='f',trial_id='test')
    v=r.binding();assert v['schema']=='Mora3.NativeForestObjectiveBinding/v2'
    assert json.loads(_forest_freeze_wire(v))==v
    for changed in(dict(v,schema='Mora3.NativeForestObjectiveBinding/v1'),
                   dict(v,skeleton_policy='source-declared-fixed-skeleton')):
        with pytest.raises(ValueError):_forest_freeze_wire(changed)


def test_new_objective_checkpoint_real_owner_refusal_and_zero_update_replay(tmp_path,actual_core_assets,monkeypatch):
    engine,route=native_topology(tmp_path,actual_core_assets);before=engine.checkpoint_state()
    assert before['schema']=='Mora3.NativeMuUpdateState/v6'
    assert before['forest_objective_binding']['schema']=='Mora3.NativeForestObjectiveBinding/v2'
    with pytest.raises(ValueError,match='not full corpus'):
        native_engine_forest_loss(engine,route,trial_id='test-only-topology')
    assert torch.equal(before['sampler_rng']['state'],engine.sampler_generator.get_state())
    # Test-only admission interception exercises math/recovery, not production authority.
    monkeypatch.setattr(SourceBoundOwner,'require_training_admission',lambda self,**kw:None)
    first=native_engine_forest_loss(engine,route,trial_id='test-only-topology')
    after=engine.sampler_generator.get_state().clone()
    assert first['counts']['topology']==12 and first['component_losses']['topology'].requires_grad
    gradient=torch.autograd.grad(first['loss'],engine.model.forest_topology.action_readout.weight)[0]
    assert torch.isfinite(gradient).all()and gradient.abs().sum()>0
    engine.restore_state(before)
    second=native_engine_forest_loss(engine,route,trial_id='test-only-topology')
    torch.testing.assert_close(first['loss'],second['loss'],atol=0,rtol=0)
    assert torch.equal(after,engine.sampler_generator.get_state())
    assert engine.steps==0 and not engine.optimizer.state
    assert engine.source_cursors['forest']==route.initial_cursor()
    bad=copy.deepcopy(before);bad['forest_objective_binding']['schema']='Mora3.NativeForestObjectiveBinding/v1'
    bad['model']['forest_topology.action_readout.weight'].add_(50)
    weights=engine.model.forest_topology.action_readout.weight.clone()
    with pytest.raises(ValueError):engine.restore_state(bad)
    assert torch.equal(weights,engine.model.forest_topology.action_readout.weight)
