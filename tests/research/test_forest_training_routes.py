"""Native engine/source cursor integration. No optimizer update is permitted."""
from dataclasses import fields,is_dataclass,replace
import copy
import json
import pytest
import torch
from test_forest_training import fixture
from test_forest_flow_owner import decoder_assets,AXES
from test_full_owner import actual_core_assets,cases
from mora3.model.full_owner import PrimaryConfiguration,SourceBoundOwner,construct_primary_owner
from mora3.model.full_owner_mup import full_owner_mup_parameter_roles,full_owner_mup_width_family
from mora3.model.mu_optimizer import MuAdamW
from mora3.model.forest_training import ForestTrainingHeads,forest_batch_sha256
from mora3.model.forest_analogy import ForestAnalogySpec
from mora3.model.forest_quality import ForestQualitySpec
from mora3.model.forest_token_decoder import prepare_forest_token_targets
from mora3.research.primary_training import NativeMuUpdateEngine
from mora3.research.forest_training_routes import (ForestTrainingRoute,heads_from_owner,
    native_engine_forest_loss,run_native_forest_update)


def cast32(x):
    if type(x)is torch.Tensor:return x.detach().float()if x.is_floating_point()else x.detach().clone()
    if is_dataclass(x):return type(x)(**{f.name:cast32(getattr(x,f.name))for f in fields(x)})
    if type(x)is tuple:return tuple(cast32(v)for v in x)
    return x


def native(tmp_path,assets):
    tmp_path.mkdir(parents=True,exist_ok=True)
    source=replace(decoder_assets(tmp_path,assets),forest_quality_spec=ForestQualitySpec(('order','constraint'),'f'*64,(1.,2.)),
                   forest_analogy_spec=ForestAnalogySpec(4,'d'*64,max_memory=16))
    models=[construct_primary_owner(PrimaryConfiguration(w,8,'a'*64),source)for w in(8,12,16)]
    owner=models[0];h=heads_from_owner(owner);_,spec,batch=fixture();batch=cast32(batch)
    target,mask=prepare_forest_token_targets(h.decoder.axis,(batch.template,)*2,
        ((None,b'abc',b'hi',b'x',None),)*2,max_slots=8)
    batch=replace(batch,tokens=replace(batch.tokens,values=target,available=mask))
    spec=replace(spec,heads_sha256=h.sha256)
    route=ForestTrainingRoute(h,spec,(batch,),task_key='forest',trial_id='test-only-forest',max_passes=2)
    family=full_owner_mup_width_family(*models,**AXES)
    opt=MuAdamW(owner,family,full_owner_mup_parameter_roles(owner,**AXES),base_lr=.001,weight_decay=0.,decay_aliases=(),member='base')
    engine=NativeMuUpdateEngine(owner,opt,binding_sha256='f'*64,task_cycle=('forest',),
        source_cursors={'forest':route.initial_cursor()},sampler_generator=torch.Generator().manual_seed(731),
        forest_objective_binding=route.binding())
    return engine,route


def test_route_freezes_complete_source_and_preserves_original_ordinals():
    h,s,b=fixture();route=ForestTrainingRoute(h,s,(b,),task_key='forest',trial_id='synthetic',max_passes=2)
    before=route.initial_cursor();first=route.select(h,before,max_windows=1)
    assert first['batch'].identity.original_ordinals==(17,29)
    assert first['batch'].identity.record_ids==('example0','example1')
    original=forest_batch_sha256(first['batch'])
    b.flow.values.add_(100)
    assert forest_batch_sha256(route.select(h,before,max_windows=1)['batch'])==original
    first['batch'].flow.values.add_(200)
    assert forest_batch_sha256(route.select(h,before,max_windows=1)['batch'])==original
    second=route.select(h,first['cursor_after'],max_windows=1)
    assert second['status']=='prepared'and second['cursor_after']!=first['cursor_after']
    assert route.select(h,second['cursor_after'],max_windows=1)['status']=='source-exhausted'
    for wrong in (b'{}',before.replace(b'"position":0',b'"position":true'),before+b' '):
        with pytest.raises(ValueError):route.select(h,wrong,max_windows=1)


def test_route_refuses_duplicate_rows_and_capacity_or_contents_drift():
    h,s,b=fixture()
    with pytest.raises(ValueError):ForestTrainingRoute(h,s,(b,b),task_key='f',trial_id='t')
    with pytest.raises(ValueError):ForestTrainingRoute(h,s,(b,),task_key='f',trial_id='t',max_resident_bytes=1)
    route=ForestTrainingRoute(h,s,(b,),task_key='f',trial_id='t');cursor=route.initial_cursor()
    route._batches[0].flow.values.add_(1)
    with pytest.raises(ValueError):route.select(h,cursor,max_windows=1)


def test_route_skips_explicit_unavailable_batches_without_committing_cursor():
    h,s,b=fixture();empty=replace(b,flow=None,tokens=None,quality=None,analogy=None)
    second=replace(b,identity=replace(b.identity,record_ids=('example2','example3'),original_ordinals=(30,31)))
    route=ForestTrainingRoute(h,s,(empty,second),task_key='f',trial_id='t')
    c=route.initial_cursor();scan=route.select(h,c,max_windows=1)
    assert scan['status']=='unavailable-scan-limit'and scan['cursor_after']==c
    chosen=route.select(h,c,max_windows=2)
    assert chosen['status']=='prepared'and chosen['unavailable_batches']==1
    assert chosen['batch'].identity.original_ordinals==(30,31)


def test_real_owner_admission_precedes_source_read_and_random_draw(tmp_path,actual_core_assets,monkeypatch):
    engine,route=native(tmp_path,actual_core_assets);before=engine.sampler_generator.get_state().clone()
    def forbidden(*a,**k):raise AssertionError('source read before real admission')
    monkeypatch.setattr(route,'select',forbidden)
    with pytest.raises(ValueError,match='not full corpus'):
        native_engine_forest_loss(engine,route,trial_id='test-only-forest')
    assert torch.equal(before,engine.sampler_generator.get_state())and engine.steps==0
    assert not engine.optimizer.state


def test_actual_native_checkpoint_replays_loss_gradients_and_sampler(tmp_path,actual_core_assets,monkeypatch):
    engine,route=native(tmp_path,actual_core_assets);saved=engine.checkpoint_state()
    assert saved['schema']=='Mora3.NativeMuUpdateState/v6'
    calls=[]
    monkeypatch.setattr(SourceBoundOwner,'require_training_admission',lambda self,**kw:calls.append(kw))
    a=native_engine_forest_loss(engine,route,trial_id='test-only-forest')
    after=engine.sampler_generator.get_state().clone()
    params=(engine.model.forest_flow.readout.weight,engine.model.forest_decoder.readout.weight,
            engine.model.forest_quality.node_readout.weight,engine.model.forest_analogy.query_readout.weight)
    grads=torch.autograd.grad(a['loss'],params)
    assert all(torch.isfinite(g).all()and g.abs().sum()>0 for g in grads)
    assert not torch.equal(saved['sampler_rng']['state'],after)
    assert engine.steps==0 and engine.source_cursors['forest']==route.initial_cursor()and not engine.optimizer.state
    engine.restore_state(saved)
    b=native_engine_forest_loss(engine,route,trial_id='test-only-forest')
    torch.testing.assert_close(a['loss'],b['loss'],atol=0,rtol=0)
    assert torch.equal(after,engine.sampler_generator.get_state())
    assert any(c.get('campaign_sha256')=='f'*64 and c.get('trial_id')=='test-only-forest'for c in calls)
    assert a['training_admission']is False and a['counts']['flow']==16


def test_engine_frozen_binding_and_rng_rollback_before_optimizer(tmp_path,actual_core_assets,monkeypatch):
    import mora3.research.forest_training_routes as implementation
    engine,route=native(tmp_path,actual_core_assets)
    monkeypatch.setattr(SourceBoundOwner,'require_training_admission',lambda self,**kw:None)
    before=engine.sampler_generator.get_state().clone();global_before=torch.get_rng_state().clone()
    def broken(*a,**k):
        torch.rand(3)
        raise ValueError('deliberate post-draw objective failure')
    with monkeypatch.context()as patch:
        patch.setattr(implementation,'forest_joint_loss',broken)
        with pytest.raises(ValueError,match='post-draw'):native_engine_forest_loss(engine,route,trial_id='test-only-forest')
    assert torch.equal(before,engine.sampler_generator.get_state())and torch.equal(global_before,torch.get_rng_state())
    with pytest.raises(ValueError):native_engine_forest_loss(engine,route,trial_id='other-trial')
    changed=copy.deepcopy(engine.checkpoint_state());changed['forest_objective_binding']['route_sha256']='0'*64
    with pytest.raises(ValueError):engine.restore_state(changed)
    assert torch.equal(before,engine.sampler_generator.get_state())and engine.steps==0
    engine.sampler_generator=torch.Generator().set_state(before)
    with pytest.raises(ValueError,match='sampler|generator'):native_engine_forest_loss(engine,route,trial_id='test-only-forest')


def test_update_handoff_reaches_real_engine_without_test_optimizer_execution(tmp_path,actual_core_assets,monkeypatch):
    engine,route=native(tmp_path,actual_core_assets)
    monkeypatch.setattr(SourceBoundOwner,'require_training_admission',lambda self,**kw:None)
    received=[]
    def intercept(task,loss,**kw):
        received.append((task,loss,kw));raise RuntimeError('test intercept before update')
    monkeypatch.setattr(engine,'update',intercept)
    before=engine.sampler_generator.get_state().clone()
    with pytest.raises(RuntimeError,match='before update'):
        run_native_forest_update(engine,route,trial_id='test-only-forest')
    assert len(received)==1 and received[0][0]=='forest'
    assert received[0][1].requires_grad and received[0][2]['cursor_before']==route.initial_cursor()
    assert received[0][2]['cursor_after']!=route.initial_cursor()
    # Once handed to update, the wrapper does not pretend it can roll back optimizer state.
    assert not torch.equal(before,engine.sampler_generator.get_state())
    assert engine.steps==0 and not engine.optimizer.state


@pytest.mark.parametrize('fault',('wrong-route','past-end','bool-position','noncanonical'))
def test_corrupt_forest_cursor_refuses_restore_before_parameter_copy(tmp_path,actual_core_assets,fault):
    engine,route=native(tmp_path,actual_core_assets);saved=engine.checkpoint_state()
    value=json.loads(saved['source_cursors']['forest'])
    if fault=='wrong-route':value['route_sha256']='0'*64
    if fault=='past-end':value['position']=3
    if fault=='bool-position':value['position']=True
    raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    if fault=='noncanonical':raw+=b' '
    saved['source_cursors']['forest']=raw
    saved['model']['forest_decoder.readout.bias'].add_(5.)
    before=engine.model.forest_decoder.readout.bias.detach().clone()
    with pytest.raises(ValueError,match='forest.*cursor'):
        engine.restore_state(saved)
    assert torch.equal(engine.model.forest_decoder.readout.bias,before)
    assert engine.steps==0 and engine.source_cursors['forest']==route.initial_cursor()


def test_independent_native_engine_restore_replays_same_next_objective(tmp_path,actual_core_assets,monkeypatch):
    engine,route=native(tmp_path/'first',actual_core_assets)
    checkpoint=tmp_path/'native-forest-zero-update.pt';engine.save_checkpoint(checkpoint)
    monkeypatch.setattr(SourceBoundOwner,'require_training_admission',lambda self,**kw:None)
    expected=native_engine_forest_loss(engine,route,trial_id='test-only-forest')
    other,other_route=native(tmp_path/'second',actual_core_assets)
    assert other is not engine and other.model.forest_flow is not engine.model.forest_flow
    other.load_checkpoint(checkpoint)
    result=native_engine_forest_loss(other,other_route,trial_id='test-only-forest')
    torch.testing.assert_close(expected['loss'],result['loss'],atol=0,rtol=0)
    assert torch.equal(engine.sampler_generator.get_state(),other.sampler_generator.get_state())
    assert result['cursor_after']==expected['cursor_after']
    assert not engine.optimizer.state and not other.optimizer.state
