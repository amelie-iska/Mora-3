"""Optional topology supervision uses the existing native source route; zero updates."""
from dataclasses import replace,asdict,fields,is_dataclass
from hashlib import sha256
import pytest
import torch
from test_forest_training import fixture
from test_forest_flow import state
from mora3.model.forest_flow import _wire
from mora3.model.forest_training import (ForestTrainingHeads,ForestTopologySupervision,
    preflight_forest_batch,forest_joint_loss,forest_batch_sha256,_identity)
from mora3.model.forest_topology import NativeForestTopologyHead,ForestTopologySpec,forest_topology_loss
from mora3.reasoning.forest_topology import TopologyBounds
from mora3.research.forest_training_routes import ForestTrainingRoute


def enabled():
    h,s,b=fixture()
    policy=NativeForestTopologyHead(h.flow,ForestTopologySpec(TopologyBounds(8,7,7),'b'*64))
    h=replace(h,topology=policy);s=replace(s,heads_sha256=h.sha256,topology_weight=.7)
    b=replace(b,topology=ForestTopologySupervision(1,((-1,0,0),(-1,0,1,1)),'b'*64,'a'*64))
    return h,s,b


def test_new_objective_is_fixed_weight_sum_and_uses_original_topology_nll():
    h,s,b=enabled();noise=state();t=torch.tensor([.2,.8],dtype=noise.dtype)
    actual=forest_joint_loss(h,s,b,noise=noise,time=t)
    old=forest_joint_loss(h,replace(s,topology_weight=0.),replace(b,topology=None),noise=noise,time=t)
    topo=forest_topology_loss(h.topology,b.record,b.task,b.topology.forests,root_count=1,split='train')['loss']
    torch.testing.assert_close(actual['loss'],old['loss']+.7*topo,atol=0,rtol=0)
    assert actual['counts']['topology']==12
    actual['loss'].backward()
    assert h.topology.action_readout.weight.grad.abs().sum()>0
    assert h.flow.geometry.graph.feature.weight.grad.abs().sum()>0


def test_topology_only_source_segment_prepares_and_hashes_actual_label_content():
    h,s,b=enabled();b=replace(b,flow=None,tokens=None,quality=None,analogy=None)
    route=ForestTrainingRoute(h,s,(b,),task_key='forest',trial_id='synthetic-no-run')
    selected=route.select(h,route.initial_cursor(),max_windows=1)
    assert selected['status']=='prepared'and selected['counts']['topology']==12
    value=forest_joint_loss(h,s,selected['batch'])
    assert set(value['component_losses'])=={'topology'}and value['loss'].requires_grad
    different=replace(b,topology=replace(b.topology,forests=((-1,0),(-1,0))))
    assert forest_batch_sha256(different)!=forest_batch_sha256(b)


@pytest.mark.parametrize('fault',('heldout','policy','missing-head','bad-parent','source-count','root-count','over-budget','unknown-type'))
def test_topology_failure_is_before_any_head_forward(fault):
    h,s,b=enabled()
    if fault=='heldout':b=replace(b,identity=replace(b.identity,split='test'))
    elif fault=='policy':b=replace(b,topology=replace(b.topology,construction_policy_sha256='0'*64))
    elif fault=='missing-head':h=replace(h,topology=None);s=replace(s,heads_sha256=h.sha256)
    elif fault=='bad-parent':b=replace(b,topology=replace(b.topology,forests=((-1,2),(-1,0))))
    elif fault=='source-count':b=replace(b,topology=replace(b.topology,forests=((-1,0),)))
    elif fault=='root-count':b=replace(b,topology=replace(b.topology,root_count=2))
    elif fault=='over-budget':b=replace(b,topology=replace(b.topology,forests=((-1,)*9,(-1,))))
    elif fault=='unknown-type':b=replace(b,topology={})
    calls=[];handles=[m.register_forward_pre_hook(lambda *a:calls.append(1))for m in(h.flow,h.decoder,h.quality,h.analogy)]
    try:
        with pytest.raises(ValueError):forest_joint_loss(h,s,b,noise=state(),time=torch.tensor([.3,.6],dtype=torch.float64))
        assert calls==[]
    finally:
        for handle in handles:handle.remove()


def test_absent_optional_topology_preserves_exact_old_spec_and_batch_wire():
    h,s,b=fixture()
    assert h.binding()['schema']=='Mora3.NativeForestTrainingHeads/v1'
    assert set(h.binding()['heads'])=={'flow','decoder','quality','analogy'}
    d=asdict(s);d.pop('topology_weight')
    expected=sha256(_wire(dict(schema='Mora3.ForestTrainingObjective/v1',**d,
        reduction='fixed-weight-sum-of-active-head-means-no-renormalization',topology='source-declared-fixed-skeleton',
        decoder_input='separate-detached-source-encoder-export',quality_input='separate-detached-scored-candidate'))).hexdigest()
    assert s.sha256==expected
    identity=_identity(b)
    assert 'topology'not in identity['fields']
    assert forest_batch_sha256(b)==sha256(_wire(identity)).hexdigest()


@pytest.mark.parametrize('weight',(True,-1.,float('nan'),float('inf')))
def test_optional_weight_rejects_nonfinite_and_wrong_type(weight):
    h,s,b=fixture()
    with pytest.raises(ValueError):replace(s,topology_weight=weight).__post_init__()
