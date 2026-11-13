"""Complete prefix conditioning through the existing shared graph transformer."""
from dataclasses import replace
import pytest
import torch
from test_molecule_profile_model import model,view
from test_reasoning_trajectories import branching_trace
from mora3.model.profile_gfn import ProfileGFNPolicy
from mora3.model.geometry_conditioning import GeometryConditioner
from mora3.reasoning.trajectory_contracts import prefix_native_graph
from mora3.reasoning.trajectory_history import NativePrefixContext
from mora3.reasoning.trajectory_gfn import sample_native_path_batch,pathspace_objective,PathSpaceReward

H='a'*64


def setup():
    torch.manual_seed(20261110)
    policy=ProfileGFNPolicy(model().double(),dimension=2,horizon=3,hidden=5,source_binding={'pool_sha256':H})
    geometry=GeometryConditioner(policy.profiles.shared.backbone,id_width=8).double()
    history=NativePrefixContext(policy,geometry,source_context_sha256=H,coefficient=.25)
    views=(view('profile_to_molecule'),)
    return policy,geometry,history,views


def test_full_past_not_only_current_latent_changes_condition():
    p,g,h,views=setup();context=p.context(views)
    a=torch.tensor([[[1.,0.],[0.,0.]]],dtype=torch.float64)
    b=torch.tensor([[[-1.,0.],[0.,0.]]],dtype=torch.float64)
    assert torch.equal(a[:,-1],b[:,-1])
    ca=h.condition(a,context);cb=h.condition(b,context)
    assert not torch.equal(ca,cb)
    assert not torch.equal(p.parameters_at(a[:,-1],2,ca)[0],p.parameters_at(b[:,-1],2,cb)[0])
    ca.square().sum().backward()
    assert g.payload.embedding.weight.grad.abs().sum()>0
    assert p.profiles.shared.backbone.feature.weight.grad.abs().sum()>0
    assert h.binding()['adds_learned_parameters']is False
    assert h.binding()['prefix_direction']=='computational-not-biological-causal'


def test_native_history_sampler_and_rescoring_reach_all_shared_owners():
    p,g,h,views=setup()
    batch=sample_native_path_batch(p,views,generator=torch.Generator().manual_seed(22),source_context_sha256=H,history_context=h)
    guide=torch.tensor([[[0.,0.],[1.,0.],[1.,1.],[2.,1.]]],dtype=torch.float64)
    r=PathSpaceReward(guide,source_sha256='b'*64,encoder_sha256='c'*64,alignment_sha256='d'*64,
        group_policy_sha256='e'*64,guide_groups=(('guide',),),query_groups=(('query',),),
        scale=1.,metric_weight=.1,topology_weight=.1,evidence_kind='synthetic_fixture')
    result=pathspace_objective(p,views,batch,r,history_context=h)
    result['loss'].backward()
    assert result['forward_history_parameterization']=='complete-prefix-native-shared-TokenGT'
    assert p.forward_head.weight.grad.abs().sum()>0 and g.payload.embedding.weight.grad.abs().sum()>0
    with pytest.raises(ValueError):pathspace_objective(p,views,batch,r)
    with torch.no_grad():g.payload.embedding.weight.add_(.1)
    with pytest.raises(ValueError):pathspace_objective(p,views,batch,r,history_context=h)


def test_history_adapter_rejects_unshared_transformer_and_rank_overflow():
    p,g,h,views=setup()
    other=GeometryConditioner(model().shared.backbone,id_width=8).double()
    with pytest.raises(ValueError):NativePrefixContext(p,other,source_context_sha256=H,coefficient=.25)
    with pytest.raises(ValueError):h.condition(torch.zeros((1,8,2),dtype=torch.float64),p.context(views))
    with pytest.raises(ValueError):h.condition(torch.zeros((1,2,2),dtype=torch.float64,requires_grad=True),p.context(views))


def test_prepared_native_graph_keeps_all_steps_but_excludes_future_and_target_data():
    trace=branching_trace();record,task,receipt=prefix_native_graph(trace,('a',))
    assert len(record.nodes)==2 and len(record.edges)==1 and record.edges[0].directed
    assert any(m.name=='depends_on:left_result'and m.value=='["question"]'for m in record.nodes[-1].measurements)
    assert any(m.name=='requires'and m.value=='["question"]'for m in record.edges[0].measurements)
    assert task.available_targets(record)==()
    changed=replace(trace,identity='different',facts=tuple(replace(f,value_json='{"gold":999}')if f.role=='target'else f for f in trace.facts))
    other,other_task,other_receipt=prefix_native_graph(changed,('a',))
    assert record.nodes==other.nodes and record.edges==other.edges
    assert task==other_task and receipt['condition_sha256']==other_receipt['condition_sha256']
    assert receipt['observed_biological_trajectory']is False


def test_twelve_transition_native_history_is_not_reduced_to_endpoint_pair():
    torch.manual_seed(20261110)
    policy=ProfileGFNPolicy(model().double(),dimension=2,horizon=12,hidden=5,source_binding={'pool_sha256':H})
    geometry=GeometryConditioner(policy.profiles.shared.backbone,id_width=16).double()
    history=NativePrefixContext(policy,geometry,source_context_sha256=H,coefficient=.25)
    views=(view('profile_to_molecule'),)
    batch=sample_native_path_batch(policy,views,generator=torch.Generator().manual_seed(122),source_context_sha256=H,history_context=history)
    t=torch.linspace(0,1,13,dtype=torch.float64)
    guide=torch.stack((t,t.square()),-1)[None]
    reward=PathSpaceReward(guide,source_sha256='b'*64,encoder_sha256='c'*64,alignment_sha256='d'*64,
        group_policy_sha256='e'*64,guide_groups=(('guide',),),query_groups=(('query',),),
        scale=1.,metric_weight=.1,topology_weight=.1,evidence_kind='synthetic_fixture')
    result=pathspace_objective(policy,views,batch,reward,history_context=history)
    assert result['observed_transitions']==12 and batch.paths['states'].shape==(1,12,2)
    result['loss'].backward()
    assert policy.forward_head.weight.grad.abs().sum()>0
    assert geometry.payload.embedding.weight.grad.abs().sum()>0
    assert history.binding()['objective_kind']=='complete-trajectory-pathspace-not-endpoint-reward'
