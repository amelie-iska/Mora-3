"""Native hybrid-forest mathematical/integration checks. No optimizer updates."""
import copy
from dataclasses import replace
import math
import pytest
import torch
from test_full_owner import actual_core_assets, cases
from test_forest_flow_owner import assets, AXES
from test_forest_flow import problem, SOURCE, COORD
from mora3.model.full_owner import construct_primary_owner, PrimaryConfiguration, unique_parameter_inventory
from mora3.model.profile_gfn import ProfileGFNPolicy
from mora3.model.forest_topology import ForestTopologySpec, topology_log_probability
from mora3.model.forest_quality import ForestQualitySpec
from mora3.reasoning.forest_topology import TopologyBounds, prefix_state
from mora3.model.hybrid_forest_teacher import (
    HybridForestSpec, NativeHybridForestTeacher, FrozenForestReward,
    uniform_topology_log_mass, sample_hybrid_forests, hybrid_forest_balance,
    hybrid_forest_student_loss, hybrid_forest_objective, native_hybrid_forest_objective)


def setup_owner(tmp_path, source, width=8, bounds=None):
    src=replace(assets(tmp_path,source),
        forest_topology_spec=ForestTopologySpec(bounds or TopologyBounds(4,3,3),'f'*64),
        forest_quality_spec=ForestQualitySpec(('constraint','order'),'e'*64,(1.,1.)))
    owner=construct_primary_owner(PrimaryConfiguration(width,8,SOURCE),src).eval()
    policy=next(m for m in owner.modules() if type(m)is ProfileGFNPolicy)
    teacher=NativeHybridForestTeacher(owner.forest_topology,policy,
        HybridForestSpec('1'*64,'2'*64,'3'*64,context_coefficient=.25))
    frozen=copy.deepcopy(owner.forest_quality).requires_grad_(False).eval()
    reward=FrozenForestReward(frozen,teacher,training_evidence_sha256='4'*64,
        supported_criteria=('constraint','order'),scale=1.3,node_weight=.2)
    return owner,teacher,reward


@pytest.fixture
def native(tmp_path,actual_core_assets):
    return setup_owner(tmp_path,actual_core_assets)


def draw(teacher,count=8,seed=81,roots=None):
    r,t=problem(); roots=torch.tensor([[.2,.4]],dtype=torch.float32) if roots is None else roots
    return sample_hybrid_forests(teacher,r,t,root_ids=tuple('root:'+str(i) for i in range(len(roots))),
        root_states=roots,count=count,generator=torch.Generator().manual_seed(seed))


def test_topology_reference_is_normalized_over_variable_sizes_and_multiple_roots():
    bounds=TopologyBounds(5,3,3)
    for roots in (1,2,3,5):
        pending=[()]; masses=[];sizes=set()
        while pending:
            actions=pending.pop(); state=prefix_state(bounds,roots,actions)
            if state.complete:
                masses.append(math.exp(uniform_topology_log_mass(bounds,roots,actions)))
                sizes.add(len(state.parents))
            else: pending.extend(actions+(a,) for a in state.allowed_actions)
        assert math.isclose(sum(masses),1.,abs_tol=1e-14)
        assert roots in sizes


def test_exact_joint_density_matches_independent_normal_distribution_and_TB(native):
    owner,teacher,reward=native;r,t=problem();batch=draw(teacher,16)
    result=hybrid_forest_balance(teacher,r,t,batch,reward,split='train')
    assert len({len(f.parents) for f in batch.forests})>1
    manual=[];base=[]
    for forest in batch.forests:
        topology=topology_log_probability(teacher.topology,r,t,forest.parents,root_count=1)['log_probability']
        terms=[]
        for i in range(1,len(forest.parents)):
            mean,ls=teacher.transition_parameters(r,t,forest,i)
            delta=forest.states[i]-forest.states[forest.parents[i]]
            terms.append(torch.distributions.Normal(mean,ls.exp()).log_prob(delta).sum())
        manual.append(topology+sum(terms))
        q=uniform_topology_log_mass(teacher.topology.spec.bounds,1,
            tuple(batch.topology_samples[len(manual)-1]['actions']))
        for i in range(1,len(forest.parents)):
            delta=forest.states[i]-forest.states[forest.parents[i]]
            q=q+torch.distributions.Normal(torch.zeros_like(delta),torch.full_like(delta,reward.scale)).log_prob(delta).sum()
        base.append(torch.as_tensor(q,dtype=forest.states.dtype))
    torch.testing.assert_close(result['log_forward'],torch.stack(manual),atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(result['log_reference'],torch.stack(base),atol=2e-6,rtol=2e-6)
    assert torch.all(result['energy']>=0)
    torch.testing.assert_close(result['log_reward'],result['log_reference']-result['energy'])
    residual=result['log_partition']+result['log_forward']-result['log_reward']
    torch.testing.assert_close(result['loss'],residual.square().mean())
    assert result['reward_proportionality_verified']is False
    result['loss'].backward()
    for p in (teacher.policy.forward_head.weight,teacher.policy.normalizer.weight,
              teacher.topology.action_readout.weight,teacher.topology.field.geometry.graph.feature.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert all(p.grad is None for p in reward.quality.parameters())


def test_ancestor_kernel_excludes_siblings_future_values_and_hidden_problem_target(native):
    _,teacher,_=native;r,t=problem()
    from mora3.reasoning.flow_forest_refinement import ForestState
    from mora3.model.forest_flow import forest_context_sha256
    states=torch.tensor([[.2,.4],[.8,.9],[1.,1.1],[2.,2.1]],dtype=torch.float32)
    f=ForestState(('r','a','b','c'),(-1,0,0,2),('open','terminal','open','terminal'),states,(0,),
        ('root','computational','computational','computational'),forest_context_sha256(r,t,source_snapshot_sha256=SOURCE),COORD,'generated')
    original=teacher.transition_parameters(r,t,f,3)
    changed=f.copy();changed.states[1]=987.;changed.states[3]=-555.
    other=teacher.transition_parameters(*problem(target=-999.),changed,3)
    for a,b in zip(original,other):torch.testing.assert_close(a,b,atol=0,rtol=0)
    changed.states[2]+=2.
    assert not torch.equal(original[0],teacher.transition_parameters(r,t,changed,3)[0])


def test_sampling_and_student_generator_replay_no_global_RNG_and_no_parameter_changes(native):
    owner,teacher,reward=native;r,t=problem();before=unique_parameter_inventory(owner)
    g=torch.Generator().manual_seed(12);global_before=torch.get_rng_state().clone()
    roots=torch.tensor([[.2,.4]],dtype=torch.float32)
    a=sample_hybrid_forests(teacher,r,t,root_ids=('r',),root_states=roots,count=8,generator=g)
    g.set_state(a.generator_before)
    b=sample_hybrid_forests(teacher,r,t,root_ids=('r',),root_states=roots,count=8,generator=g)
    assert a.sample_sha256==b.sample_sha256 and torch.equal(a.generator_after,b.generator_after)
    gs=torch.Generator().manual_seed(48);s0=gs.get_state().clone()
    x=hybrid_forest_student_loss(teacher,r,t,a,generator=gs,split='train');s1=gs.get_state().clone()
    gs.set_state(s0);y=hybrid_forest_student_loss(teacher,r,t,a,generator=gs,split='train')
    torch.testing.assert_close(x['loss'],y['loss'],atol=0,rtol=0)
    assert torch.equal(gs.get_state(),s1) and torch.equal(torch.get_rng_state(),global_before)
    x['loss'].backward();assert teacher.topology.field.readout.weight.grad.abs().sum()>0
    assert all(not f.states.requires_grad for f in a.forests)
    assert unique_parameter_inventory(owner)==before


@pytest.mark.parametrize('fault',('edited','reordered','weight','context','root','reward','beam','subset'))
def test_stale_pruned_modified_or_wrong_snapshot_samples_refuse(native,fault):
    _,teacher,reward=native;r,t=problem();batch=draw(teacher)
    if fault=='edited':batch.forests[-1].states[-1,0]+=1.
    elif fault=='reordered':batch=replace(batch,forests=batch.forests[::-1])
    elif fault=='weight':
        with torch.no_grad():teacher.policy.forward_head.weight.add_(.01)
    elif fault=='context':r,t=problem(dose=2.)
    elif fault=='root':batch.root_states[0,0]+=1
    elif fault=='reward':
        with torch.no_grad():reward.quality.forest_readout.weight.add_(.01)
    elif fault=='beam':batch=replace(batch,sampler='beam-pruned')
    elif fault=='subset':batch=replace(batch,forests=batch.forests[:1])
    with pytest.raises(ValueError):hybrid_forest_balance(teacher,r,t,batch,reward,split='train')


def test_sampler_failure_restores_explicit_rng(native,monkeypatch):
    _,teacher,_=native;r,t=problem();g=torch.Generator().manual_seed(81);before=g.get_state().clone()
    def fail(*a,**k):raise ValueError('injected transition failure')
    monkeypatch.setattr(teacher,'transition_parameters',fail)
    with pytest.raises(ValueError):sample_hybrid_forests(teacher,r,t,root_ids=('r',),
        root_states=torch.zeros(1,2,dtype=torch.float32),count=16,generator=g)
    assert torch.equal(before,g.get_state())


@pytest.mark.parametrize('split',('test','validation'))
def test_heldout_refuses_before_neural_calls_or_rng(native,split):
    _,teacher,reward=native;r,t=problem();calls=[]
    h=teacher.topology.register_forward_pre_hook(lambda *a:calls.append(1))
    g=torch.Generator().manual_seed(8);before=g.get_state().clone()
    try:
        with pytest.raises(ValueError):hybrid_forest_objective(teacher,r,t,reward,root_ids=('r',),
            root_states=torch.zeros(1,2,dtype=torch.float32),count=4,generator=g,split=split)
        assert not calls and torch.equal(before,g.get_state())
    finally:h.remove()


def test_root_only_support_has_unit_continuous_mass_and_zero_student_loss(tmp_path,actual_core_assets):
    _,teacher,reward=setup_owner(tmp_path,actual_core_assets,bounds=TopologyBounds(2,0,0))
    r,t=problem();roots=torch.tensor([[.2,.4],[.7,.1]],dtype=torch.float32)
    b=draw(teacher,count=2,roots=roots)
    out=hybrid_forest_balance(teacher,r,t,b,reward,split='train')
    assert torch.equal(out['log_forward'],torch.zeros_like(out['log_forward']))
    assert torch.equal(out['log_reference'],torch.zeros_like(out['log_reference']))
    student=hybrid_forest_student_loss(teacher,r,t,b,generator=torch.Generator().manual_seed(1),split='train')
    assert student['loss'].item()==0 and student['supervised_coordinates']==0


def test_native_admission_remains_first_and_alias_inventory_unchanged(native):
    owner,teacher,reward=native
    before=unique_parameter_inventory(owner);rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='not full corpus'):
        native_hybrid_forest_objective(owner,teacher,'untrusted inputs not yet inspected')
    assert before==unique_parameter_inventory(owner) and torch.equal(rng,torch.get_rng_state())


def test_joint_objective_reaches_flow_topology_GFN_and_normalizer(native):
    owner,teacher,reward=native;r,t=problem();g=torch.Generator().manual_seed(81)
    out=hybrid_forest_objective(teacher,r,t,reward,root_ids=('r',),root_states=torch.zeros(1,2,dtype=torch.float32),
        count=16,generator=g,split='train',balance_weight=.3,student_weight=.7)
    torch.testing.assert_close(out['loss'],.3*out['balance']['loss']+.7*out['student']['loss'])
    out['loss'].backward()
    for p in (teacher.topology.field.readout.weight,teacher.topology.action_readout.weight,
              teacher.policy.forward_head.weight,teacher.policy.normalizer.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert out['optimizer_updates']==0 and out['training_admission']is False
