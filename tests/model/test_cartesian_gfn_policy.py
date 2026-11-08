"""New shared variable-N Cartesian policy equations; synthetic, no admission."""
import copy
import dataclasses
import math
import pytest
import torch
from torch.nn import functional as F
from mora3.teacher_protocol import SystemSpec,TeacherSpec,TeacherRefusal,canonical,sha,FAIRCHEM_REVISION,REPOSITORIES
from mora3.boltzmann_target import BoxSupport,BoltzmannTarget
from mora3.model.cartesian_gfn_policy import (SharedCartesianGFNPolicy,CartesianBudget,
    box_coordinates_and_log_measure,normalized_terminal_log_reward,FEATURE_NAMES)
from mora3.model.cartesian_mup import cartesian_mup_parameter_roles,cartesian_mup_width_family
from mora3.model.mu_optimizer import MuAdamW
from mora3.model.parameterization import parameter_ledger
from mora3.reasoning.gfn_sampling import sample_paths,score_paths
from mora3.reasoning.continuous_gfn import diagonal_gaussian_log_density
from mora3.model.profile_gfn import conditional_trajectory_balance
from test_sciplex_mup import make


def target(n=2,fixed=False):
    # Complete synthetic even-electron He system, not measured/molecular truth.
    atoms=[dict(key='a'+str(i),entity='synthetic',ordinal=i,element='He',atomic_number=2,
                name=None,component=None,residue=None,isotope=None) for i in range(n)]
    payload=dict(system_id='synthetic-'+str(n),topology_id='explicit-unknown',atoms=atoms,bonds=None,
        formal_charges=[0]*n,total_charge=0,spin_multiplicity=1,pbc=[False]*3,cell_angstrom=None,
        protonation=None,cofactor_ids=[],chain_links=[],chemistry_context_json='{}',source_refs=[],
        sequence_chains=[],cofactor_atom_map=[])
    system=SystemSpec.from_export_mapping(payload)
    repo,revision=REPOSITORIES['UMA']
    teacher=TeacherSpec('unit-fixture','UMA',repo,revision,FAIRCHEM_REVISION,'/fixture/never-loaded.pt',
        'a'*64,'omol','angstrom','eV','eV/angstrom','raw-local-no-reference-restoration',None,None,
        '{}','b'*64,17)
    free=tuple(range(1 if fixed else 0,n));positions=((0.,0.,0.),)+(None,)*(n-1) if fixed else (None,)*n
    support=BoxSupport(system.sha256,free,positions,(-2.,)*len(free)*3,(3.,)*len(free)*3,
                       'explicit-synthetic-frame','c'*64)
    return BoltzmannTarget(system,teacher,support,300.,0.,-1000.,'d'*64,'unit-only',True)


def policy(width=8,base=8,**bounds):
    t=target()
    return SharedCartesianGFNPolicy(make(width,base_width=base).backbone,id_width=4,horizon=3,
        budget=dataclasses.replace(CartesianBudget(4,10,16,100000,4,10000),**bounds),teacher_sha256=t.teacher.sha256,
        input_condition_contract_sha256='e'*64,readout_mult=1.5).double()


def test_bind_two_complete_systems_same_registered_parameters_and_rng():
    m=policy();before=tuple((id(p),p.detach().clone()) for p in m.parameters());rng=torch.get_rng_state().clone()
    a=m.bind(target(2));b=m.bind(target(3,True));c=m.bind(target(4))
    assert a.dimension==6 and b.dimension==6 and c.dimension==12 and a.owner is m and b.owner is m
    assert not isinstance(a,torch.nn.Module) and not hasattr(a,'parameters')
    assert torch.equal(rng,torch.get_rng_state())
    assert all(id(p)==i and torch.equal(p,v) for p,(i,v) in zip(m.parameters(),before))
    assert m.graph.layers is m.backbone.layers and m.graph.feature is m.backbone.feature
    assert m.graph.incidence is not m.backbone.incidence
    assert m.graph.configuration()['id_width']==4
    assert len(FEATURE_NAMES)==167 and len(set(FEATURE_NAMES))==167


def test_complete_state_joint_normalized_gaussian_and_independent_readout_equations():
    m=policy(16);bound=m.bind(target(3,True));z=torch.zeros(2,6,dtype=torch.float64)
    context=bound.context(2);mean,ls,stop=bound.parameters_at(z,1,context)
    encoded=m.graph(bound.graphs(z,1))[0]
    nodes=encoded[:,1:4];factor=1.5/(16/8)
    expected=F.linear(nodes[:,[1,2]]*factor,m.forward_head.weight,m.forward_head.bias)
    torch.testing.assert_close(mean,expected[...,:3].flatten(1),rtol=0,atol=0)
    torch.testing.assert_close(ls,expected[...,3:].flatten(1),rtol=0,atol=0)
    torch.testing.assert_close(stop,F.linear(encoded[:,0]*factor,m.stop_head.weight,m.stop_head.bias).squeeze(-1),rtol=0,atol=0)
    value=z+.2
    density=diagonal_gaussian_log_density(value,mean,ls)
    oracle=(-.5*((value-mean)/ls.exp()).square()-ls-.5*math.log(2*math.pi)).sum(-1)
    torch.testing.assert_close(density,oracle,rtol=1e-12,atol=1e-12)
    changed=z.clone();changed[:,3:]=.5
    changed_mean=bound.parameters_at(changed,1,context)[0]
    assert not torch.equal(changed_mean[:,:3],mean[:,:3]),'remote free atom must affect other atoms through shared full-system attention'
    torch.testing.assert_close(bound.log_partition(context),m.normalizer(context*factor).squeeze(-1),rtol=0,atol=0)


def test_exact_current_state_features_static_logz_and_target_exclusion():
    m=policy();b=m.bind(target(3,True));z=torch.tensor([[.1,.2,.3,.4,.5,.6]],dtype=torch.float64)
    graphs=b.graphs(z,2);raw=b.feature_rows(z,2)[0]
    full,_=box_coordinates_and_log_measure(z,b.target.support)
    assert raw.shape==(3,167)
    torch.testing.assert_close(raw[:,-8:-5],full[0])
    torch.testing.assert_close(raw[1:,-5:-2],z.reshape(2,3))
    assert torch.equal(raw[:,-2],torch.ones(3,dtype=raw.dtype))
    assert torch.equal(raw[:,-1],torch.full((3,),2/3,dtype=raw.dtype))
    before=b.context(1);b.parameters_at(z,2,before);after=b.context(1)
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    # Audit names/IDs/provenance never become feature values.
    p=__import__('json').loads(b.target.system.topology_json);p['source_refs']=['untrusted-audit-target']
    p['system_id']='renamed';p['topology_id']='renamed'
    s=SystemSpec.from_export_mapping(p);support=dataclasses.replace(b.target.support,system_sha256=s.sha256)
    other=m.bind(dataclasses.replace(b.target,system=s,support=support,target_revision='audit-only'))
    torch.testing.assert_close(other.feature_rows(z,2)[0],raw,rtol=0,atol=0)
    torch.testing.assert_close(other.context(1),before,rtol=0,atol=0)
    g=graphs[0];assert torch.equal(g['identifiers'],torch.eye(3,4,dtype=torch.float64))


def test_joint_atom_identifier_entity_permutation_consistency():
    m=policy();b=m.bind(target(3));g=b.graphs(torch.zeros(1,9,dtype=torch.float64),1)[0]
    perm=torch.tensor([2,0,1]);inverse=torch.argsort(perm)
    changed={**g,'features':g['features'][perm],'roles':g['roles'][perm],
        'identifiers':g['identifiers'][perm],'endpoints':inverse[g['endpoints'][perm]]}
    original=m.graph([g])[0];actual=m.graph([changed])[0]
    torch.testing.assert_close(actual[:,0],original[:,0],rtol=1e-11,atol=1e-11)
    torch.testing.assert_close(actual[:,1:],original[:,1:][:,perm],rtol=1e-11,atol=1e-11)


def test_box_jacobian_normalized_base_fixed_atoms_and_saturation_refusal():
    t=target(3,True);z=torch.tensor([[.1,.2,.3,-.2,-.3,-.4]],dtype=torch.float64,requires_grad=True)
    full,j=box_coordinates_and_log_measure(z,t.support)
    torch.testing.assert_close(full[:,0],torch.zeros(1,3,dtype=z.dtype))
    torch.testing.assert_close(full[:,1:].flatten(1),-2+5*z.sigmoid())
    torch.testing.assert_close(j,(F.logsigmoid(z)+F.logsigmoid(-z)).sum(-1))
    _,r=normalized_terminal_log_reward(z,(1.,),t,horizon=3)
    assert not r.requires_grad
    torch.testing.assert_close(r,z.new_tensor([t.energy_log_reward(1.)])+j.detach()-math.log(3))
    with pytest.raises(TeacherRefusal,match='boundary'):box_coordinates_and_log_measure(z.detach()+1000,t.support)


@pytest.mark.parametrize('failure',['atoms','teacher','entities','attention','batch','nonfinite'])
def test_refusal_before_graph_allocation_rng_or_parameter_changes(failure):
    bounds={'max_entities':2} if failure=='entities' else ({'max_attention_elements':1} if failure=='attention' else {})
    m=policy(**bounds);t=target(2);before=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    if failure=='atoms':t=target(5)
    elif failure=='teacher':t=dataclasses.replace(t,teacher=dataclasses.replace(t.teacher,teacher_id='different'))
    with pytest.raises((TeacherRefusal,ValueError)):
        b=m.bind(t)
        if failure=='batch':b.context(5)
        elif failure=='nonfinite':b.parameters_at(torch.full((1,6),float('nan')),1,b.context(1))
        elif failure=='attention':b.context(1)
    assert torch.equal(rng,torch.get_rng_state())
    for name,p in m.state_dict().items():
        if isinstance(p,torch.Tensor):assert torch.equal(p,before[name])


def test_sample_score_real_conditional_tb_all_kernels_and_shared_backbone_gradients():
    m=policy();b=m.bind(target(3,True));context=b.context(2)
    paths=sample_paths(b,2,generator=torch.Generator().manual_seed(617),optional_stop=False,context=context)
    scores=score_paths(b,paths,context=context)
    loss=conditional_trajectory_balance(b.log_partition(context),scores['log_forward'],scores['log_backward'],
        scores['log_stop'],torch.tensor([-.3,-.7],dtype=context.dtype),paths['mask'])['loss']
    loss.backward()
    for head in ('forward_head','backward_head','normalizer'):
        p=getattr(m,head).weight;assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert m.backbone.feature.weight.grad.abs().sum()>0
    # Optional stopping path separately exercises normalized Bernoulli gradient.
    m.zero_grad();context=b.context(1);mean,ls,stop=b.parameters_at(torch.zeros(1,6,dtype=context.dtype),1,context)
    (-F.logsigmoid(stop)).sum().backward();assert m.stop_head.weight.grad.abs().sum()>0


def test_complete_live_width_family_aliases_unique_counts_and_optimizer_no_update():
    ms=tuple(policy(w) for w in (8,12,16));f=cartesian_mup_width_family(*ms)
    m=ms[-1];roles=cartesian_mup_parameter_roles(m)
    ledger=parameter_ledger(m,roles)
    assert ledger.total_numel==sum(p.numel() for p in m.parameters())
    assert roles['geometry_features.weight'].role=='input'
    assert roles['graph.incidence.weight'].axes[1].kind=='finite'
    assert roles['forward_head.weight'].role=='output'
    entry=next(e for e in ledger.parameters if 'graph.layers.0.attention.qkv.weight' in e.aliases)
    assert 'backbone.layers.0.attention.qkv.weight' in entry.aliases
    opt=MuAdamW(m,f,roles,base_lr=.02,weight_decay=0.,decay_aliases=())
    by={n:g for g in opt.param_groups for n in g['mup_aliases']}
    assert by['geometry_features.weight']['lr']==.02 and by['forward_head.weight']['lr']==.02
    assert by['backbone.layers.0.attention.qkv.weight']['lr']==.01
    assert not opt.state


def test_configuration_alias_and_checkpoint_refusal_before_shared_parameter_mutation():
    m=policy();state=copy.deepcopy(m.state_dict());before=[p.detach().clone() for p in m.parameters()]
    first=next(k for k,v in state.items() if isinstance(v,torch.Tensor));state[first]+=4
    state['_extra_state']['teacher_sha256']='f'*64
    with pytest.raises(ValueError):m.load_state_dict(state)
    assert all(torch.equal(p,v) for p,v in zip(m.parameters(),before))
    m.graph.layers=copy.deepcopy(m.graph.layers)
    with pytest.raises(ValueError,match='alias'):m.get_extra_state()


def test_fresh_head_rescale_exact_same_rng_native_fanin_and_never_reinitialized_on_load():
    torch.manual_seed(916);base=policy(16,base=16);rng=torch.get_rng_state().clone()
    torch.manual_seed(916);wide=policy(16,base=8)
    assert torch.equal(torch.get_rng_state(),rng)
    for name in ('forward_head','backward_head','stop_head','normalizer'):
        for field in ('weight','bias'):
            # Rescaling is applied in native FP32 at construction, then double.
            expected=(getattr(getattr(base,name),field).float()*math.sqrt(2)).double()
            torch.testing.assert_close(getattr(getattr(wide,name),field),expected,rtol=0,atol=0)
    assert torch.equal(base.geometry_features.weight,wide.geometry_features.weight)
    state=copy.deepcopy(wide.state_dict());wide.load_state_dict(state)
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in wide.state_dict().items():
        if isinstance(value,torch.Tensor):assert torch.equal(value,state[key])


def test_declared_bonds_bidirectional_unknown_mask_and_source_budget_conservation():
    m=policy();t=target(3);payload=__import__('json').loads(t.system.topology_json)
    payload['bonds']=[dict(first='a0',second='a2',kind='noncovalent_contact',order=None,
                           periodic_image=None,provenance='synthetic-explicit-input')]
    system=SystemSpec.from_export_mapping(payload);support=dataclasses.replace(t.support,system_sha256=system.sha256)
    b=m.bind(dataclasses.replace(t,system=system,support=support))
    g=b.graphs(torch.zeros(1,9,dtype=torch.float64),1)[0];raw=b.feature_rows(torch.zeros(1,9,dtype=torch.float64),1)[0]
    assert len(raw)==5 and torch.equal(g['endpoints'],torch.tensor([[0,0],[1,1],[2,2],[0,2],[2,0]]))
    assert torch.equal(g['roles'],torch.tensor([0,0,0,1,1]))
    assert raw[:,FEATURE_NAMES.index('topology-unknown')].sum()==0
    assert raw[3:,FEATURE_NAMES.index('bond-kind:noncovalent_contact')].sum()==2
    assert raw[3:,FEATURE_NAMES.index('bond-order-unknown')].sum()==2
    assert raw[3:,FEATURE_NAMES.index('current-state-available')].sum()==0
    b.target=target(2)
    with pytest.raises(TeacherRefusal,match='bound'):b.context(1)
