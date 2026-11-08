"""Fixed finite GFN hidden coordinates; main-context width is the only scale."""
import copy
import pytest
import torch
from torch.nn import functional as F
from mora3.model.profile_gfn import ProfileGFNPolicy
from mora3.model.gfn_mup import gfn_mup_parameter_roles,gfn_mup_width_family
from test_profile_sequence_mup import profiles


def make(width=8):
    return ProfileGFNPolicy(profiles(width),dimension=2,horizon=3,hidden=5,
        source_binding={'scope':'synthetic-reference-not-biological-reward'},parameterization_policy='mup-profile-gfn-v1')


@pytest.mark.parametrize('backward',[False,True])
def test_actual_gaussian_stop_and_log_partition_outputs_all_gradients(backward):
    torch.manual_seed(2026110677);p=make(16).double()
    states=torch.tensor([[.2,-.4],[.3,.6]],dtype=torch.float64,requires_grad=True)
    context=torch.linspace(-1,1,32,dtype=torch.float64).reshape(2,16).requires_grad_()
    projection=(p.backward_net if backward else p.forward_net)[0]
    fixed=torch.cat((states,states.new_full((2,1),1/3)),-1)
    h=torch.tanh(F.linear(fixed,projection.fixed_weight,projection.bias)
                 +F.linear(context*(1.5/(16/8)),projection.context_weight))
    head=p.backward_head if backward else p.forward_head
    gaussian=F.linear(h,head.weight,head.bias)
    expected=(*gaussian.chunk(2,-1),F.linear(h,p.stop_head.weight,p.stop_head.bias).squeeze(-1),
              F.linear(context*(1.5/(16/8)),p.normalizer.weight,p.normalizer.bias).squeeze(-1))
    actual=(*p.parameters_at(states,1,context,backward=backward),p.log_partition(context))
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    params=(states,context,*p.parameters())
    for a,b in zip(torch.autograd.grad(sum(x.sum() for x in actual),params,retain_graph=True,allow_unused=True),
                   torch.autograd.grad(sum(x.sum() for x in expected),params,allow_unused=True)):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)


def test_complete_fixed_hidden_main_context_family_and_checkpoint_sidecars():
    ms=[make(w) for w in (8,12,16)]
    family=gfn_mup_width_family(*ms,identifier_axis=('protein-coordinate',))
    assert family.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in ms)
    roles=gfn_mup_parameter_roles(ms[-1],identifier_axis=('protein-coordinate',))
    assert tuple(a.kind for a in roles['forward_net.0.fixed_weight'].axes)==('finite','finite')
    assert tuple(a.kind for a in roles['forward_net.0.context_weight'].axes)==('finite','width')
    assert not any(r.role=='custom' for r in roles.values())
    state=copy.deepcopy(ms[-1].state_dict());state['forward_net.0._extra_state']['context_mult']=99.
    state['profiles.shared.payload.embedding.weight'].fill_(99)
    before={n:p.clone() for n,p in ms[-1].named_parameters()}
    with pytest.raises(ValueError,match='configuration|policy'):ms[-1].load_state_dict(state)
    for n,p in ms[-1].named_parameters():assert torch.equal(p,before[n])


from test_full_owner import actual_core_assets,cases


def test_primary_registers_same_real_policy_and_normalized_tb_has_kernel_gradients(tmp_path,actual_core_assets):
    from dataclasses import replace
    from test_goa_objective import view as goa_packet
    from test_molecule_profile_model import view as profile_view
    from mora3.model.full_owner import PrimaryConfiguration,construct_primary_owner
    from mora3.model.profile_gfn import conditional_trajectory_balance
    from mora3.model.profile_gfn_training import LatentGaussianReward
    from mora3.reasoning.gfn_sampling import sample_paths,score_paths
    torch.manual_seed(2026110679)
    assets=replace(actual_core_assets,goa_packets=(goa_packet(tmp_path),))
    owner=construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),assets)
    anchor=owner.core.parent.parent;policy=anchor.profile_gfn
    assert policy.profiles is anchor.profile
    assert anchor.reconstruction._policy==policy.get_extra_state()
    owned={id(p) for p in owner.parameters()}
    assert all(id(p) in owned for p in policy.parameters())
    context=policy.context((profile_view('profile_to_molecule'),)*8)
    paths=sample_paths(policy,8,context=context,generator=torch.Generator().manual_seed(2026110679))
    assert (paths['lengths']>1).any()
    states=paths['states'];terminal=states[torch.arange(len(states)),paths['lengths']-1]
    reward=LatentGaussianReward()(terminal,horizon=policy.horizon)
    terms=conditional_trajectory_balance(policy.log_partition(context),
        **score_paths(policy,paths,context=context),log_reward=reward,mask=paths['mask'])
    terms['loss'].backward()
    for module in (policy.forward_net,policy.backward_net,policy.forward_head,policy.backward_head,policy.stop_head,policy.normalizer):
        gradients=[p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients)>0
