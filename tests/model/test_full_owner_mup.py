"""Literal assembled owner family with every alias, synthetic source fixtures."""
import copy
from dataclasses import replace
import pytest
import torch
from test_full_owner import actual_core_assets,cases
from mora3.model.full_owner import construct_primary_owner,PrimaryConfiguration,unique_parameter_inventory
from mora3.model.full_owner_mup import full_owner_mup_parameter_roles,full_owner_mup_width_family
from mora3.model.parameterization import parameter_ledger
from mora3.model.mu_optimizer import MuAdamW

AXES=dict(identifier_axis=('context','observation'),
    geometry_identifier_axis=tuple('geometry-source-coordinate:'+str(i) for i in range(16)),
    color_identifier_axis=('color-context-coordinate:0','color-context-coordinate:1'))


def members(tmp_path,assets):
    from test_goa_objective import view
    assets=replace(assets,goa_packets=(view(tmp_path),))
    return tuple(construct_primary_owner(PrimaryConfiguration(w,8,'a'*64,readout_mult=1.5),assets) for w in (8,12,16))


def test_full_literal_family_every_alias_shared_roles_unique_counts_no_updates(tmp_path,actual_core_assets):
    ms=members(tmp_path,actual_core_assets);before=[[p.detach().clone() for p in m.parameters()] for m in ms]
    rng=torch.get_rng_state().clone();family=full_owner_mup_width_family(*ms,**AXES);m=ms[-1]
    roles=full_owner_mup_parameter_roles(m,**AXES);ledger=parameter_ledger(m,roles)
    assert set(roles)==set(dict(m.named_parameters(remove_duplicate=False)))
    assert ledger.total_numel==unique_parameter_inventory(m)['total_parameters']==sum(p.numel() for p in m.parameters())
    assert len(roles)>len(tuple(m.parameters()))
    assert not any(r.role=='custom' for r in roles.values())
    residual=[(n,r) for n,r in roles.items() if n.endswith('reconstruction.latent_projection.weight')]
    assert residual and all(r.role=='input' and tuple(a.kind for a in r.axes)==('width','finite') for _,r in residual)
    normalizer=[n for n in roles if n.endswith('profile_gfn.normalizer.weight')];assert normalizer
    assert any(n.endswith('profile_gfn.forward_net.0.fixed_weight') for n in roles)
    assert any(n.endswith('profile_gfn.forward_net.0.context_weight') for n in roles)
    assert any(n.endswith('decoder.rnn.weight_hh_l0') for n in roles)
    assert any(n.endswith('goa.output.weight') for n in roles)
    opt=MuAdamW(m,family,roles,base_lr=.02,weight_decay=0.,decay_aliases=())
    assert not opt.state and sum(len(g['params']) for g in opt.param_groups)==len(tuple(m.parameters()))
    by={n:g for g in opt.param_groups for n in g['mup_aliases']}
    assert all(by[n]['lr']==.02 for n,_ in residual)
    for n in normalizer:assert by[n]['lr']==.02
    hidden=next(n for n in roles if n.endswith('backbone.layers.0.attention.qkv.weight'));assert by[hidden]['lr']==.01
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(p,v) and p.grad is None for m,values in zip(ms,before) for p,v in zip(m.parameters(),values))
    with pytest.raises(ValueError,match='not full corpus'):m.require_training_admission()


def test_conflicting_shared_native_finite_id_semantics_refuse_before_mutation(tmp_path,actual_core_assets):
    m=members(tmp_path,actual_core_assets)[0];before=[p.detach().clone() for p in m.parameters()];rng=torch.get_rng_state().clone()
    axes=dict(AXES,identifier_axis=('wrong-native-axis:0','wrong-native-axis:1'))
    with pytest.raises(ValueError,match='conflicting'):full_owner_mup_parameter_roles(m,**axes)
    assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(p,v) for p,v in zip(m.parameters(),before))


@pytest.mark.parametrize('drift',['hidden','snapshot','multiplier'])
def test_whole_source_family_refuses_finite_active_gfn_or_source_policy_drift(tmp_path,actual_core_assets,drift):
    from test_goa_objective import view
    assets=replace(actual_core_assets,goa_packets=(view(tmp_path),));ms=list(members(tmp_path,actual_core_assets))
    options=dict(width=16,base_width=8,source_snapshot_sha256='a'*64,readout_mult=1.5)
    if drift=='hidden':assets=replace(assets,reconstruction_hidden=6)
    elif drift=='snapshot':options['source_snapshot_sha256']='b'*64
    else:options['readout_mult']=2.
    ms[-1]=construct_primary_owner(PrimaryConfiguration(**options),assets)
    before=[[p.detach().clone() for p in m.parameters()] for m in ms];rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):full_owner_mup_width_family(*ms,**AXES)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(p,v) for m,values in zip(ms,before) for p,v in zip(m.parameters(),values))


def test_latent_residual_requires_actual_registered_policy_and_exact_decoder_identity(tmp_path,actual_core_assets):
    from mora3.model.full_owner_mup import _latent_roles
    m=members(tmp_path,actual_core_assets)[0];anchor=m.core.parent.parent
    roles=_latent_roles(anchor.reconstruction,anchor.decoder,anchor.profile_gfn)
    assert roles['latent_projection.weight'].role=='input'
    clone=copy.deepcopy(anchor.profile_gfn)
    with pytest.raises(ValueError,match='ownership'):_latent_roles(anchor.reconstruction,anchor.decoder,clone)
    with pytest.raises(ValueError):_latent_roles(anchor.reconstruction,anchor.nucleotide,anchor.profile_gfn)


def test_whole_registered_owner_paired_copy_checkpoint_scheduler_continuation(tmp_path,actual_core_assets):
    from test_molecule_profile_model import view as profile_view
    from mora3.reasoning.gfn_sampling import sample_paths,score_paths
    from mora3.model.profile_gfn import conditional_trajectory_balance
    ms=members(tmp_path,actual_core_assets);family=full_owner_mup_width_family(*ms,**AXES);model=ms[-1]
    def optimizer(m):return MuAdamW(m,family,full_owner_mup_parameter_roles(m,**AXES),base_lr=.001,weight_decay=0.,decay_aliases=())
    opt=optimizer(model);scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.5)
    def step(m,o,s,seed):
        o.zero_grad(set_to_none=True);p=m.core.parent.parent.profile_gfn
        context=p.context((profile_view('profile_to_molecule'),)*2)
        paths=sample_paths(p,2,context=context,generator=torch.Generator().manual_seed(seed),optional_stop=False)
        terminal=paths['states'][:,-1];reward=(-.5*terminal.square().sum(-1)).detach()
        loss=conditional_trajectory_balance(p.log_partition(context),**score_paths(p,paths,context=context),
            log_reward=reward,mask=paths['mask'])['loss']
        assert torch.isfinite(loss);loss.backward();o.step();s.step();o.zero_grad(set_to_none=True)
        return loss.detach()
    step(model,opt,scheduler,1901)
    before=[p.detach().clone() for p in model.parameters()];rng=torch.get_rng_state().clone()
    clone,copt,cscheduler=copy.deepcopy((model,opt,scheduler))
    assert cscheduler.optimizer is copt and torch.equal(rng,torch.get_rng_state())
    clone.get_extra_state();assert clone.core.parent.parent.reconstruction.decoder is clone.core.parent.parent.decoder
    assert clone.core.parent.parent.profile_gfn.profiles is clone.core.parent.parent.profile
    checkpoint=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict()))
    restored=members(tmp_path,actual_core_assets)[-1];ropt=optimizer(restored)
    rscheduler=torch.optim.lr_scheduler.StepLR(ropt,step_size=1,gamma=.5)
    restored.load_state_dict(checkpoint['model']);ropt.load_state_dict(checkpoint['optimizer']);rscheduler.load_state_dict(checkpoint['scheduler'])
    a=step(clone,copt,cscheduler,1902);b=step(restored,ropt,rscheduler,1902)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for x,y in zip(clone.parameters(),restored.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert cscheduler.state_dict()==rscheduler.state_dict()
    assert [g['lr'] for g in copt.param_groups]==[g['lr'] for g in ropt.param_groups]
    assert all(torch.equal(p,v) for p,v in zip(model.parameters(),before)), 'copied optimizer must not update original owner'
    for m,o in ((clone,copt),(restored,ropt)):
        owned={id(p) for p in m.parameters()}
        assert all(id(p) in owned for g in o.param_groups for p in g['params'])


def test_all_registered_optional_admet_mcsa_pubchem_cartesian_family_aliases(tmp_path,actual_core_assets):
    from test_full_owner import _native_extension_assets
    from test_mcsa_role_list import vocab
    assets=replace(_native_extension_assets(actual_core_assets),mcsa_role_vocabulary=vocab())
    ms=members(tmp_path,assets);family=full_owner_mup_width_family(*ms,**AXES);m=ms[-1]
    assert all(getattr(m,name)is not None for name in ('admet','mcsa','pubchem','cartesian'))
    roles=full_owner_mup_parameter_roles(m,**AXES)
    for prefix in ('admet.','mcsa.','pubchem.','cartesian.'):
        assert any(n.startswith(prefix) for n in roles)
    assert family.total_numel[-1]==sum(p.numel() for p in m.parameters())
    opt=MuAdamW(m,family,roles,base_lr=.001,weight_decay=0.,decay_aliases=())
    assert not opt.state
    anchor=m.core.parent.parent
    assert m.pubchem.shared is m.admet.shared is m.mcsa.residue.shared is anchor.annotation
    assert m.cartesian.backbone is anchor.annotation.backbone
    ledger=parameter_ledger(m,roles)
    row=next(r for r in ledger.parameters if 'cartesian.graph.layers.0.attention.qkv.weight' in r.aliases)
    assert 'cartesian.backbone.layers.0.attention.qkv.weight' in row.aliases
    assert 'pubchem.shared.backbone.layers.0.attention.qkv.weight' in row.aliases
    assert any(n.startswith('core.') for n in row.aliases)


def test_actual_source_text_alias_wrapper_family_and_drift_refusal(tmp_path,actual_core_assets):
    from test_full_owner import _native_extension_assets
    assets=replace(_native_extension_assets(actual_core_assets),source_text_max_target_bytes=4096)
    ms=members(tmp_path,assets);family=full_owner_mup_width_family(*ms,**AXES);m=ms[-1]
    assert m.text.decoder is m.pubchem
    roles=full_owner_mup_parameter_roles(m,**AXES)
    assert roles['text.decoder.rnn.weight_hh_l0']==roles['pubchem.rnn.weight_hh_l0']
    assert family.total_numel[-1]==sum(p.numel() for p in m.parameters())
    m.text.max_total_target_bytes=4097
    with pytest.raises(ValueError,match='drift'):full_owner_mup_parameter_roles(m,**AXES)
