"""Native explicit-muP owner participation; no optimizer execution or release."""
from dataclasses import replace
import pytest
import torch
from test_full_owner import actual_core_assets,cases
from test_goa_objective import view as goa_view
from test_molecule_profile_model import view
from mora3.model.full_owner import construct_primary_owner,PrimaryConfiguration,unique_parameter_inventory
from mora3.model.profile_gfn import ProfileGFNPolicy
from mora3.model.geometry_conditioning import GeometryConditioner
from mora3.reasoning.trajectory_history import NativePrefixContext
from mora3.reasoning.trajectory_gfn import PathSpaceReward,sample_native_path_batch,pathspace_objective,native_pathspace_objective


@pytest.mark.parametrize('width',(8,12,16))
def test_full_native_owner_shared_history_and_muP_policy_gradients(tmp_path,actual_core_assets,width):
    sources=replace(actual_core_assets,goa_packets=(goa_view(tmp_path),))
    torch.manual_seed(20261110)
    owner=construct_primary_owner(PrimaryConfiguration(width,8,'a'*64),sources).double()
    policies=[(n,m)for n,m in owner.named_modules()if type(m)is ProfileGFNPolicy]
    assert len(policies)==1
    path,policy=policies[0]
    geometries=[(n,m)for n,m in owner.named_modules()if type(m)is GeometryConditioner
        and m._source_backbone_reference[0]is policy.profiles.shared.backbone
        and m.config['id_width']>=policy.horizon+1]
    assert geometries
    geometry_path,geometry=geometries[0]
    assert policy.parameterization_policy=='mup-profile-gfn-v1'
    assert geometry.parameterization_policy=='mup-geometry-v1'
    before=unique_parameter_inventory(owner)
    history=NativePrefixContext(policy,geometry,source_context_sha256='a'*64,coefficient=.25)
    views=(view('profile_to_molecule'),)
    batch=sample_native_path_batch(policy,views,generator=torch.Generator().manual_seed(122),source_context_sha256='a'*64,history_context=history)
    guide=torch.tensor([[[0.,0.],[1.,0.],[1.,1.],[2.,1.]]],dtype=torch.float64)
    reward=PathSpaceReward(guide,source_sha256='b'*64,encoder_sha256='c'*64,alignment_sha256='d'*64,
        group_policy_sha256='e'*64,guide_groups=(('guide',),),query_groups=(('query',),),
        scale=1.,metric_weight=.1,topology_weight=.1,evidence_kind='synthetic_fixture')
    result=pathspace_objective(policy,views,batch,reward,history_context=history)
    result['loss'].backward()
    assert torch.isfinite(result['loss'])
    assert geometry.payload.embedding.weight.grad.abs().sum()>0
    assert policy.forward_head.weight.grad.abs().sum()>0
    assert policy.profiles.shared.backbone.feature.weight.grad.abs().sum()>0
    after=unique_parameter_inventory(owner)
    assert before==after
    with pytest.raises(ValueError,match='not full corpus'):
        native_pathspace_objective(owner,path,views,batch,reward,history_context=history,history_module_path=geometry_path)
