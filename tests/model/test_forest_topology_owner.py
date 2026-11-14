"""Optional native owner registration and width-axis consistency; zero updates."""
from dataclasses import replace
import copy
import pytest
import torch
from test_forest_flow_owner import actual_core_assets,cases,assets,AXES
from mora3.model.full_owner import construct_primary_owner,PrimaryConfiguration,unique_parameter_inventory
from mora3.model.full_owner_mup import full_owner_mup_parameter_roles,full_owner_mup_width_family
from mora3.model.forest_topology import ForestTopologySpec,NativeForestTopologyHead,native_forest_topology_objective
from mora3.reasoning.forest_topology import TopologyBounds
from test_forest_flow import SOURCE,problem


def sources(tmp_path,source):
    return replace(assets(tmp_path,source),forest_topology_spec=ForestTopologySpec(TopologyBounds(8,7,7),'f'*64))


@pytest.mark.parametrize('width',(8,12,16))
def test_actual_factory_shares_native_field_and_tracks_only_new_parameters(tmp_path,actual_core_assets,width):
    source=sources(tmp_path,actual_core_assets)
    owner=construct_primary_owner(PrimaryConfiguration(width,8,SOURCE),source)
    old=construct_primary_owner(PrimaryConfiguration(width,8,SOURCE),replace(source,forest_topology_spec=None))
    assert type(owner.forest_topology)is NativeForestTopologyHead
    assert owner.forest_topology.field is owner.forest_flow
    assert owner.get_extra_state()['forest_topology']==owner.forest_topology.get_extra_state()
    assert unique_parameter_inventory(owner)['total_parameters']-unique_parameter_inventory(old)['total_parameters']==10*width+2
    assert 'forest_topology'not in old.get_extra_state()
    assert not any(n.startswith('forest_topology.')for n in old.state_dict())
    roles=full_owner_mup_parameter_roles(owner,**AXES)
    assert set(roles)==set(dict(owner.named_parameters(remove_duplicate=False)))
    assert unique_parameter_inventory(owner)['alias_names']>unique_parameter_inventory(owner)['unique_parameter_tensors']


def test_complete_width_family_keeps_finite_action_support_and_aliases(tmp_path,actual_core_assets):
    source=sources(tmp_path,actual_core_assets)
    owners=[construct_primary_owner(PrimaryConfiguration(w,8,SOURCE),source)for w in(8,12,16)]
    family=full_owner_mup_width_family(*owners,**AXES)
    assert family.total_numel==tuple(sum(p.numel()for p in o.parameters())for o in owners)
    changed=construct_primary_owner(PrimaryConfiguration(16,8,SOURCE),replace(source,
        forest_topology_spec=replace(source.forest_topology_spec,construction_policy_sha256='a'*64)))
    with pytest.raises(ValueError):full_owner_mup_width_family(owners[0],owners[1],changed,**AXES)


@pytest.mark.parametrize('fault',('missing-flow','bounds','type'))
def test_factory_preflight_before_rng(fault,tmp_path,actual_core_assets):
    source=sources(tmp_path,actual_core_assets)
    if fault=='missing-flow':source=replace(source,forest_flow_spec=None)
    if fault=='bounds':source=replace(source,forest_topology_spec=ForestTopologySpec(TopologyBounds(17,16,16),'a'*64))
    if fault=='type':source=replace(source,forest_topology_spec={})
    before=torch.get_rng_state().clone()
    with pytest.raises(ValueError):construct_primary_owner(PrimaryConfiguration(8,8,SOURCE),source)
    assert torch.equal(before,torch.get_rng_state())


def test_full_checkpoint_bad_topology_policy_never_partially_copies(tmp_path,actual_core_assets):
    owner=construct_primary_owner(PrimaryConfiguration(8,8,SOURCE),sources(tmp_path,actual_core_assets))
    state=copy.deepcopy(owner.state_dict());before={n:p.clone()for n,p in owner.named_parameters()}
    state['forest_topology.action_readout.weight'].add_(1)
    state['forest_topology._extra_state']['spec']['construction_policy_sha256']='1'*64
    with pytest.raises(ValueError):owner.load_state_dict(state)
    assert all(torch.equal(before[n],p)for n,p in owner.named_parameters())
    clone=copy.deepcopy(owner);clone.load_state_dict(copy.deepcopy(owner.state_dict()))
    assert clone.forest_topology.field is clone.forest_flow
    r,t=problem();torch.testing.assert_close(owner.forest_topology(r,t,root_count=1,actions=()),
        clone.forest_topology(r,t,root_count=1,actions=()),atol=0,rtol=0)


def test_native_entry_still_requires_real_full_owner_admission_first(tmp_path,actual_core_assets):
    owner=construct_primary_owner(PrimaryConfiguration(8,8,SOURCE),sources(tmp_path,actual_core_assets))
    before=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='not full corpus'):
        native_forest_topology_objective(owner,'invalid source deliberately not inspected')
    assert torch.equal(before,torch.get_rng_state())
