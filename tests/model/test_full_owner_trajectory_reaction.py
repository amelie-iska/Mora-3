"""The two native additions preserve shared owners and account for only new readout weights."""
import copy
from dataclasses import replace

import pytest
import torch

from mora3.data.cema_native_trajectories import ARMS_SHA
from mora3.model.cema_trajectory_model import K
from mora3.model.full_owner import PrimaryConfiguration,construct_primary_owner,unique_parameter_inventory
from mora3.model.full_owner_mup import full_owner_mup_parameter_roles
from test_full_owner import actual_core_assets,cases,_construct,_native_extension_assets
from test_full_owner_mup import AXES
from test_reactzyme_product import axes


def test_native_trajectory_and_reaction_owners_all_aliases_and_recovery(tmp_path,actual_core_assets):
    base=_native_extension_assets(actual_core_assets)
    baseline=_construct(tmp_path,base)
    model=_construct(tmp_path,replace(base,cema_trajectory_source_sha256=ARMS_SHA,reactzyme_source_axes=axes()))
    assert model.cema_trajectory.shared is model.core.parent.parent.annotation
    assert model.reactzyme.decoder is model.pubchem
    count=unique_parameter_inventory(model)
    assert count['total_parameters']-unique_parameter_inventory(baseline)['total_parameters']==(K+2)*(8+1)
    assert len(full_owner_mup_parameter_roles(model,**AXES))==count['alias_names']
    assert model.get_extra_state()['schema']=='mora3-source-bound-owner-v11'
    state=copy.deepcopy(model.state_dict());clone=copy.deepcopy(model);clone.load_state_dict(state)
    state['_extra_state']['cema_trajectory']['source_manifest_sha256']='0'*64
    with pytest.raises(ValueError):clone.load_state_dict(state)
    assert clone.get_extra_state()==model.get_extra_state()


@pytest.mark.parametrize('options',[dict(cema_trajectory_source_sha256='f'*64),
    dict(reactzyme_source_axes=axes(),payload_max_tokens=1),
    dict(reactzyme_source_axes=axes(),reactzyme_max_total_target_bytes=1)])
def test_trajectory_and_reaction_preflight_before_rng(tmp_path,actual_core_assets,options):
    base=_native_extension_assets(actual_core_assets)
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),replace(base,**options))
    assert torch.equal(rng,torch.get_rng_state())
