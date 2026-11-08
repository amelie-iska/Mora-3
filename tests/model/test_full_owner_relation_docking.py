"""Native Reactome, BioGRID chemical and LinCx readouts keep exact shared owners."""
import copy
from dataclasses import replace

import pytest
import torch

from mora3.data.biogrid_chemical_current import BioGRIDChemicalAxes
from mora3.data.lincx_fetch_native_rows import SOURCE_SHA
from mora3.model.full_owner import PrimaryConfiguration,construct_primary_owner,unique_parameter_inventory
from mora3.model.full_owner_mup import full_owner_mup_parameter_roles
from test_biogrid_chemical_model import AXIS
from test_full_owner import actual_core_assets,cases,_construct
from test_full_owner_mup import AXES
from test_reactome_static_model import axes


def _biogrid():
    return BioGRIDChemicalAxes('a'*64,'b'*64,4,AXIS,(19,190,8,7))


def test_exact_relation_and_docking_owner_registration_and_recovery(tmp_path,actual_core_assets):
    base=_construct(tmp_path,actual_core_assets)
    model=_construct(tmp_path,replace(actual_core_assets,reactome_static_axes=axes(),
        biogrid_chemical_axes=_biogrid(),lincx_fetch_source_sha256=SOURCE_SHA))
    shared=model.core.parent.parent.annotation
    assert model.reactome_static.shared is model.biogrid_chemical.shared is model.lincx_fetch.shared is shared
    assert model.get_extra_state()['schema']=='mora3-source-bound-owner-v13'
    count=unique_parameter_inventory(model);previous=unique_parameter_inventory(base)['total_parameters']
    added=(2*axes().max_nodes+4)*8+4+(len(AXIS)+6)*9
    assert count['total_parameters']-previous==added
    assert len(full_owner_mup_parameter_roles(model,**AXES))==count['alias_names']
    state=copy.deepcopy(model.state_dict());clone=copy.deepcopy(model);clone.load_state_dict(state)
    state['_extra_state']['lincx_fetch']['source_manifest_sha256']='0'*64
    with pytest.raises(ValueError):clone.load_state_dict(state)
    assert clone.get_extra_state()==model.get_extra_state()


@pytest.mark.parametrize('bad',[dict(lincx_fetch_source_sha256='f'*64),dict(reactome_static_axes=axes(),reactome_max_attention_elements=30),
    dict(biogrid_chemical_axes=_biogrid(),biogrid_chemical_max_batch_size=True),dict(reactome_max_attention_elements=1)])
def test_exact_relation_resource_preflight_before_rng(tmp_path,actual_core_assets,bad):
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),replace(actual_core_assets,**bad))
    assert torch.equal(rng,torch.get_rng_state())
