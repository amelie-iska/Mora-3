"""Assay wrappers share registered weight owners and fail before partial recovery."""
import copy
from dataclasses import replace
import pytest
import torch

from mora3.data.nci_current_assays import NciAssaySourceAxes,_all_partitions
from mora3.model.expansionrx_native_model import expansionrx_official_contracts
from mora3.model.full_owner import PrimaryConfiguration,construct_primary_owner,unique_parameter_inventory
from mora3.model.full_owner_mup import full_owner_mup_parameter_roles
from test_expansionrx_native_model import make_axes
from test_full_owner import actual_core_assets,cases,_construct,_native_extension_assets
from test_full_owner_mup import AXES


def test_complete_assay_aliases_exact_contracts_and_preflight(tmp_path,actual_core_assets):
    base=_native_extension_assets(actual_core_assets)
    exp=make_axes();existing=expansionrx_official_contracts(exp)[0]
    assets=replace(base,admet_contracts=base.admet_contracts+(existing,),
        nci_assay_source_axes=NciAssaySourceAxes('a'*64,'b'*64,_all_partitions(),580,714),
        expansionrx_source_axes=exp)
    model=_construct(tmp_path,assets)
    assert model.nci.decoder is model.expansionrx.decoder is model.pubchem
    assert model.expansionrx.admet is model.admet
    assert len(model.admet.contracts)==9
    assert model.get_extra_state()['schema']=='mora3-source-bound-owner-v10'
    rows=unique_parameter_inventory(model)['parameters']
    for required in ('nci.decoder.embedding.weight','expansionrx.decoder.embedding.weight','pubchem.embedding.weight'):
        assert required in next(r['aliases'] for r in rows if 'pubchem.embedding.weight' in r['aliases'])
    assert len(full_owner_mup_parameter_roles(model,**AXES))==sum(len(r['aliases']) for r in rows)
    state=copy.deepcopy(model.state_dict())
    clone=copy.deepcopy(model);clone.load_state_dict(state)
    state['_extra_state']['nci']['max_batch_size']=True
    with pytest.raises(ValueError):
        clone.load_state_dict(state)
    assert clone.get_extra_state()==model.get_extra_state()


@pytest.mark.parametrize('invalid',[dict(nci_max_total_target_bytes=579),dict(expansionrx_max_batch_size=True),
    dict(pubchem_source_root=None),dict(nci_assay_source_axes=None,nci_max_batch_size=1)])
def test_assay_factory_rejects_incomplete_resources_before_rng(tmp_path,actual_core_assets,invalid):
    assets=replace(_native_extension_assets(actual_core_assets),
        nci_assay_source_axes=NciAssaySourceAxes('a'*64,'b'*64,_all_partitions(),580,714),
        expansionrx_source_axes=make_axes())
    before=torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),replace(assets,**invalid))
    assert torch.equal(before,torch.get_rng_state())
