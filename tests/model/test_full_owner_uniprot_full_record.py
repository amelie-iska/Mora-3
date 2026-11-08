"""Exact optional full-record factory owner, checkpoint and early refusal."""
from dataclasses import replace
import copy

import pytest
import torch

from mora3.data.uniprot_full_record import FullRecordUniProtAxes,FullRecordFieldAxis
from mora3.model.full_owner import PrimaryConfiguration,construct_primary_owner,unique_parameter_inventory
from mora3.model.full_owner_mup import full_owner_mup_parameter_roles
from mora3.model.uniprot_full_record_model import full_record_stream_bound
from test_full_owner import actual_core_assets,cases,_construct,_native_extension_assets
from test_full_owner_mup import AXES


def _source():
    return FullRecordUniProtAxes('a'*64,'b'*64,3,35_213,(
        FullRecordFieldAxis('uniProtKBCrossReferences','list',676_562,2),))


def test_full_record_alias_checkpoint_and_roles(tmp_path,actual_core_assets):
    base=replace(_native_extension_assets(actual_core_assets),payload_max_tokens=37_000)
    baseline=_construct(tmp_path,base)
    model=_construct(tmp_path,replace(base,uniprot_full_record_axes=_source()))
    assert model.uniprot_full_record.decoder is model.pubchem
    old=unique_parameter_inventory(baseline);new=unique_parameter_inventory(model)
    assert old['total_parameters']==new['total_parameters']
    assert new['alias_names']>old['alias_names']
    assert len(full_owner_mup_parameter_roles(model,**AXES))==new['alias_names']
    assert model.get_extra_state()['schema']=='mora3-source-bound-owner-v15'
    state=copy.deepcopy(model.state_dict());clone=copy.deepcopy(model)
    clone.load_state_dict(state)
    state['_extra_state']['uniprot_full_record']['source']['census_sha256']='f'*64
    with pytest.raises(ValueError):clone.load_state_dict(state)
    assert clone.get_extra_state()==model.get_extra_state()


@pytest.mark.parametrize('bad',[dict(uniprot_full_record_max_batch_size=17),
    dict(uniprot_full_record_axes=_source(),pubchem_source_root=None),
    dict(uniprot_full_record_axes=_source(),payload_max_tokens=full_record_stream_bound(_source())-1),
    dict(uniprot_full_record_axes=_source(),uniprot_full_record_max_target_bytes=676_561)])
def test_full_record_refusals_before_rng(actual_core_assets,bad):
    base=replace(_native_extension_assets(actual_core_assets),payload_max_tokens=37_000)
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='full-record UniProt|source-bound PubChem root/manifest/SHA'):
        construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),replace(base,**bad))
    assert torch.equal(rng,torch.get_rng_state())
