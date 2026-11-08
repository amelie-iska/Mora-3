"""Exact STRING and UniProt source owner aliases, bounds, and checkpoint recovery."""
from dataclasses import replace
import copy

import pytest
import torch

from mora3.data.string_score_current import StringScoreAxes
from mora3.model.full_owner import PrimaryConfiguration,construct_primary_owner,unique_parameter_inventory
from mora3.model.full_owner_mup import full_owner_mup_parameter_roles
from mora3.model.uniprot_native_objectives import _GUARD
from test_full_owner import actual_core_assets,cases,_construct,_native_extension_assets
from test_full_owner_mup import AXES


def _string():
    return StringScoreAxes('a'*64,'b'*64,'c'*64,'d'*64,2,14507,'ACDEFGHIKLMNPQRSTVWY')


def test_exact_aliases_and_legacy_extension_checkpoint(tmp_path,actual_core_assets):
    base=replace(_native_extension_assets(actual_core_assets),payload_max_tokens=37_000)
    baseline=_construct(tmp_path,base)
    model=_construct(tmp_path,replace(base,string_score_axes=_string(),uniprot_native_annotation_guard_sha256=_GUARD))
    assert model.string_score.shared is model.core.parent.parent.annotation
    assert model.uniprot_native_annotation.decoder is model.pubchem
    inventory=unique_parameter_inventory(model)
    assert inventory['total_parameters']-unique_parameter_inventory(baseline)['total_parameters']==4*(8+1)
    assert len(full_owner_mup_parameter_roles(model,**AXES))==inventory['alias_names']
    assert model.get_extra_state()['schema']=='mora3-source-bound-owner-v14'
    state=copy.deepcopy(model.state_dict());clone=copy.deepcopy(model);clone.load_state_dict(state)
    state['_extra_state']['uniprot_native_annotation']['source_prospective_guard_sha256']='f'*64
    with pytest.raises(ValueError):clone.load_state_dict(state)
    assert clone.get_extra_state()==model.get_extra_state()


@pytest.mark.parametrize('bad',[dict(payload_max_tokens=36999,uniprot_native_annotation_guard_sha256=_GUARD),
    dict(uniprot_native_annotation_guard_sha256='f'*64),dict(string_score_axes=_string(),string_score_max_batch_sequence_bytes=29013),
    dict(uniprot_native_annotation_guard_sha256=_GUARD,uniprot_native_annotation_max_target_bytes=35687)])
def test_string_uniprot_resource_refusals_before_rng(tmp_path,actual_core_assets,bad):
    base=replace(_native_extension_assets(actual_core_assets),payload_max_tokens=37_000)
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),replace(base,**bad))
    assert torch.equal(rng,torch.get_rng_state())
