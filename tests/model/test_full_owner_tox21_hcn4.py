"""Native toxicology reuses all weights; HCN4 adds only its marker readouts."""
import copy
from dataclasses import replace

import pytest
import torch

from mora3.data.hcn4_native_laterality import LEDGER_SHA,CATEGORIES,MARKERS
from mora3.data.tox21_current import Tox21CurrentAxes
from mora3.model.full_owner import PrimaryConfiguration,construct_primary_owner,unique_parameter_inventory
from mora3.model.full_owner_mup import full_owner_mup_parameter_roles
from test_full_owner import actual_core_assets,cases,_construct,_native_extension_assets
from test_full_owner_mup import AXES


def test_tox21_hcn4_aliases_and_complete_recovery(tmp_path,actual_core_assets):
    base=_native_extension_assets(actual_core_assets)
    baseline=_construct(tmp_path,base)
    tox=Tox21CurrentAxes('a'*64,'b'*64,(4,2,1),332)
    model=_construct(tmp_path,replace(base,tox21_current_axes=tox,hcn4_laterality_source_sha256=LEDGER_SHA))
    anchor=model.core.parent.parent
    assert model.tox21_source.tox21 is anchor.tox21
    assert model.tox21_source.decoder is model.pubchem
    assert model.hcn4_laterality.shared is anchor.annotation
    count=unique_parameter_inventory(model)
    assert count['total_parameters']-unique_parameter_inventory(baseline)['total_parameters']==len(MARKERS)*len(CATEGORIES)*(8+1)
    assert len(full_owner_mup_parameter_roles(model,**AXES))==count['alias_names']
    assert model.get_extra_state()['schema']=='mora3-source-bound-owner-v12'
    clone=copy.deepcopy(model);state=copy.deepcopy(model.state_dict());clone.load_state_dict(state)
    state['_extra_state']['hcn4_laterality']['native_count_ledger_sha256']='0'*64
    with pytest.raises(ValueError):clone.load_state_dict(state)
    assert clone.get_extra_state()==model.get_extra_state()


@pytest.mark.parametrize('options',[dict(hcn4_laterality_source_sha256='f'*64),
    dict(tox21_current_axes=Tox21CurrentAxes('a'*64,'b'*64,(4,2,1),332),tox21_max_total_target_bytes=331),
    dict(tox21_max_batch_size=1)])
def test_tox21_hcn4_preflight_before_rng(tmp_path,actual_core_assets,options):
    base=_native_extension_assets(actual_core_assets)
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),replace(base,**options))
    assert torch.equal(rng,torch.get_rng_state())
