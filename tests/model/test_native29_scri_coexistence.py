"""Staged29-head source factory must coexist with existing SCRI and stay closed."""
from dataclasses import fields,replace
import copy
import pytest
import torch

from test_full_owner import actual_core_assets,cases
from test_goa_objective import view as goa_view
from mora3.model.full_owner import PrimaryConfiguration,SourceAssemblyInputs,construct_primary_owner,unique_parameter_inventory
from mora3.model.full_owner_training_admission import _owner_identity,_NATIVE29_HEADS,_REVIEWED_RELEASES
from mora3.model.full_owner_mup import full_owner_mup_parameter_roles
from mora3.model.parameterization import parameter_ledger
from mora3.extensions.scri.model import SCRITargetAxis
from mora3.extensions.scri.native_owner import SpatialBudget


def test_complete_optional_source_contract_preserves_both_old_and_new_branches():
    names={f.name for f in fields(SourceAssemblyInputs)}
    assert {'scri_axis','scri_spatial_budget','betse_source','raw_cell_axis'}<=names
    from mora3.model import primary_source_factory
    assert callable(primary_source_factory.load_reviewed_native29_source_inputs)
    assert _REVIEWED_RELEASES==frozenset()


def test_invalid_betse_or_raw_gene_source_is_refused_before_model_rng(tmp_path,actual_core_assets):
    config=PrimaryConfiguration(8,8,'a'*64)
    for wrong in (replace(actual_core_assets,betse_source=object()),
                  replace(actual_core_assets,raw_cell_axis=object())):
        rng=torch.get_rng_state().clone()
        with pytest.raises(ValueError):
            construct_primary_owner(config,wrong)
        assert torch.equal(rng,torch.get_rng_state())


def test_real_native_owner_retains_scri_aliases_while_admission_remains_closed(tmp_path,actual_core_assets):
    sources=replace(actual_core_assets,goa_packets=(goa_view(tmp_path),),
        scri_axis=SCRITargetAxis('d'*64,(('rna','gene','count'),),'poisson'),
        scri_spatial_budget=SpatialBudget(32,1025,5000000,2000000,1))
    model=construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),sources)
    assert model.scri.core.shared is model.core.parent.parent.annotation
    assert model.betse is model.raw_cell is None
    roles=full_owner_mup_parameter_roles(model,identifier_axis=('context','observation'),
        geometry_identifier_axis=tuple('geometry:'+str(i)for i in range(16)),
        color_identifier_axis=('color:0','color:1'))
    ledger=parameter_ledger(model,roles)
    inventory=unique_parameter_inventory(model)
    assert ledger.total_numel==inventory['total_parameters']
    assert inventory['unique_parameter_tensors']==len(list(model.parameters()))
    checkpoint=copy.deepcopy(model.state_dict());clone=copy.deepcopy(model)
    clone.load_state_dict(checkpoint)
    assert clone.get_extra_state()==model.get_extra_state()
    before={name:p.detach().clone()for name,p in model.named_parameters()}
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='not full corpus'):
        model.require_training_admission(campaign_sha256='a'*64,trial_id='hp-01-2026110201')
    with pytest.raises(ValueError,match='No repository-reviewed'):
        model.authorize_primary_campaign_training(tmp_path,
            path='outputs/absent-release.json',expected_sha256='e'*64)
    # An N29-only admission declaration must NOT quietly ignore a registered
    # extra SCRI module, and cannot stand in for an extended campaign release.
    with pytest.raises(ValueError,match='branch inventory'):
        _owner_identity(model,_NATIVE29_HEADS)
    assert all(torch.equal(p,before[name])for name,p in model.named_parameters())
    assert torch.equal(rng,torch.get_rng_state())
