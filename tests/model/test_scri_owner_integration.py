"""Isolated full factory proposal; no corpus release or optimizer execution."""
import copy
from dataclasses import replace
import pytest
import torch
from test_full_owner import actual_core_assets,cases
from test_goa_objective import view
from mora3.model.full_owner import construct_primary_owner,PrimaryConfiguration,unique_parameter_inventory
from mora3.model.full_owner_mup import full_owner_mup_parameter_roles
from mora3.model.parameterization import parameter_ledger
from mora3.extensions.scri.model import SCRITargetAxis
from mora3.extensions.scri.native_owner import SpatialBudget


def test_full_factory_scri_registration_aliases_roles_checkpoint_and_closed_admission(tmp_path,actual_core_assets):
    sources=replace(actual_core_assets,goa_packets=(view(tmp_path),),
        scri_axis=SCRITargetAxis('d'*64,(('rna','gene','count'),),'poisson'),
        scri_spatial_budget=SpatialBudget(32,1025,5000000,2000000,1))
    m=construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),sources)
    assert m.scri.core.shared is m.core.parent.parent.annotation
    assert m.scri.graph.layers is m.core.parent.parent.native.response.backbone.layers
    assert m.scri.graph.incidence.in_features==64
    roles=full_owner_mup_parameter_roles(m,identifier_axis=('context','observation'),
        geometry_identifier_axis=tuple('geometry:'+str(i) for i in range(16)),
        color_identifier_axis=('color:0','color:1'))
    ledger=parameter_ledger(m,roles);inventory=unique_parameter_inventory(m)
    assert ledger.total_numel==inventory['total_parameters']
    assert inventory['alias_names']>inventory['unique_parameter_tensors']
    state=copy.deepcopy(m.state_dict());clone=copy.deepcopy(m);clone.load_state_dict(state)
    assert clone.get_extra_state()==m.get_extra_state()
    with pytest.raises(ValueError,match='not full corpus'):m.require_training_admission()
    broken=copy.deepcopy(state);name='scri.graph.layers.0.attention.output.weight'
    broken[name]=broken[name].clone()+1
    with pytest.raises(ValueError,match='alias'):m.load_state_dict(broken)


def test_partial_scri_factory_source_binding_refuses_before_rng(tmp_path,actual_core_assets):
    sources=replace(actual_core_assets,scri_axis=SCRITargetAxis('d'*64,(('rna','gene','count'),),'poisson'))
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='together'):
        construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),sources)
    assert torch.equal(rng,torch.get_rng_state())
