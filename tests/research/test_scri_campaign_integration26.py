"""New production-adapter qualification; no optimizer execution or admission."""
from pathlib import Path
from dataclasses import replace
import inspect
import pytest
import torch
from test_full_owner import actual_core_assets, cases
from test_goa_objective import view
from test_scri_native_routes import AXIS, envelope, finite_route
from mora3.model.full_owner import PrimaryConfiguration, construct_primary_owner
from mora3.extensions.scri.native_owner import SpatialBudget
from mora3.extensions.scri.native_routes import build_native_scri_binding
from mora3.research.mup_campaign import executable_adapter_sha256
from mora3.research.primary_native_routes import select_native_route_targets


@pytest.fixture
def setup(tmp_path,actual_core_assets):
    sources=replace(actual_core_assets,goa_packets=(view(tmp_path),),scri_axis=AXIS,
        scri_spatial_budget=SpatialBudget(16,257,300000,2000000,2))
    owner=construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),sources)
    path=tmp_path/'route';path.mkdir()
    route=finite_route(path,[envelope(missing=True),envelope()])
    return owner,route,build_native_scri_binding(owner,route)


def test_scri_binding_has_inspectable_campaign_fingerprint(setup):
    owner,route,binding=setup
    root=Path(inspect.getfile(build_native_scri_binding)).resolve().parents[4]
    digest=executable_adapter_sha256(binding.prepare,root=root)
    assert len(digest)==64
    assert inspect.ismethod(binding.prepare)
    assert binding.prepare.__func__.__closure__ is None


def test_direct_training_evaluator_cannot_bypass_owner_admission(setup):
    owner,route,binding=setup
    result=select_native_route_targets(binding,route.initial_cursor,max_windows=2)
    before=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='not full corpus'):
        result.targets.evaluate(owner)
    assert torch.equal(before,torch.get_rng_state())


def test_adapter_freezes_route_and_axis_but_not_model_weights(setup):
    owner,route,binding=setup
    adapter=binding.prepare.__self__
    first=adapter.executable_binding()
    with torch.no_grad():owner.scri.core.output.weight.add_(.01)
    assert adapter.executable_binding()==first
    route._batch_size+=1
    with pytest.raises(ValueError,match='binding|changed|drift'):
        adapter.executable_binding()
