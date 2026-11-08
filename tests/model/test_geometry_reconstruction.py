"""Geometry must reach existing shared heads without changing legacy losses."""
import copy
import hashlib
from dataclasses import replace
import pytest
import torch
from test_reconstruction_cooperative import build as reconstruction_build
from test_segmentation_inference import setup
from mora3.model import reconstruction_cooperative as module


def build(tmp_path, device):
    base, packet = reconstruction_build(tmp_path, device)
    assert hasattr(module, 'GeometryReconstructionCooperativeModel'), 'fourteen-objective geometry interface missing'
    model = module.GeometryReconstructionCooperativeModel(
        base.native, base.annotation, base.residue.head.feature_axis, base.profile.axes,
        codec=base.decoder.codec, max_tokens=64,
        annotation_registry=__import__('annotation_fixture').fixture_registry(),
        family_axis=base.rfam.family_axis, rfam_manifest_sha256='a'*64, goa_packets=(packet,),
        reconstruction_dimension=2, reconstruction_horizon=3, reconstruction_hidden=5,
        reconstruction_source={'pool':'fixture'}, geometry_id_width=16).to(device).double()
    return model, packet


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_geometry_reference_gradients_and_shared_ownership(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(13501)
    task, native, request = setup(tmp_path, device)
    model, packet = build(tmp_path, device)
    assert model.geometry_fusion.image is model.native.image
    assert model.geometry_fusion.geometry.graph.layers is model.annotation.backbone.layers
    binding = dict(request=request, native=native,
        admitted_task_sha256=hashlib.sha256(task.canonical_bytes()).hexdigest(),
        prediction_asset='prediction.pt')
    actual = model.geometry_objective(task, **binding)
    from mora3.model.morphology import foreground_objective
    from mora3.model.instance_affinity import affinity_objective
    from mora3.data.segmentation_image import SegmentationImageInput
    image = SegmentationImageInput(request.artifacts[0], task.image_dtype, bytes.fromhex(task.image_hex))
    z, audit = model.geometry_fusion(image, request, native, prediction_asset='prediction.pt')
    truth = task.instances(device=device)
    reference = foreground_objective(z[0], truth) + affinity_objective(z[1:], truth)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    params = tuple(model.parameters())
    ga = torch.autograd.grad(actual, params, allow_unused=True)
    gr = torch.autograd.grad(reference, params, allow_unused=True)
    for a,b in zip(ga,gr):
        assert (a is None) == (b is None)
        if a is not None: torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-10)
    gradients = dict(zip(map(id,params),ga))
    assert gradients[id(model.annotation.backbone.feature.weight)] is not None
    assert gradients[id(model.geometry_fusion.geometry.graph.incidence.weight)] is not None
    assert gradients[id(model.native.response.response_head.weight)] is None
    assert all(p.grad is None for p in native.model.parameters())
    optimizer = torch.optim.Adam(model.parameters())
    ids=[id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))
    assert torch.isfinite(model.goa_objective((packet,)))
    cloned = copy.deepcopy(model)
    assert cloned.geometry_fusion.image is cloned.native.image
    assert cloned.geometry_fusion.image is not model.native.image
    model.load_state_dict(copy.deepcopy(model.state_dict()))
    old = copy.deepcopy(model.get_extra_state()); old['geometry']['geometry']['id_width']=99
    with pytest.raises(ValueError): model.set_extra_state(old)
    # Targets never enter pixel/geometry conditioning; changing labels affects loss only.
    changed=replace(task, labels_hex='0100'*12)
    _, z2, audit2 = model.geometry_fusion.training_objective(changed, request, native,
        admitted_task_sha256=hashlib.sha256(changed.canonical_bytes()).hexdigest(), prediction_asset='prediction.pt')
    assert torch.equal(z,z2)
    assert audit['input_graph'].canonical_bytes()==audit2['input_graph'].canonical_bytes()


@pytest.mark.parametrize('change', ['digest','checkpoint','hidden','test','source'])
def test_geometry_reconstruction_rejects_invalid_binding(tmp_path, change):
    task,native,request=setup(tmp_path)
    model,_=build(tmp_path,'cpu')
    if change=='test': task=replace(task,split='test')
    if change=='source': task=replace(task,revision='other')
    if change=='checkpoint': request=replace(request,head_checkpoint='wrong')
    if change=='hidden': request=replace(request,artifacts=(replace(request.artifacts[0],role='target'),))
    digest=hashlib.sha256(task.canonical_bytes()).hexdigest()
    if change=='digest':digest='0'*64
    with pytest.raises(ValueError):model.geometry_objective(task,request=request,native=native,
        admitted_task_sha256=digest,prediction_asset='prediction.pt')
