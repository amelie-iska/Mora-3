import hashlib
from dataclasses import replace
import pytest
import torch
from test_segmentation_inference import setup
from mora3.model.geometry_pixel_fusion import GeometryPixelFusion
from mora3.model.cooperative_training import CooperativeModel


def digest(t): return hashlib.sha256(t.canonical_bytes()).hexdigest()

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_bound_training_objective_target_exclusion_and_gradients(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    t,n,r=setup(tmp_path,device)
    t=replace(t,labels_hex='0100'*6+'0000'*6)
    f=GeometryPixelFusion(CooperativeModel(('A','B'),width=8).image,id_width=16).to(device)
    loss,z,audit=f.training_objective(t,r,n,admitted_task_sha256=digest(t),prediction_asset='p.pt')
    from mora3.model.morphology import foreground_objective
    from mora3.model.instance_affinity import affinity_objective
    labels=t.instances(device=device)
    torch.testing.assert_close(loss,foreground_objective(z[0],labels)+affinity_objective(z[1:],labels))
    changed=replace(t,labels_hex='0200'*12)
    other,z2,audit2=f.training_objective(changed,r,n,admitted_task_sha256=digest(changed),prediction_asset='p.pt')
    assert torch.equal(z,z2)
    assert audit['input_graph'].canonical_bytes()==audit2['input_graph'].canonical_bytes()
    assert loss!=other
    loss.backward()
    assert f.image.head.weight.grad.abs().sum()>0
    assert f.geometry.graph.incidence.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in n.model.parameters())

@pytest.mark.parametrize('change',['split','binding','parent','context','origin','role','source','asset','direction','scale'])
def test_training_rejects_nonadmitted_or_mismatched_source(tmp_path,change):
    t,n,r=setup(tmp_path)
    sha=digest(t)
    if change=='split': t=replace(t,split='test');sha=digest(t)
    if change=='binding': sha='0'*64
    if change=='parent': t=replace(t,parent_group='other');sha=digest(t)
    if change=='context': t=replace(t,strain='other');sha=digest(t)
    if change=='source': t=replace(t,revision='other');sha=digest(t)
    if change=='asset': t=replace(t,source_id='other');sha=digest(t)
    if change=='direction': t=replace(t,direction='mask_to_image');sha=digest(t)
    if change=='scale': t=replace(t,pixel_size_um=.1);sha=digest(t)
    if change=='origin': r=replace(r,artifacts=(replace(r.artifacts[0],evidence='model_generated',checkpoint='generator'),))
    if change=='role': r=replace(r,artifacts=(replace(r.artifacts[0],role='desired',evidence='desired'),))
    f=GeometryPixelFusion(CooperativeModel(('A','B'),width=8).image,id_width=16)
    with pytest.raises(ValueError):
        f.training_objective(t,r,n,admitted_task_sha256=sha,prediction_asset='p.pt')
