"""Joint geometry model retains response heads and matched disabled parameters."""
import copy
import hashlib
import pytest
import torch
from test_segmentation_inference import setup
from test_sciplex_shared import task as response_task
from test_geometry_pixel_trainer import equal
from mora3.model import cooperative_training as module

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_joint_geometry_objectives_and_full_checkpoint(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(83)
    task, native, request = setup(tmp_path, device)
    m = module.GeometryCooperativeModel(('A','B'), width=8, id_width=16).to(device)
    assert m.fusion.image is m.image
    assert m.fusion.geometry.graph.layers is m.response.backbone.layers
    kwargs = dict(request=request, native=native,
        admitted_task_sha256=hashlib.sha256(task.canonical_bytes()).hexdigest(),
        prediction_asset='prediction.pt')
    opt = torch.optim.Adam(m.parameters(), lr=.001)
    def step(kind):
        opt.zero_grad(set_to_none=True)
        loss = m.geometry_objective(task, **kwargs) if kind == 'image' else m.objective(kind,[response_task(kind)])
        loss.backward(); opt.step()
        return loss.item()
    step('image')
    assert m.fusion.geometry.graph.incidence.weight.grad is not None
    assert m.response.response_head.weight.grad is None
    step('response')
    assert m.response.response_head.weight.grad is not None
    assert m.image.head.weight.grad is None
    from mora3.model.training import save_checkpoint, load_checkpoint
    binding=m.geometry_binding({'image':(task,), 'response':(response_task(),)},
        batch_size=1, image_requests=(request,), native=native)
    path=tmp_path/'joint.pt'; save_checkpoint(path,m,opt,step=2,binding=binding)
    expected=step('intervention'); state=copy.deepcopy(m.state_dict()); adam=copy.deepcopy(opt.state_dict())
    assert load_checkpoint(path,m,opt,binding=binding)==2
    assert step('intervention')==expected
    equal(state,m.state_dict()); equal(adam,opt.state_dict())
    disabled=module.GeometryCooperativeModel(('A','B'),width=8,id_width=16,geometry_enabled=False).to(device)
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in disabled.parameters())
    with pytest.raises(ValueError): disabled.load_state_dict(m.state_dict())
    disabled.zero_grad(set_to_none=True)
    reference=disabled.objective('image',[task])
    actual=disabled.geometry_objective(task,**kwargs)
    torch.testing.assert_close(reference,actual,atol=0,rtol=0)
    actual.backward()
    assert all(p.grad is None for p in disabled.fusion.geometry.graph.incidence.parameters())
    with pytest.raises(ValueError):m.geometry_binding({'image':(task,)},batch_size=1,image_requests=(),native=native)
