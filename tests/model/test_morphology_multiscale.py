from dataclasses import replace
import io
import pytest
import torch
from test_morphology import task
from mora3.data.livecell_graph import LiveCellGraphTask
from mora3.data.livecell_raster import rasterize_instances
from mora3.model.morphology import MorphologyInstance, foreground_objective
from mora3.model.instance_affinity import affinity_objective


def loss(m,a):
    z=m(a,for_training=True); target=rasterize_instances(a.example).to(z.device)
    return foreground_objective(z[0],target)+affinity_objective(z[1:],target)


def test_odd_alignment_hidden_targets_and_every_branch_gradient():
    torch.manual_seed(43)
    a=task(); model=MorphologyInstance(8)
    z=model(a)
    assert z.shape==(3,5,6)
    b=LiveCellGraphTask(replace(a.example,polygons=()),a.direction)
    assert torch.equal(z,model(b))
    loss(model,a).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.abs().sum()>0,name
    with pytest.raises(ValueError,match='split'): model(task('test'),for_training=True)
    with pytest.raises(ValueError,match='direction'): model(LiveCellGraphTask(a.example,'mask_to_image'))


def test_snapshot_config_and_exact_optimizer_continuation():
    torch.manual_seed(43); a=task(); m=MorphologyInstance(8)
    opt=torch.optim.Adam(m.parameters(),lr=.001)
    def step(m,o):
        o.zero_grad(); loss(m,a).backward(); o.step()
    step(m,opt)
    buf=io.BytesIO(); torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict()),buf)
    step(m,opt); buf.seek(0); state=torch.load(buf,weights_only=True)
    r=MorphologyInstance(8); o=torch.optim.Adam(r.parameters(),lr=.001)
    r.load_state_dict(state['model']); o.load_state_dict(state['optimizer']); step(r,o)
    assert all(torch.equal(v,r.state_dict()[k]) for k,v in m.state_dict().items() if isinstance(v,torch.Tensor))
    with pytest.raises((ValueError,RuntimeError)): MorphologyInstance(8,normalization='image-minmax').load_state_dict(state['model'])


def test_cuda_parity():
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(43); a=task(); cpu=MorphologyInstance(8); gpu=MorphologyInstance(8).cuda()
    gpu.load_state_dict(cpu.state_dict())
    torch.testing.assert_close(cpu(a),gpu(a).cpu(),atol=3e-6,rtol=3e-5)
    loss(cpu,a).backward(); loss(gpu,a).backward()
    for p,q in zip(cpu.parameters(),gpu.parameters()):
        torch.testing.assert_close(p.grad,q.grad.cpu(),atol=3e-6,rtol=3e-4)


@pytest.mark.parametrize('shape',[(1,1),(1,9),(9,1),(17,19)])
def test_multiscale_keeps_every_pixel(shape):
    from mora3.model.morphology import MultiscalePixels
    pixels=MultiscalePixels(8)
    x=torch.randn(1,1,*shape,requires_grad=True)
    y=pixels(x)
    assert y.shape==(1,8,*shape)
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
