from dataclasses import replace
import numpy as np
import pytest
import torch
from mora3.data.livecell import LiveCellExample
from mora3.data.livecell_graph import LiveCellGraphTask
from mora3.model.morphology import MorphologyForeground, foreground_objective


def task(split='train'):
    x=LiveCellExample.from_coco(image=np.arange(30,dtype=np.uint8).reshape(5,6),
        image_info=dict(id=1,file_name='A172_Phase_C7_1_00d00h00m_1.tif',height=5,width=6),
        annotations=[dict(id=1,image_id=1,segmentation=[[0,0,3,0,3,3,0,3]])],
        split=split,source_sha256='a'*64)
    return LiveCellGraphTask(x,'image_to_mask')


def test_union_bce_equation_overlap_and_empty():
    z=torch.tensor([[0.,1.],[-2.,3.]],requires_grad=True)
    m=torch.tensor([[[True,False],[False,False]],[[True,False],[False,True]]])
    y=m.any(0).float()
    expected=(torch.logaddexp(torch.zeros_like(z),z)-y*z).mean()
    torch.testing.assert_close(foreground_objective(z,m),expected)
    foreground_objective(z,m).backward()
    torch.testing.assert_close(z.grad,(z.detach().sigmoid()-y)/4)
    torch.testing.assert_close(foreground_objective(z,m[:0]),torch.nn.functional.softplus(z).mean())
    with pytest.raises(ValueError): foreground_objective(z,m[:,:,:1])


def test_full_resolution_target_exclusion_and_shared_backbone_gradients():
    torch.manual_seed(29)
    model=MorphologyForeground(width=8)
    a=task(); b=LiveCellGraphTask(replace(a.example,polygons=()),a.direction)
    z=model(a)
    assert z.shape==(5,6)
    assert torch.equal(z,model(b))
    z.square().mean().backward()
    for module in (model.pixels,model.backbone,model.head):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(p.grad.abs().sum() for p in module.parameters())>0
    c=LiveCellGraphTask(replace(a.example,pixels_hex='ff'*30),a.direction)
    assert not torch.equal(z,model(c))


def test_training_split_and_direction_fail_closed():
    model=MorphologyForeground(width=8)
    with pytest.raises(ValueError,match='split'): model(task('test'),for_training=True)
    a=task()
    with pytest.raises(ValueError,match='direction'): model(LiveCellGraphTask(a.example,'mask_to_image'))


def test_cpu_cuda_prediction_and_gradient_parity():
    if not torch.cuda.is_available(): pytest.skip('no CUDA')
    torch.manual_seed(29)
    cpu=MorphologyForeground(width=8); gpu=MorphologyForeground(width=8).cuda()
    gpu.load_state_dict(cpu.state_dict())
    a=task(); x=cpu(a); y=gpu(a)
    torch.testing.assert_close(x,y.cpu(),atol=2e-6,rtol=2e-5)
    x.square().mean().backward(); y.square().mean().backward()
    for p,q in zip(cpu.parameters(),gpu.parameters()):
        torch.testing.assert_close(p.grad,q.grad.cpu(),atol=2e-6,rtol=2e-4)


def test_model_optimizer_checkpoint_exact_continuation(tmp_path):
    from mora3.data.livecell_raster import rasterize_instances
    torch.manual_seed(29)
    model=MorphologyForeground(width=8); opt=torch.optim.Adam(model.parameters(),lr=.003)
    a=task(); target=rasterize_instances(a.example)
    def step(m,o):
        o.zero_grad(); loss=foreground_objective(m(a,for_training=True),target)
        loss.backward(); o.step()
    step(model,opt)
    path=tmp_path/'state.pt'
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),task=a.canonical_bytes()),path)
    step(model,opt)
    saved=torch.load(path,weights_only=True)
    restored=MorphologyForeground(width=8); other=torch.optim.Adam(restored.parameters(),lr=.003)
    restored.load_state_dict(saved['model']); other.load_state_dict(saved['optimizer'])
    a.assert_resume_matches(LiveCellGraphTask.from_bytes(saved['task']))
    step(restored,other)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
