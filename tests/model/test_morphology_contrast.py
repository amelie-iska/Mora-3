from dataclasses import replace
import numpy as np
import pytest
import torch
from mora3.model.morphology import MorphologyForeground
from test_morphology import task

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_input_minmax_affine_invariance_constant_stability_target_exclusion(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    torch.manual_seed(31)
    model=MorphologyForeground(width=8,normalization='image-minmax').to(device)
    a=task(); pixels=np.arange(30,dtype=np.uint8).reshape(5,6)
    b=replace(a,example=replace(a.example,pixels_hex=(pixels*3+12).tobytes().hex(),polygons=()))
    torch.testing.assert_close(model(a),model(b),atol=2e-6,rtol=2e-5)
    c=replace(a,example=replace(a.example,pixels_hex='99'*30))
    d=replace(a,example=replace(a.example,pixels_hex='00'*30))
    assert torch.isfinite(model(c)).all()
    assert torch.equal(model(c),model(d))
    model(c).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_default_unchanged_and_invalid_mode_rejected():
    torch.manual_seed(31); a=MorphologyForeground(width=8)
    torch.manual_seed(31); b=MorphologyForeground(width=8,normalization='unsigned-dtype-max')
    assert torch.equal(a(task()),b(task()))
    with pytest.raises(ValueError,match='normalization'): MorphologyForeground(normalization='unknown')


def test_minmax_cpu_cuda_forward_backward_parity():
    if not torch.cuda.is_available(): pytest.skip('no CUDA')
    torch.manual_seed(31)
    cpu=MorphologyForeground(width=8,normalization='image-minmax')
    gpu=MorphologyForeground(width=8,normalization='image-minmax').cuda()
    gpu.load_state_dict(cpu.state_dict())
    x=cpu(task()); y=gpu(task())
    torch.testing.assert_close(x,y.cpu(),atol=2e-6,rtol=2e-5)
    x.square().mean().backward(); y.square().mean().backward()
    for a,b in zip(cpu.parameters(),gpu.parameters()):
        torch.testing.assert_close(a.grad,b.grad.cpu(),atol=2e-6,rtol=2e-4)
