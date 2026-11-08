import pytest
import torch

from mora3.model.source_byte_bounded_adjoint import source_byte_bounded_adjoint
from mora3.model.source_byte_objective import source_byte_objective
from test_pubchem_admet_mup import pubchem


@pytest.mark.parametrize('chunk_size',[1,8,64])
def test_equal_complete_loss_and_all_first_order_gradients(chunk_size):
    decoder=pubchem(16).double().eval()
    ctx=torch.arange(32,dtype=torch.float64).reshape(2,16).mul(.01).requires_grad_()
    targets=('Téxture α'.encode(),b'')
    args=dict(chunk_size=chunk_size,max_batch_size=2,max_total_target_bytes=64)
    params=tuple(p for layer in (decoder.embedding,decoder.rnn,decoder.output)for p in layer.parameters())
    current=source_byte_objective(decoder,ctx,targets,recompute=False,**args)
    bounded=source_byte_bounded_adjoint(decoder,ctx,targets,**args)
    assert torch.equal(current,bounded)
    a=torch.autograd.grad(current,(ctx,*params))
    b=torch.autograd.grad(bounded,(ctx,*params))
    assert all(torch.allclose(x,y,rtol=1e-12,atol=1e-12) for x,y in zip(a,b,strict=True))


def test_refuses_mixed_precision_and_parameter_change_after_forward():
    decoder=pubchem(16).float().eval()
    ctx=torch.zeros(1,16,dtype=torch.float32,requires_grad=True)
    args=dict(chunk_size=2,max_batch_size=1,max_total_target_bytes=32)
    with torch.autocast('cpu'),pytest.raises(ValueError,match='without autocast'):
        source_byte_bounded_adjoint(decoder,ctx,(b'abc',),**args)
    score=source_byte_bounded_adjoint(decoder,ctx,(b'abc',),**args)
    with torch.no_grad():decoder.output.weight.add_(1)
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):
        score.backward()
