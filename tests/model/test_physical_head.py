import pytest
import torch
from mora3.model.physical_head import StructuredPhysicalHead


def targets():
    return [[dict(parameter_type='mass',unit_status='unknown',molecule_scope='PRO_1',method='MALDI')],[]]


def test_reference_missingness_gradients_and_binding():
    torch.manual_seed(48)
    h=StructuredPhysicalHead(4,('mass',),('MALDI','ESI'))
    x=torch.randn(2,4,requires_grad=True)
    z=h(x,targets())
    loss=h.objective(z,targets())
    reference=(torch.nn.functional.cross_entropy(z['unit'],torch.tensor([0]))+
               torch.nn.functional.cross_entropy(z['scope'],torch.tensor([1]))+
               torch.nn.functional.cross_entropy(z['method'],torch.tensor([0])))
    torch.testing.assert_close(loss,reference)
    loss.backward()
    assert x.grad[0].abs().sum()>0
    assert x.grad[1].abs().sum()==0
    other=StructuredPhysicalHead(4,('mass',),('ESI','MALDI'))
    with pytest.raises(ValueError,match='configuration'):
        other.load_state_dict(h.state_dict())


def test_missing_method_and_empty_are_connected_zero():
    h=StructuredPhysicalHead(4,('mass',),('MALDI',))
    x=torch.randn(2,4,requires_grad=True)
    t=targets();t[0][0]['method']=None
    z=h(x,t);h.objective(z,t).backward()
    assert h.method.weight.grad.abs().sum()==0
    h.zero_grad();x.grad=None
    h.objective(h(x,[[],[]]),[[],[]]).backward()
    assert x.grad.abs().sum()==0
    with pytest.raises(ValueError,match='axis'):
        h(x,[[dict(parameter_type='invented')],[]])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_parity():
    torch.manual_seed(48)
    cpu=StructuredPhysicalHead(4,('mass',),('MALDI','ESI'))
    gpu=StructuredPhysicalHead(4,('mass',),('MALDI','ESI')).cuda()
    gpu.load_state_dict(cpu.state_dict())
    a=torch.randn(2,4,requires_grad=True);b=a.detach().cuda().requires_grad_()
    la=cpu.objective(cpu(a,targets()),targets());lb=gpu.objective(gpu(b,targets()),targets())
    la.backward();lb.backward()
    torch.testing.assert_close(la,lb.cpu())
    torch.testing.assert_close(a.grad,b.grad.cpu())


def test_annotation_values_and_scope_do_not_enter_query_features():
    import copy
    h=StructuredPhysicalHead(4,('mass',),('MALDI','ESI'))
    x=torch.randn(2,4)
    a=targets();b=copy.deepcopy(a)
    b[0][0].update(unit_status='explicit_source_unit',molecule_scope=None,method='ESI',value=999,parent={'molecule':'other'})
    for key,z in h(x,a).items():
        assert torch.equal(z,h(x,b)[key])
