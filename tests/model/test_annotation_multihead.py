from dataclasses import replace
import copy
import pytest
import torch
from test_annotation_vocabulary import admitted
from mora3.model.annotation_vocabulary import fit_annotation_vocabulary
from mora3.model.annotation_multihead import SharedAnnotationModel


def model():
    return SharedAnnotationModel(fit_annotation_vocabulary([admitted()],learned_budget=2,max_length=2),
                                 ('GO:0000001',),('1.1.1.1',),width=4,heads=1,layers=1)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_target_exclusion_and_inactive_gradient(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(4401)
    m=model().to(device)
    a=admitted();b=replace(a,target_bytes=b'{"annotation_packet":{"proteinDescription":{"recommendedName":{"ecNumbers":[{"value":"1.1.1.1"}]}},"uniProtKBCrossReferences":[{"database":"GO","id":"GO:0000001"}]}}')
    za=m([a]);zb=m([b])
    for k in za:assert torch.equal(za[k],zb[k])
    active={'GO':torch.tensor([True],device=device),'EC':torch.tensor([False],device=device)}
    losses=m.objective(zb,[b],applicable=active)
    torch.testing.assert_close(losses['GO'],torch.nn.functional.softplus(-zb['GO']).mean())
    assert losses['EC'].item()==0
    losses['total'].backward()
    assert torch.equal(m.ec.output.weight.grad,torch.zeros_like(m.ec.output.weight))
    assert m.go.output.weight.grad.abs().sum()>0
    clone=model().to(device);clone.load_state_dict(copy.deepcopy(m.state_dict()))
    for k in zb:assert torch.equal(clone([b])[k],zb[k])


def test_missing_and_invalid_applicability():
    m=model();v=[admitted()];z=m(v)
    active={k:torch.tensor([True]) for k in z}
    loss=m.objective(z,v,applicable=active)
    assert loss['total'].item()==0
    loss['total'].backward()
    for bad in [{}, {'GO':torch.tensor([1]),'EC':torch.tensor([True])},
                {'GO':torch.tensor([True,False]),'EC':torch.tensor([True])}]:
        with pytest.raises(ValueError,match='applicability'):m.objective(z,v,applicable=bad)


def test_configuration_bound_checkpoint():
    m=model();state=copy.deepcopy(m.state_dict());state['_extra_state']['layers']=2
    with pytest.raises(ValueError,match='configuration'):m.load_state_dict(state)
