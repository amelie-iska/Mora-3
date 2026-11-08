import pytest
import torch
from torch.nn import functional as F
from mora3.tokenization.shortest_path import ByteTokenizer
from mora3.data.uniprot_annotations import AnnotationPredictionView, _canonical
from mora3.model.residue_objective import align_token_residues, observed_residue_loss


def view(features):
    return AnnotationPredictionView('P12345','MAGK',_canonical({'annotation_packet':{'features':features}}),b'{}')


def feature(kind,a,b,scope=None):
    loc=dict(start=dict(value=a,modifier='EXACT'),end=dict(value=b,modifier='EXACT'))
    if scope:loc['sequence']=scope
    return dict(type=kind,location=loc)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_alignment_and_gradient_matches_explicit_expansion(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    tok=ByteTokenizer([b'MAG'])
    x=torch.tensor([[1.,2.],[3.,4.]],device=device,requires_grad=True)
    y=align_token_residues('MAGK',tok,x)
    assert torch.equal(y,torch.stack([x[0],x[0],x[0],x[1]]))
    y.sum().backward()
    assert torch.equal(x.grad,torch.tensor([[3.,3.],[1.,1.]],device=device))
    with pytest.raises(ValueError):align_token_residues('MAGK',tok,x[:1])
    with pytest.raises(ValueError):align_token_residues('MÄ',tok,x)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_positive_equation_bond_endpoints_unknown_and_scope_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    v=view([feature('Region',1,2),feature('Region',2,2),feature('Disulfide bond',1,4),feature('Region',1,4,'P12345-2')])
    z=torch.arange(12.,device=device,dtype=torch.float64).reshape(4,3).requires_grad_()
    loss=observed_residue_loss([z],[v],feature_axis=('Region','Disulfide bond','Binding site'),applicable=torch.tensor([True],device=device))
    ref=F.softplus(-torch.stack([z[0,0],z[1,0],z[0,1],z[3,1]])).mean()
    assert torch.equal(loss,ref)
    actual=torch.autograd.grad(loss,z,retain_graph=True)[0]
    expected=torch.autograd.grad(ref,z)[0]
    assert torch.equal(actual,expected)
    assert torch.count_nonzero(actual)==4
    changed=view([feature('Region',1,2)])
    assert changed.inputs()==v.inputs()


def test_empty_and_inapplicable_are_connected_zero_without_nan_targets():
    z=torch.full((4,1),float('nan'),requires_grad=True)
    for v,active in [(view([]),True),(view([feature('Region',1,4)]),False)]:
        loss=observed_residue_loss([z],[v],feature_axis=('Region',),applicable=torch.tensor([active]))
        assert loss.item()==0
        assert torch.equal(torch.autograd.grad(loss,z)[0],torch.zeros_like(z))
    with pytest.raises(ValueError):observed_residue_loss([z],[view([])],feature_axis=('Region',),applicable=torch.tensor([1]))
    with pytest.raises(ValueError):observed_residue_loss([z[:2]],[view([])],feature_axis=('Region',),applicable=torch.tensor([True]))
