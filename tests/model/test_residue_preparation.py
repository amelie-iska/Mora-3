from dataclasses import replace
import pytest
import torch
from test_residue_objective import view, feature
from mora3.tokenization.shortest_path import ByteTokenizer
from mora3.model import residue_objective as reference

@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_prepared_reference_equations_and_gradients(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    from mora3.model import residue_preparation as module
    tok=ByteTokenizer([b'MAG'])
    views=(view([feature('Region',1,2),feature('Region',2,2),feature('Disulfide bond',1,4),feature('Region',1,4,'P12345-2')]),view([]))
    axis=('Region','Disulfide bond')
    p=module.prepare_residues(views,tok,feature_axis=axis,applicable=(True,False),max_residues=8)
    p.validate(views,tok,feature_axis=axis,applicable=(True,False))
    x=torch.randn(2,3,device=device,requires_grad=True)
    a=p.align(0,x); b=reference.align_token_residues('MAGK',tok,x)
    assert torch.equal(a,b)
    assert torch.equal(torch.autograd.grad(a.sum(),x)[0],torch.autograd.grad(b.sum(),x)[0])
    z=[torch.randn(4,2,device=device,requires_grad=True) for _ in views]
    a=p.loss(z); b=reference.observed_residue_loss(z,views,feature_axis=axis,applicable=torch.tensor([True,False],device=device))
    assert torch.equal(a,b)
    assert all(torch.equal(x,y) for x,y in zip(torch.autograd.grad(a,z),torch.autograd.grad(b,z)))
    for changed in [dict(views=(replace(views[0],sequence='MAAK'),views[1])),dict(views=(view([]),views[1])),dict(feature_axis=axis[::-1]),dict(applicable=(False,False)),dict(tokenizer=ByteTokenizer([]))]:
        args=dict(views=views,tokenizer=tok,feature_axis=axis,applicable=(True,False));args.update(changed)
        with pytest.raises(ValueError,match='binding'):p.validate(**args)
    with pytest.raises(ValueError,match='budget'):module.prepare_residues(views,tok,feature_axis=axis,applicable=(True,False),max_residues=7)
    with pytest.raises(ValueError):p.align(0,x[:1])
    with pytest.raises(ValueError):p.loss([z[0][:2],z[1]])
