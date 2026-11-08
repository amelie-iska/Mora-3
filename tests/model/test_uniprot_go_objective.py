import json
import pytest
import torch
from mora3.data.uniprot_annotations import AnnotationPredictionView
from mora3.model import uniprot_go as go


def view(refs):
    return AnnotationPredictionView('P1','MAG',json.dumps({'annotation_packet':{'uniProtKBCrossReferences':refs}}).encode(),b'{}')


def ref(term, qualifier=None, **extra):
    props=[{'key':'GoTerm','value':'F:binding'},{'key':'GoEvidenceType','value':'IDA:UniProtKB'}]
    if qualifier is not None: props.append({'key':'Qualifier','value':qualifier})
    return dict(database='GO',id=term,properties=props,**extra)


def test_projection_missing_not_negative_and_scoped_exclusion():
    v=view([ref('GO:0000001'),ref('GO:0000002','NOT|enables'),ref('GO:0000003',isoformId='P1-2')])
    y,m=go.go_targets([v],('GO:0000001','GO:0000002','GO:0000003','GO:0000004'))
    assert y.tolist()==[[1.,0.,0.,0.]]
    assert m.tolist()==[[True,True,False,False]]
    assert v.targets()['annotation_packet']['uniProtKBCrossReferences'][0]['properties'][1]['value']=='IDA:UniProtKB'


def test_conflicting_source_polarity_fails_closed():
    with pytest.raises(ValueError,match='conflict'):
        go.go_targets([view([ref('GO:0000001'),ref('GO:0000001','NOT')])],('GO:0000001',))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_loss_reference_unknown_nan_zero_grad_and_empty(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    z=torch.tensor([[.4,-.8,2.]],device=device,requires_grad=True)
    y=torch.tensor([[1.,0.,float('nan')]],device=device)
    m=torch.tensor([[True,True,False]],device=device)
    loss=go.observed_go_loss(z,y,m)
    expected=(torch.nn.functional.softplus(-z[0,0])+torch.nn.functional.softplus(z[0,1]))/2
    torch.testing.assert_close(loss,expected)
    loss.backward();torch.testing.assert_close(z.grad,torch.tensor([[(torch.sigmoid(z[0,0]).item()-1)/2,torch.sigmoid(z[0,1]).item()/2,0]],device=device))
    z.grad=None
    empty=go.observed_go_loss(z,torch.full_like(z,float('nan')),torch.zeros_like(m));empty.backward()
    assert empty.item()==0 and torch.equal(z.grad,torch.zeros_like(z))


def test_head_resume_axis_binding_and_target_exclusion():
    torch.manual_seed(40)
    head=go.GOAnnotationHead(4,('GO:0000001','GO:0000002'))
    latent=torch.randn(2,4,requires_grad=True)
    before=head(latent).detach().clone()
    a=view([ref('GO:0000001')]);b=view([ref('GO:0000002')])
    go.go_targets([a,b],head.axis_ids)
    assert torch.equal(head(latent),before)
    go.observed_go_loss(head(latent),*go.go_targets([a,b],head.axis_ids)).backward()
    assert latent.grad.abs().sum()>0 and head.output.weight.grad.abs().sum()>0
    clone=go.GOAnnotationHead(4,head.axis_ids);clone.load_state_dict(head.state_dict())
    assert torch.equal(head(latent),clone(latent))
    with pytest.raises(ValueError,match='axis'):
        go.GOAnnotationHead(4,tuple(reversed(head.axis_ids))).load_state_dict(head.state_dict())


@pytest.mark.parametrize('axis',[(),('GO:0000001','GO:0000001'),('invalid',)])
def test_axis_validation(axis):
    with pytest.raises(ValueError):go.GOAnnotationHead(4,axis)


def test_mask_and_observed_value_validation():
    z=torch.zeros(1,2)
    with pytest.raises(ValueError):go.observed_go_loss(z,z,torch.ones_like(z))
    with pytest.raises(ValueError):go.observed_go_loss(z,z+2,torch.ones_like(z,dtype=torch.bool))
