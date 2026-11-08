import json
import pytest
import torch
from mora3.data.uniprot_annotations import AnnotationPredictionView
from mora3.model import uniprot_ec as ec


def view(packet):
    return AnnotationPredictionView('P1','MAG',json.dumps({'annotation_packet':packet}).encode(),b'{}')


def test_exact_partial_and_parent_scope_no_invented_negatives():
    packet={'proteinDescription':{'recommendedName':{'ecNumbers':[{'value':'3.6.4.-'}]},
              'contains':[{'recommendedName':{'ecNumbers':[{'value':'1.1.1.1'}]}}]},
            'comments':[{'commentType':'CATALYTIC ACTIVITY','reaction':{'ecNumber':'2.7.11.1'}},
                        {'commentType':'CATALYTIC ACTIVITY','molecule':'Isoform 2','reaction':{'ecNumber':'1.1.1.1'}}]}
    y,m=ec.ec_targets([view(packet)],('3.6.4.-','3.6.4.1','2.7.11.1','1.1.1.1'))
    assert y.tolist()==[[1.,0.,1.,0.]]
    assert m.tolist()==[[True,False,True,False]]
    assert ec.ec_assertions(view(packet))==('2.7.11.1','3.6.4.-')


def test_description_parent_molecule_excluded():
    p={'proteinDescription':{'recommendedName':{'molecule':'Chain A','ecNumbers':[{'value':'1.1.1.1'}]},
                            'alternativeNames':[{'ecNumbers':[{'value':'2.7.11.1'}]}]}}
    assert ec.ec_assertions(view(p))==('2.7.11.1',)


@pytest.mark.parametrize('axis',[(),('1.1.1.1','1.1.1.1'),('1.-.1.-',),('GO:0000001',),('8.1.1.1',)])
def test_invalid_axis_fails_closed(axis):
    with pytest.raises(ValueError):ec.ECAnnotationHead(4,axis)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_observed_reference_and_unknown_gradient(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    z=torch.tensor([[.4,-.8]],device=device,requires_grad=True)
    y=torch.tensor([[1.,float('nan')]],device=device);m=torch.tensor([[True,False]],device=device)
    loss=ec.observed_ec_loss(z,y,m)
    torch.testing.assert_close(loss,torch.nn.functional.softplus(-z[0,0]))
    loss.backward();torch.testing.assert_close(z.grad,torch.tensor([[torch.sigmoid(z[0,0]).item()-1,0.]],device=device))
    z.grad=None
    ec.observed_ec_loss(z,y,torch.zeros_like(m)).backward()
    assert torch.equal(z.grad,torch.zeros_like(z))


def test_head_axis_binding_and_missingness():
    head=ec.ECAnnotationHead(4,('1.1.1.1','2.7.11.1'))
    latent=torch.randn(1,4,requires_grad=True)
    y,m=ec.ec_targets([view({})],head.axis_ids)
    assert not m.any()
    ec.observed_ec_loss(head(latent),y,m).backward()
    assert torch.equal(latent.grad,torch.zeros_like(latent))
    clone=ec.ECAnnotationHead(4,head.axis_ids);clone.load_state_dict(head.state_dict())
    assert torch.equal(head(latent),clone(latent))
    with pytest.raises(ValueError,match='axis'):
        ec.ECAnnotationHead(4,tuple(reversed(head.axis_ids))).load_state_dict(head.state_dict())


def test_target_mutation_does_not_change_shared_inputs():
    from mora3.model.uniprot_go import annotation_graph_task
    from mora3.model.payload import PayloadEncoder
    from mora3.tokenization.shortest_path import ByteTokenizer
    from dataclasses import replace
    a=replace(view({}),provenance_bytes=b'{"split":"train","license":"CC-BY-4.0","source_ref":{"source_release":"2026_03","raw_file_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}')
    b=replace(a,target_bytes=json.dumps({'annotation_packet':{'proteinDescription':{'recommendedName':{'ecNumbers':[{'value':'1.1.1.1'}]}}}}).encode())
    encoder=PayloadEncoder(ByteTokenizer(controls=('pad','node','edge')),4)
    ra,ta=annotation_graph_task(a);rb,tb=annotation_graph_task(b)
    ga=encoder.forward_task(ra,task=ta,identifiers=torch.eye(1))
    gb=encoder.forward_task(rb,task=tb,identifiers=torch.eye(1))
    # production payload contracts include both graph features and identifiers
    assert torch.equal(ga['features'],gb['features'])
