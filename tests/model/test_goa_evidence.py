import pytest
import torch
from mora3.model import uniprot_go as go


def assertion(term='GO:0000001', qualifier='enables', evidence='IDA', subject='UniProtKB:P1'):
    return dict(goId=term, qualifier=qualifier, goEvidence=evidence, geneProductId=subject,
                reference='PMID:1', evidenceCode='ECO:0000314')


def project(rows, **kw):
    assert hasattr(go, 'goa_targets'), 'native GOA evidence projection missing'
    return go.goa_targets(rows, ('GO:0000001','GO:0000002','GO:0000003'),
                          subjects=['UniProtKB:P1']*len(rows), **kw)


def test_nd_missing_evidence_unknown_qualifier_and_wrong_form_are_not_labels():
    y,m=project([[assertion(evidence='ND'), assertion('GO:0000002', evidence=None),
                  assertion('GO:0000003', qualifier=None),
                  assertion(subject='UniProtKB:P1-2')]])
    assert not m.any() and not y.any()


def test_not_negative_and_nd_does_not_conflict_with_supported_positive():
    y,m=project([[assertion(),assertion(evidence='ND',qualifier='NOT|enables'),
                  assertion('GO:0000002',qualifier='NOT|enables')]])
    assert y.tolist()==[[1.,0.,0.]] and m.tolist()==[[True,True,False]]
    with pytest.raises(ValueError,match='conflict'):
        project([[assertion(),assertion(qualifier='NOT|enables')]])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_reference_loss_gradient_and_source_immutability(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    import copy
    rows=[[assertion(),assertion('GO:0000002',qualifier='NOT|enables')]]
    original=copy.deepcopy(rows)
    y,m=project(rows,device=device,dtype=torch.float64)
    z=torch.tensor([[.3,-.7,float('nan')]],device=device,dtype=torch.float64,requires_grad=True)
    loss=go.observed_go_loss(z,y,m)
    ref=(torch.nn.functional.softplus(-z[0,0])+torch.nn.functional.softplus(z[0,1]))/2
    torch.testing.assert_close(loss,ref)
    a=torch.autograd.grad(loss,z,retain_graph=True)[0];b=torch.autograd.grad(ref,z)[0]
    torch.testing.assert_close(a,b,rtol=1e-14,atol=1e-15)
    assert a[0,2]==0 and rows==original


def test_subject_and_batch_validation():
    assert hasattr(go,'goa_targets'), 'native GOA evidence projection missing'
    with pytest.raises(ValueError):go.goa_targets([[]],('GO:0000001',),subjects=[])


def test_kb_nd_excluded_before_conflict_resolution():
    from test_uniprot_go_objective import ref,view
    nd=ref('GO:0000001','NOT');nd['properties'][1]['value']='ND:UniProtKB'
    y,m=go.go_targets([view([ref('GO:0000001'),nd])],('GO:0000001',),exclude_nd=True)
    assert y.item()==1 and m.item()


def test_context_extensions_and_future_evidence_are_not_unconditional_facts():
    a=assertion();a['extensions']=[{'connectedXrefs':[{'relation':'occurs_in','db':'CL','id':'1'}]}]
    _,m=project([[a,assertion('GO:0000002',evidence='FUTURE')]])
    assert not m.any()


def test_unknown_relation_is_not_promoted_to_positive():
    _,m=project([[assertion(qualifier='future_relation')]])
    assert not m.any()


def test_legacy_kb_policy_retained_for_historical_checkpoint_recovery():
    from test_uniprot_go_objective import ref,view
    nd=ref('GO:0000001');nd['properties'][1]['value']='ND:UniProtKB'
    _,m=go.go_targets([view([nd])],('GO:0000001',))
    assert m.item()
