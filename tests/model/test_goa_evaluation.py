"""Evidence strata retain observation masks rather than invent negatives."""
import copy
import pytest
import torch
from test_goa_evidence import assertion


def evaluate(z, rows, axis=('GO:0000001','GO:0000002','GO:0000003')):
    from mora3.model import goa_evaluation as module
    return module.evaluate_goa_evidence(z, rows, axis, subjects=['UniProtKB:P1']*len(rows))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_observed_reference_evidence_counts_and_unknown_nan(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    rows=[[assertion(),assertion('GO:0000002',qualifier='NOT|enables',evidence='IEA'),
           assertion('GO:0000003',evidence='ND')]]
    original=copy.deepcopy(rows)
    z=torch.tensor([[.3,-.7,float('nan')]],dtype=torch.float64,device=device,requires_grad=True)
    before=z.detach().clone()
    result=evaluate(z,rows)
    assert result['policy']=='exact-subject-coarse-GOA-evidence-evaluation-v1'
    for code, index, sign in [('IDA',0,1),('IEA',1,0)]:
        metric=result['by_evidence'][code]
        assert metric['observed']==1 and metric['positive']==sign and metric['negative']==1-sign
        ref=torch.nn.functional.softplus(-z[0,index] if sign else z[0,index]).item()
        assert metric['bce']==pytest.approx(ref,abs=1e-14)
        assert metric['brier']==pytest.approx((torch.sigmoid(z[0,index]).item()-sign)**2)
    assert result['pooled']['observed']==2
    assert result['by_evidence']['EXP']['bce'] is None
    assert rows==original and z.grad is None
    torch.testing.assert_close(z.detach(),before,equal_nan=True)


def test_duplicate_assertions_count_once_per_term_and_evidence():
    a=assertion();r=evaluate(torch.zeros(1,3),[[a,a,assertion(evidence='IEA')]])
    assert r['pooled']['observed']==1
    assert r['by_evidence']['IDA']['observed']==1
    assert r['by_evidence']['IEA']['observed']==1


def test_cross_evidence_conflict_refuses_before_stratification():
    with pytest.raises(ValueError,match='conflict'):
        evaluate(torch.zeros(1,3),[[assertion(),assertion(evidence='IEA',qualifier='NOT|enables')]])


def test_context_form_and_unknown_evidence_are_not_metrics():
    a=assertion();a['extensions']=[{'unknown':'context'}]
    r=evaluate(torch.full((1,3),float('nan')),[[a,assertion(subject='UniProtKB:P1-2'),assertion(evidence='FUTURE')]])
    assert r['pooled']==dict(observed=0,positive=0,negative=0,bce=None,brier=None)


@pytest.mark.parametrize('kind',['shape','integer','observed_nan','axis'])
def test_malformed_inputs_refuse(kind):
    z=torch.zeros(1,3);axis=('GO:0000001','GO:0000002','GO:0000003')
    if kind=='shape':z=torch.zeros(2,3)
    if kind=='integer':z=z.long()
    if kind=='observed_nan':z[0,0]=float('nan')
    if kind=='axis':axis=('GO:0000001',)*3
    with pytest.raises(ValueError):evaluate(z,[[assertion()]],axis)
