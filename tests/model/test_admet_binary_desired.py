"""Desired classes are requests, never observed labels or thresholded probabilities."""
from dataclasses import replace
import pytest
import torch
from test_admet_model import setup
from mora3.data.admet import DesiredProperty, NativeCell, AssayContext

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('literal',['0','1','1.000','0e3'])
def test_binary_desired_probability_nll_and_shared_gradients(device,literal):
    model,v,c,_,_=setup(device,'binary');out=model((v,),(c,))
    r=DesiredProperty(c.definition.key,NativeCell.parse(literal),c.assay_context,b'{}')
    got=model.assess_desired_class(out,(v,),(c,),(r,),native_unit=c.unit)
    logit=out[0]['logit'];y=logit.new_tensor(float(literal))
    expected=torch.nn.functional.binary_cross_entropy_with_logits(logit,y)
    torch.testing.assert_close(got['penalty'],expected)
    torch.testing.assert_close(got['desired_class_probability'],torch.sigmoid((2*y-1)*logit).reshape(1))
    params=tuple(model.parameters())
    a=torch.autograd.grad(got['penalty'],params,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(expected,params,allow_unused=True)
    assert sum(x is not None for x in a)>10
    for x,z in zip(a,b):
        assert (x is None)==(z is None)
        if x is not None:torch.testing.assert_close(x,z)
    assert got['native_constraints']==(r.native_constraint.native_bytes,)
    assert got['observed_label_count']==0 and 'satisfied' not in got
    assert got['evidence_role']=='model_prediction_desired_consistency'

@pytest.mark.parametrize('bad',['regression','unit','endpoint','context','count','nan','inf','fraction','inequality','interval'])
def test_binary_desired_rejects_ambiguous_requests(bad):
    model,v,c,_,_=setup('cpu','regression' if bad=='regression' else 'binary')
    out=model((v,),(c,));r=DesiredProperty(c.definition.key,NativeCell.parse('1'),c.assay_context,b'{}');unit=c.unit
    if bad=='unit':unit='foreign'
    if bad=='endpoint':r=replace(r,endpoint_key='foreign')
    if bad=='context':r=replace(r,assay_context=AssayContext.from_mapping({'species':'foreign'}))
    if bad in ('nan','inf'):out=({'logit':torch.tensor(float(bad),dtype=torch.float64)},)
    if bad in ('fraction','inequality','interval'):
        cell={'fraction':lambda:NativeCell.parse('.5'),'inequality':lambda:NativeCell.parse('>=1'),
              'missing':lambda:NativeCell.parse('NA'),'interval':lambda:NativeCell.interval('range',lower='0',upper='1')}[bad]()
        r=replace(r,native_constraint=cell)
    with pytest.raises(ValueError):model.assess_desired_class(out,(v,),(c,),() if bad=='count' else (r,),native_unit=unit)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_binary_desired_extreme_logits_and_provenance_exclusion(device):
    model,v,c,_,_=setup(device,'binary')
    logits=torch.tensor([-1000.,0.,1000.],device=device,dtype=torch.float64,requires_grad=True)
    out=tuple({'logit':x} for x in logits)
    requests=tuple(DesiredProperty(c.definition.key,NativeCell.parse('1'),c.assay_context,b'{}') for _ in logits)
    a=model.assess_desired_class(out,(v,)*3,(c,)*3,requests,native_unit=c.unit)
    b=model.assess_desired_class(out,(v,)*3,(c,)*3,tuple(replace(r,request_provenance_bytes=b'{"hidden":1}') for r in requests),native_unit=c.unit)
    assert torch.isfinite(a['penalty']) and torch.isfinite(torch.autograd.grad(a['penalty'],logits)[0]).all()
    assert a.keys()==b.keys()
    for k in a:
        if isinstance(a[k],torch.Tensor):assert torch.equal(a[k],b[k])
        else:assert a[k]==b[k]
