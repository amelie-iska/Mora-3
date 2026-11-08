"""Desired point-coordinate assessments are not observed labels or calibration."""
from dataclasses import replace
import pytest
import torch
from test_admet_model import setup
from mora3.data.admet import DesiredProperty, NativeCell, AssayContext


def request(c,cell,provenance=b'{}'):
    return DesiredProperty(c.definition.key,cell,c.assay_context,provenance)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('literal,bound,side',[('=1.25',1.25,'equal'),('<1.25',1.25,'left'),
    ('<=1.25',1.25,'left'),('>1.25',1.25,'right'),('>=1.25',1.25,'right')])
def test_typed_constraints_equations_and_all_shared_gradients(device,literal,bound,side):
    model,v,c,_,_=setup(device);outputs=model((v,),(c,));p=outputs[0]['mean']
    cell=NativeCell.parse(literal)
    result=model.assess_desired(outputs,(v,),(c,),(request(c,cell),),native_unit=c.unit,
                                decimal_policy='nearest-model-dtype')
    diff=p-bound
    expected=diff.square() if side=='equal' else torch.relu(diff if side=='left' else -diff).square()
    torch.testing.assert_close(result['penalty'],expected,rtol=1e-12,atol=1e-12)
    params=tuple(model.parameters())
    a=torch.autograd.grad(result['penalty'],params,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(expected,params,allow_unused=True)
    assert sum(x is not None for x in a)>10
    for x,y in zip(a,b):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)
    assert result['evidence_role']=='model_prediction_desired_consistency'
    assert result['observed_label_count']==0
    assert result['native_constraints']==(cell.native_bytes,)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('inclusive',[True,False])
def test_interval_strict_boundaries_and_provenance_exclusion(device,inclusive):
    model,v,c,_,_=setup(device)
    p=torch.tensor(1.25,device=device,dtype=torch.float64,requires_grad=True)
    outputs=({'mean':p,'scale':p.new_tensor(1)},)
    cell=NativeCell.interval('source interval',lower='1.25',upper='2.5',lower_inclusive=inclusive)
    kwargs=dict(native_unit=c.unit,decimal_policy='nearest-model-dtype')
    a=model.assess_desired(outputs,(v,),(c,),(request(c,cell,b'{"hidden":"a"}'),),**kwargs)
    b=model.assess_desired(outputs,(v,),(c,),(request(c,cell,b'{"hidden":"b"}'),),**kwargs)
    assert a['satisfied'].tolist()==[inclusive]
    assert a['penalty']==0 and torch.autograd.grad(a['penalty'],p)[0]==0
    assert a.keys()==b.keys()
    for k in a:
        if isinstance(a[k],torch.Tensor):assert torch.equal(a[k],b[k])
        else:assert a[k]==b[k]
    strict=request(c,NativeCell.parse('<1.25'))
    r=model.assess_desired(outputs,(v,),(c,),(strict,),**kwargs)
    assert not r['satisfied'].item() and r['penalty']==0


@pytest.mark.parametrize('bad',['endpoint','context','unit','policy','count','binary','overflow','underflow','collapsed','nonfinite','scale'])
def test_typed_assessment_rejects_ambiguous_coordinates(bad):
    model,v,c,_,_=setup('cpu','binary' if bad=='binary' else 'regression')
    outputs=model((v,),(c,));r=request(c,NativeCell.parse('1.25'));unit=c.unit;policy='nearest-model-dtype'
    if bad=='endpoint':r=replace(r,endpoint_key='foreign')
    elif bad=='context':r=replace(r,assay_context=AssayContext.from_mapping({'species':'foreign'}))
    elif bad=='unit':unit='foreign'
    elif bad=='policy':policy='automatic'
    elif bad=='overflow':r=replace(r,native_constraint=NativeCell.parse('1e9999'))
    elif bad=='underflow':r=replace(r,native_constraint=NativeCell.parse('1e-9999'))
    elif bad=='collapsed':r=replace(r,native_constraint=NativeCell.interval('narrow',lower='1',upper='1.00000000000000000001'))
    elif bad=='nonfinite':outputs=({'mean':torch.tensor(float('nan'),dtype=torch.float64),'scale':torch.tensor(1.,dtype=torch.float64)},)
    elif bad=='scale':outputs=({'mean':torch.tensor(1.,dtype=torch.float64),'scale':torch.tensor(-1.,dtype=torch.float64)},)
    requests=() if bad=='count' else (r,)
    with pytest.raises(ValueError):model.assess_desired(outputs,(v,),(c,),requests,native_unit=unit,decimal_policy=policy)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_declared_rounding_and_batched_no_cross_partition_pooling(device,dtype):
    model,v,c,_,_=setup(device);model=model.to(dtype=dtype)
    ps=[torch.tensor(x,device=device,dtype=dtype,requires_grad=True) for x in [.2,.4]]
    outputs=tuple({'mean':p,'scale':p.new_tensor(1)} for p in ps)
    requests=tuple(request(c,NativeCell.parse(x)) for x in ['<=0.1','>=0.3'])
    r=model.assess_desired(outputs,(v,v),(c,c),requests,native_unit=c.unit,decimal_policy='nearest-model-dtype')
    expected=(torch.relu(ps[0]-ps[0].new_tensor(.1)).square()+torch.relu(ps[1].new_tensor(.3)-ps[1]).square())/2
    torch.testing.assert_close(r['penalty'],expected)
    assert r['satisfied'].tolist()==[False,True]
    assert r['decimal_policy']=='nearest-model-dtype' and r['dtype']==str(dtype)
