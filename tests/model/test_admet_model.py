"""Synthetic head/equation tests; source candidate readback is not supervision."""
import copy
import importlib
import pytest
import torch
from test_flow_unified import build
from pathlib import Path
_spec=importlib.util.spec_from_file_location("admet_head_fixtures",Path(__file__).parents[1]/"data/test_admet.py")
_fixture=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_fixture)
record,registry,canonical=_fixture.record,_fixture.registry,_fixture.canonical
from mora3.data import admet as d
from mora3.model.admet_objectives import AdmetObjectiveContract


def setup(device, kind='regression'):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(15301)
    raw=record(d,literal='1.25');reg=registry(d,raw,kind=kind)
    contract=AdmetObjectiveContract(reg.definitions[0],d.AssayContext.from_mapping(raw['known_context']),'peptide')
    cls=importlib.import_module('mora3.model.admet_model').SharedAdmetModel
    model=cls(build(device).model.annotation,(contract,)).to(device=device,dtype=torch.float64).eval()
    view=d.admet_prediction_view(d.AdmetObservation.from_native(canonical(raw),applicability='applicable'),reg)
    return model,view,contract,raw,reg


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('kind',['regression','binary'])
def test_shared_head_affine_scale_and_all_gradient_reference(device,kind):
    model,v,c,_,_=setup(device,kind)
    z=model.condition((v.inputs(),))
    got=model((v,),(c,))[0]
    h=model.heads[c.partition_key]
    ref=torch.nn.functional.linear(z,h.weight,h.bias)[0]
    expected={'logit':ref[0]} if kind=='binary' else {'mean':ref[0],'scale':torch.nn.functional.softplus(ref[1])+torch.finfo(ref.dtype).eps}
    assert set(got)==set(expected)
    for k in got:torch.testing.assert_close(got[k],expected[k],rtol=0,atol=0)
    loss=sum(x.square() for x in got.values());reference=sum(x.square() for x in expected.values())
    params=tuple(model.parameters())
    a=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(reference,params,allow_unused=True)
    assert sum(x is not None for x in a)>10
    for x,y in zip(a,b):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)
    assert model.shared.payload.embedding.weight in set(params)
    if kind=='regression':assert got['scale']>0


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_hidden_targets_never_encoded_desired_is_separate_and_admission_closed(device):
    model,v,c,raw,reg=setup(device)
    x=model((v,),(c,))
    changed=copy.deepcopy(raw);changed['cell']=d.NativeCell.parse('>900').as_mapping();changed['native_source_record']['native_half_life']='>900'
    changed['source_ref']={'source_id':'HIDDEN'};changed['source_split']={'native_export':'train'}
    w=d.admet_prediction_view(d.AdmetObservation.from_native(canonical(changed),applicability='applicable'),reg)
    for k in x[0]:assert torch.equal(x[0][k],model((w,),(c,))[0][k])
    with pytest.raises(ValueError,match='admission'):model.observed_loss(x[0],v,c)
    desired=d.DesiredProperty(c.definition.key,d.NativeCell.parse('<=2'),c.assay_context,b'{"request":"a"}')
    inputs=d.desired_admet_inputs('peptide',(desired,),reg)
    z=model.condition((inputs,));assert z.shape==(1,4) and torch.isfinite(z).all()
    other=d.DesiredProperty(c.definition.key,d.NativeCell.parse('<=3'),c.assay_context,b'{"request":"b"}')
    assert not torch.equal(z,model.condition((d.desired_admet_inputs('peptide',(other,),reg),)))
    same=d.DesiredProperty(c.definition.key,desired.native_constraint,c.assay_context,b'{"request":"changed"}')
    assert torch.equal(z,model.condition((d.desired_admet_inputs('peptide',(same,),reg),)))
    with pytest.raises(ValueError,match='decoder'):model.generate((inputs,))


@pytest.mark.parametrize('bad',['context','class','definition','unknown','empty','duplicate'])
def test_partition_configuration_fails_closed(bad):
    model,v,c,raw,reg=setup('cpu')
    cls=type(model)
    if bad=='context':
        other=AdmetObjectiveContract(c.definition,d.AssayContext.from_mapping({'species':'other'}),'peptide')
        with pytest.raises(ValueError,match='partition'):model((v,),(other,))
    elif bad=='class':
        other=AdmetObjectiveContract(c.definition,c.assay_context,'protein')
        with pytest.raises(ValueError):cls(model.shared,(other,))
    elif bad=='definition':
        s=copy.deepcopy(model.get_extra_state());s['partitions']=()
        with pytest.raises(ValueError):model.set_extra_state(s)
    elif bad=='unknown':
        r=registry(d,raw,kind='unresolved');other=AdmetObjectiveContract(r.definitions[0],c.assay_context,'peptide')
        with pytest.raises(ValueError):cls(model.shared,(other,))
    elif bad=='empty':
        with pytest.raises(ValueError):model((),())
    else:
        with pytest.raises(ValueError):cls(model.shared,(c,c))
