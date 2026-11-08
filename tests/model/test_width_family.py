"""Width families must preserve semantic axes, orientation and ownership."""
from dataclasses import replace
import pytest
from torch import nn
from mora3.model.parameterization import Axis, ParameterRole, parameter_ledger


def ledger(width, identity='a'*64):
    m=nn.Linear(width,7)
    return parameter_ledger(m, {'weight':ParameterRole('output',(Axis('finite',identity),Axis('width','hidden'))),
                               'bias':ParameterRole('vector',(Axis('finite',identity),))})


def build(*args):
    from mora3.model.parameterization import width_family
    return width_family(*args)


def test_nonsquare_output_family_preserves_finite_axis_and_counts():
    a,b,c=ledger(3),ledger(5),ledger(9)
    result=build(a,b,c)
    assert result.widths == (('hidden',3,5,9),)
    assert result.total_numel == (28,42,70)
    assert result.base is a and result.target is c


@pytest.mark.parametrize('drift',['identity','finite-size','orientation','alias','role','dtype','trainable','custom','same-delta','shrunk-target','inconsistent-width'])
def test_semantic_or_ownership_drift_refuses(drift):
    a,b,c=ledger(3),ledger(5),ledger(9)
    rows=list(c.parameters); row=next(x for x in rows if len(x.shape)==2);i=rows.index(row)
    if drift=='identity':c=ledger(9,'b'*64)
    if drift=='finite-size':rows[i]=replace(row,shape=(8,9),numel=72)
    if drift=='orientation':rows[i]=replace(row,axes=row.axes[::-1])
    if drift=='alias':rows[i]=replace(row,aliases=('another.weight',))
    if drift=='role':rows[i]=replace(row,role='input')
    if drift=='dtype':rows[i]=replace(row,dtype='torch.float64')
    if drift=='trainable':rows[i]=replace(row,trainable=False)
    if drift=='custom':rows[i]=replace(row,role='custom')
    if drift=='same-delta':b=ledger(3)
    if drift=='shrunk-target':c=ledger(4)
    if drift=='inconsistent-width':
        rows.append(replace(row,aliases=('other',),shape=(7,8),numel=56))
        a=replace(a,parameters=a.parameters+(replace(row,aliases=('other',),shape=(7,3),numel=21),))
        b=replace(b,parameters=b.parameters+(replace(row,aliases=('other',),shape=(7,5),numel=35),))
    if drift not in ('identity','same-delta','shrunk-target'):c=replace(c,parameters=tuple(rows))
    with pytest.raises(ValueError):build(a,b,c)


def test_two_hidden_axes_and_tied_aliases():
    def make(w):
        m=nn.Module();m.a=nn.Linear(w,2*w,bias=False);m.b=m.a
        role=ParameterRole('hidden',(Axis('width','wide'),Axis('width','narrow')))
        return parameter_ledger(m,{'a.weight':role,'b.weight':role})
    result=build(make(2),make(3),make(5))
    assert result.widths == (('narrow',2,3,5),('wide',4,6,10))
    assert result.total_numel == (8,18,50)
