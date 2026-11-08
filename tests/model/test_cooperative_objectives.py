from dataclasses import replace
import pytest
import torch
from mora3.data.schema import GraphRecord, Node, Measurement
from mora3.data.provenance import Provenance
from mora3.data.tasks import ConditionalTask
from mora3.model.cooperative_objectives import scalar_objectives

A=('node',0,'response'); B=('node',0,'missing'); C=('node',0,'context')
def fixture():
    g=GraphRecord('test',(Node('cell',(Measurement('response',2.,'count','observed',True),Measurement('missing',None,None,'missing',False),Measurement('context',24.,'hour','observed',True))),),(),(Provenance('fixture','v1','test','0'*64,'CC0-1.0'),))
    t=ConditionalTask(((A,'target'),(B,'target'),(C,'condition')),context=(('time',C),),topology_known=True)
    return g,t

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_equation_subset_and_gradient_ownership(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    g,t=fixture(); shared=torch.tensor(3.,device=device,requires_grad=True)
    excluded=torch.tensor(7.,device=device,requires_grad=True)
    out=scalar_objectives(g,t,{A:(2*shared,'count'),B:(excluded,None)},selected=(A,B),binding=(g.canonical_bytes(),t.canonical_bytes()))
    assert out['missing']==(B,) and set(out['terms'])=={A}
    assert out['total'].item()==16
    out['total'].backward()
    assert shared.grad.item()==16 and excluded.grad is None
    assert scalar_objectives(g,t,{},selected=(B,),binding=(g.canonical_bytes(),t.canonical_bytes()))['total'] is None

@pytest.mark.parametrize('fault',['unit','shape','nan','condition','binding','predicted','duplicate'])
def test_fail_closed(fault):
    g,t=fixture(); pred={A:(torch.tensor(3.,requires_grad=True),'count')}; selected=(A,)
    binding=(g.canonical_bytes(),t.canonical_bytes())
    if fault=='unit':pred[A]=(torch.tensor(3.),'um')
    if fault=='shape':pred[A]=(torch.ones(2),'count')
    if fault=='nan':pred[A]=(torch.tensor(float('nan')),'count')
    if fault=='condition': selected=(C,)
    if fault=='binding':binding=(b'other',binding[1])
    if fault=='predicted':g=replace(g,nodes=(replace(g.nodes[0],measurements=(replace(g.nodes[0].measurements[0],status='predicted'),)+g.nodes[0].measurements[1:]),));binding=(g.canonical_bytes(),t.canonical_bytes())
    if fault=='duplicate':selected=(A,A)
    with pytest.raises(ValueError):scalar_objectives(g,t,pred,selected=selected,binding=binding)


def test_hidden_unselected_values_and_predictions_are_not_inspected():
    g,t=fixture()
    assert scalar_objectives(g,t,{A:object()},selected=(B,),binding=(g.canonical_bytes(),t.canonical_bytes()))['total'] is None


def test_mean_uses_only_available_selected_targets():
    g,t=fixture()
    g=replace(g,nodes=(replace(g.nodes[0],measurements=(g.nodes[0].measurements[0],Measurement('missing',4.,'count','observed',True),g.nodes[0].measurements[2])),))
    a=torch.tensor(3.,requires_grad=True);b=torch.tensor(6.,requires_grad=True)
    result=scalar_objectives(g,t,{A:(a,'count'),B:(b,'count')},selected=(A,B),binding=(g.canonical_bytes(),t.canonical_bytes()))
    assert result['total'].item()==2.5
    result['total'].backward()
    assert a.grad.item()==1 and b.grad.item()==2
