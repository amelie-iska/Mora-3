"""Synthetic context/task guards; no biological coverage claimed."""
import importlib
import sys
from pathlib import Path
from dataclasses import replace
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from mora3.data.schema import GraphRecord, Node, Measurement, Provenance
from mora3.model.payload import PayloadEncoder
from mora3.tokenization.shortest_path import ByteTokenizer


def api():
    assert (Path(__file__).resolve().parents[2] / 'src/mora3/data/tasks.py').exists(), 'task contracts missing'
    return importlib.import_module('mora3.data.tasks')


def record(response=2., dose=1.):
    return GraphRecord('fixture', (Node('assay', tuple(Measurement(n,v,u,'observed',True) for n,v,u in
        [('molecule','A',None),('response',response,'count'),('dose',dose,'nM'),
         ('desired',3.,'count'),('audit','secret',None)])),), (),
        (Provenance('test','v1','a','0'*64,'test-only'),))


def task(inverse=False):
    t = api()
    roles = ['target','condition','condition','desired','evaluation'] if inverse else ['condition','target','condition','desired','evaluation']
    return t.ConditionalTask(tuple((('node',0,n),role) for n,role in zip(
        ['molecule','response','dose','desired','audit'],roles)),
        context=(('dose',('node',0,'dose')),), topology_known=True)


def test_two_masks_and_supervision():
    a,b = task(),task(True)
    assert a.input_fields(record()) == frozenset({('node',0,n) for n in ('molecule','dose','desired')})
    assert b.input_fields(record()) == frozenset({('node',0,n) for n in ('response','dose','desired')})
    assert a.available_targets(record()) == (('node',0,'response'),)
    assert api().ConditionalTask.from_bytes(a.canonical_bytes()) == a


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_hidden_values_metadata_and_gradients_do_not_affect_features(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    torch.manual_seed(19)
    m = PayloadEncoder(ByteTokenizer(controls=('pad','node','edge')),8).to(device)
    r = record()
    changed = replace(record(response=12345.),record_id='other')
    ms = list(changed.nodes[0].measurements)
    ms[1] = Measurement('response',None,'different','missing',False)
    ms[-1] = Measurement('audit','longer hidden audit',None,'predicted',True)
    changed = replace(changed,nodes=(Node('assay',tuple(ms)),))
    outputs=[]; grads=[]
    for row in (r,changed):
        y=m.forward_task(row,task=task(),identifiers=torch.eye(1,device=device))['features']
        outputs.append(y); grads.append(torch.autograd.grad(y.sum(),tuple(m.parameters())))
    torch.testing.assert_close(*outputs,atol=0,rtol=0)
    for a,b in zip(*grads): torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert task().available_targets(changed) == ()


@pytest.mark.parametrize('case',['overlap','unknown','unclassified','missing_condition','hidden_context','unknown_topology','predicted_target','bad_role'])
def test_fail_closed(case):
    t=task(); r=record()
    with pytest.raises(ValueError):
        if case=='overlap': t=replace(t,roles=t.roles+((('node',0,'response'),'condition'),))
        if case=='unknown': t=replace(t,roles=t.roles+((('node',0,'absent'),'target'),))
        if case=='unclassified': t=replace(t,roles=t.roles[:-1])
        if case=='missing_condition':
            ms=list(r.nodes[0].measurements); ms[0]=Measurement('molecule',None,None,'missing',False)
            r=replace(r,nodes=(Node('assay',tuple(ms)),))
        if case=='hidden_context': t=replace(t,context=(('dose',('node',0,'response')),))
        if case=='unknown_topology': t=replace(t,topology_known=False)
        if case=='predicted_target':
            ms=list(r.nodes[0].measurements); ms[1]=replace(ms[1],status='predicted')
            r=replace(r,nodes=(Node('assay',tuple(ms)),))
        if case=='bad_role': t=replace(t,roles=t.roles+((('node',0,'x'),'oops'),))
        t.input_fields(r); t.available_targets(r)


def test_context_matching_is_exact_and_missing_is_not_a_join():
    t=task()
    t.require_same_context(record(),record())
    with pytest.raises(ValueError): t.require_same_context(record(),record(dose=2.))
    with pytest.raises(ValueError): replace(t,context=()).require_same_context(record(),record())
    # Unit conversion is never guessed.
    r=record(); ms=list(r.nodes[0].measurements); ms[2]=replace(ms[2],units='uM')
    with pytest.raises(ValueError): t.require_same_context(r,replace(r,nodes=(Node('assay',tuple(ms)),)))
