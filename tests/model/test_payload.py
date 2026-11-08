"""Catch bag-of-tokens collapse, target/audit leakage and silent truncation."""
import copy
import importlib
import math
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import pytest
import torch
from mora3.data.schema import GraphRecord, Node, Edge, Measurement, Provenance
from mora3.tokenization.shortest_path import ByteTokenizer
from mora3.model.tokengt import TokenGT


def fixture(sequence='AC', target='secret'):
    return GraphRecord('fixture', (Node('rna', (
        Measurement('sequence', sequence, None, 'observed', True),
        Measurement('target', target, None, 'observed', True))), Node('annotation')),
        (Edge(0, 1, 'annotated', True),),
        (Provenance('fixture', 'v1', 'a', '0'*64, 'test-only'),))


SELECT = frozenset({('node', 0, 'sequence')})


def encoder(**kwargs):
    torch.manual_seed(19)
    module = importlib.import_module('mora3.model.payload')
    return module.PayloadEncoder(ByteTokenizer(controls=('pad','node','edge')), 8, **kwargs).double()


def test_order_sensitive_features_and_gradients():
    m = encoder()
    a = m(fixture('AC'), input_fields=SELECT, identifiers=torch.eye(2).double())
    b = m(fixture('CA'), input_fields=SELECT, identifiers=torch.eye(2).double())
    assert not torch.allclose(a['features'][0], b['features'][0])
    assert a['features'].shape == (3, 8)
    assert a['endpoints'].tolist() == [[0,0],[1,1],[0,1]]
    assert a['roles'].tolist() == [0,0,1]
    a['features'].sum().backward()
    assert m.embedding.weight.grad[65].abs().sum() > 0
    assert m.embedding.weight.grad[67].abs().sum() > 0


def test_targets_and_audit_never_reach_graph_token_or_gradients():
    m = encoder()
    backbone = TokenGT(8, 2, width=8, heads=2, layers=2).double()
    first = fixture()
    altered = replace(fixture(target='much longer target ZZZ'), record_id='other',
                      provenance=(Provenance('other','v9','b','1'*64,'test-only'),))
    outputs = []
    gradients = []
    for record in (first, altered):
        g = m(record, input_fields=SELECT, identifiers=torch.eye(2).double())
        y, _ = backbone([g])
        outputs.append(y)
        gradients.append(torch.autograd.grad(y[...,0].sum(), tuple(m.parameters())))
    torch.testing.assert_close(*outputs, atol=0, rtol=0)
    for a,b in zip(*gradients): torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert gradients[0][0][90].count_nonzero() == 0  # excluded Z payload


def test_absent_or_missing_unselected_target_has_no_effect():
    m = encoder()
    r = fixture()
    stripped = replace(r, nodes=(replace(r.nodes[0], measurements=r.nodes[0].measurements[:1]), r.nodes[1]))
    missing = replace(r, nodes=(replace(r.nodes[0], measurements=(r.nodes[0].measurements[0],
        Measurement('target',None,None,'missing',False))),r.nodes[1]))
    values = [m(v,input_fields=SELECT,identifiers=torch.eye(2).double())['features'] for v in (r,stripped,missing)]
    for v in values[1:]: torch.testing.assert_close(v,values[0],atol=0,rtol=0)


def test_typed_units_status_and_empty_string_distinct():
    m = encoder()
    values = []
    for field in (Measurement('sequence', '1', None, 'observed', True),
                  Measurement('sequence', 1, 'count', 'observed', True),
                  Measurement('sequence', 1, 'count', 'predicted', True),
                  Measurement('sequence', 1, 'seconds', 'observed', True),
                  Measurement('sequence', '', None, 'observed', True)):
        r = fixture()
        r = replace(r,nodes=(replace(r.nodes[0],measurements=(field,)),r.nodes[1]))
        values.append(m(r,input_fields=SELECT,identifiers=torch.eye(2).double())['features'][0])
    for i,a in enumerate(values):
        for b in values[i+1:]: assert not torch.allclose(a,b)


def test_positional_pool_equation_and_gradient_oracle():
    m = encoder()
    ids = [65, 67, 71]
    actual = m.pool(ids)
    # Independent scalar-coordinate sum: mean E * (1 + sin((p+1)/10000^(j/d))).
    coordinates = []
    for j in range(8):
        coordinates.append(sum(m.embedding.weight[t,j] *
            (1 + math.sin((p+1)/10000**(j/8))) for p,t in enumerate(ids))/3)
    expected = m.projection(torch.cat((torch.stack(coordinates),
        torch.tensor([math.log1p(3)],dtype=torch.float64))))
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    a = torch.autograd.grad(actual.sum(),tuple(m.parameters()),retain_graph=True)
    b = torch.autograd.grad(expected.sum(),tuple(m.parameters()))
    for x,y in zip(a,b): torch.testing.assert_close(x,y)


@pytest.mark.parametrize('case',['unknown','missing','undirected','limit'])
def test_fail_closed(case):
    m = encoder(max_tokens=100 if case == 'limit' else 8192)
    r = fixture('A'*200 if case == 'limit' else 'AC')
    select = SELECT
    if case == 'unknown': select = {('node',0,'typo')}
    if case == 'missing':
        r = replace(r,nodes=(Node('rna',(Measurement('sequence',None,None,'missing',False),)),r.nodes[1]))
    if case == 'undirected': r = replace(r,edges=(replace(r.edges[0],directed=False),))
    with pytest.raises(ValueError): m(r,input_fields=select,identifiers=torch.eye(2).double())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_gpu_features_and_gradients_match_cpu():
    cpu = encoder().float()
    gpu = copy.deepcopy(cpu).cuda()
    a = cpu(fixture(), input_fields=SELECT,identifiers=torch.eye(2))['features']
    b = gpu(fixture(), input_fields=SELECT,identifiers=torch.eye(2,device='cuda'))['features']
    torch.testing.assert_close(a,b.cpu(),atol=2e-6,rtol=2e-5)
    a.sum().backward(); b.sum().backward()
    for p,q in zip(cpu.parameters(),gpu.parameters()):
        torch.testing.assert_close(p.grad,q.grad.cpu(),atol=2e-6,rtol=2e-5)
