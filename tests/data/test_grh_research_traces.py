"""Whole-computation trace import into existing native prefix machinery."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import pytest
import torch
from mora3.data.grh_research_traces import (GRHNumericTrace,load_pinned_trace,
    as_reasoning_trace,verify_tropical_reference,REVISION,MANIFEST_SHA256)
from mora3.reasoning.trajectory_contracts import prefix_record,prefix_native_graph,check_trace_partitions

FIXTURES=Path(__file__).resolve().parents[1]/'fixtures/grh_research'


def encoded(data):
    raw=json.dumps(data,allow_nan=False,separators=(',',':')).encode()
    return GRHNumericTrace(raw,sha256(raw).hexdigest(),REVISION)


def tiny():
    def node(name,inputs,value):return dict(id=name,inputs=inputs,formula='known source operation',
        tensor=dict(shape=[],axes=[],dtype='float64',values=value))
    return dict(schema_version=1,id='test-only',title='Test-only numerical graph',
        provenance={'kind':'fixed-weight-reference'},nodes=[node('a',[],2.),node('b',['a'],3.),node('c',['a'],4.),node('d',['b','c'],7.)],checks={})


def test_entire_shared_dependency_graph_and_prefix_targets():
    original=tiny();source=encoded(original);trace=as_reasoning_trace(source)
    assert trace.split=='test'and len(trace.steps)==3 and len(trace.facts)==5
    assert all(step.direction_kind=='computational'for step in trace.steps)
    path=(trace.steps[0].key,);before=prefix_record(trace,path)
    assert {fact['name']for fact in before['state']['facts']}=={'program','tensor:a','tensor:b'}
    assert set(next(f for f in trace.facts if f.name=='tensor:d').depends_on)=={'tensor:b','tensor:c'}
    assert len(trace.complete_paths())==1 and len(trace.complete_paths()[0])==3
    changed=deepcopy(original);changed['nodes'][2]['tensor']['values']=999.;changed['nodes'][3]['tensor']['values']=-987.
    after=prefix_record(as_reasoning_trace(encoded(changed)),path)
    assert before['state']==after['state']and before['condition_sha256']==after['condition_sha256']
    assert before['context_sha256']==after['context_sha256']
    # Returned dictionaries cannot mutate the immutable retained source bytes.
    view=source.document();view['nodes'][0]['tensor']['values']=100
    assert source.document()['nodes'][0]['tensor']['values']==2.


def test_real_source_native_conditioning_has_nonzero_shared_gradients_and_target_exclusion():
    from mora3.model.cooperative_training import CooperativeModel
    from mora3.model.geometry_conditioning import GeometryConditioner
    source=load_pinned_trace(FIXTURES,'photonic-0');trace=as_reasoning_trace(source)
    path=(trace.steps[0].key,);graph,task,receipt=prefix_native_graph(trace,path)
    model=CooperativeModel(('A','B'),width=8).double()
    bridge=GeometryConditioner(model.response.backbone,id_width=8).double()
    value=bridge(graph,task);value.square().sum().backward()
    assert model.response.backbone.layers[0].attention.qkv.weight.grad.abs().sum()>0
    assert receipt['training_admission']is False
    changed=source.document();changed['nodes'][-1]['tensor']['values']=999999.
    other=as_reasoning_trace(encoded(changed));g,t,_=prefix_native_graph(other,path)
    assert torch.equal(value,bridge(g,t))


@pytest.mark.parametrize('name',('tropical-attention','causal-soft-moe','photonic-0'))
def test_all_selected_original_bytes_and_complete_programs(name):
    source=load_pinned_trace(FIXTURES,name);d=source.document();trace=as_reasoning_trace(source)
    assert sha256(source.raw).hexdigest()==source.sha256
    assert len(trace.facts)==len(d['nodes'])+1
    assert len(trace.steps)==sum(bool(n['inputs'])for n in d['nodes'])
    assert len(trace.nodes)==len(trace.steps)+1 and trace.split=='test'
    assert trace.groups[0]=='GRH-research-revision:'+REVISION
    last=prefix_record(trace,trace.complete_paths()[0]);assert len(last['state']['facts'])==len(trace.facts)
    assert all(f.evidence_kind=='computed'for f in trace.facts)
    assert check_trace_partitions((trace,))['source_rights_admitted']is False


def test_real_tropical_masked_max_and_all_winners_reference():
    source=load_pinned_trace(FIXTURES,'tropical-attention')
    result=verify_tropical_reference(source)
    assert result['context_max_absolute_error']<1e-12 and result['winner_mask_exact']
    assert result['masked_keys_ignored']and result['empty_queries_zero']
    assert result['mora_training_performed']is False and result['biological_causality_verified']is False
    changed=source.document();node=next(n for n in changed['nodes']if n['id']=='layer0.context')
    node['tensor']['values'][0][0][0][0]+=1
    assert verify_tropical_reference(encoded(changed))['context_pass']is False


def test_complex_parts_and_integer_boolean_types_are_preserved():
    d=tiny();d['nodes'][0]['tensor']=dict(shape=[2],axes=['complex_basis'],dtype='complex128',real=[1.,-0.],imag=[.5,2.])
    source=encoded(d);trace=as_reasoning_trace(source)
    fact=next(f for f in trace.facts if f.name=='tensor:a')
    assert json.loads(fact.value_json)==d['nodes'][0]['tensor']
    assert '-0.0'in fact.value_json


@pytest.mark.parametrize('fault',('sha','revision','future','duplicate','shape','dtype','bool-is-float','nonfinite','duplicate-input','noninteger-dimension','axes','budget'))
def test_bad_source_refuses_without_silent_repairs(fault):
    d=tiny()
    if fault=='future':d['nodes'][1]['inputs']=['d']
    elif fault=='duplicate':d['nodes'][1]['id']='a'
    elif fault=='shape':d['nodes'][0]['tensor']['shape']=[2]
    elif fault=='dtype':d['nodes'][0]['tensor']['dtype']='bytes-object'
    elif fault=='bool-is-float':d['nodes'][0]['tensor']['values']=True
    elif fault=='nonfinite':
        raw=json.dumps(d).replace('2.0','1e999').encode()
        with pytest.raises(ValueError):GRHNumericTrace(raw,sha256(raw).hexdigest(),REVISION)
        return
    elif fault=='duplicate-input':d['nodes'][1]['inputs']=['a','a']
    elif fault=='noninteger-dimension':d['nodes'][0]['tensor']['shape']=[True]
    elif fault=='axes':d['nodes'][0]['tensor']['axes']=['extra']
    raw=json.dumps(d).encode()
    with pytest.raises(ValueError):
        GRHNumericTrace(raw,'0'*64 if fault=='sha'else sha256(raw).hexdigest(),
            'not-a-revision'if fault=='revision'else REVISION,max_bytes=1 if fault=='budget'else 16*2**20)


def test_duplicate_json_keys_and_unavailable_trace_are_explicit(tmp_path):
    raw=b'{"schema_version":1,"schema_version":1}'
    with pytest.raises(ValueError):GRHNumericTrace(raw,sha256(raw).hexdigest(),REVISION)
    with pytest.raises((ValueError,FileNotFoundError)):load_pinned_trace(FIXTURES,'qcnn-9-0')
    with pytest.raises(ValueError):load_pinned_trace(FIXTURES,'../anything')


def test_complete_native_depth_budget_refuses_without_truncating():
    d=tiny();d['nodes']=d['nodes'][:1]
    for i in range(65):d['nodes'].append(dict(id='step'+str(i),inputs=[d['nodes'][-1]['id']],formula='step',tensor=dict(shape=[],axes=[],dtype='float64',values=1.)))
    with pytest.raises(ValueError,match='depth'):as_reasoning_trace(encoded(d))


def test_operation_specific_native_prefix_excludes_unrelated_siblings_and_target():
    from mora3.data.grh_research_traces import native_operation_prefix
    source=encoded(tiny());graph,task,report=native_operation_prefix(source,'c')
    native=graph.canonical_bytes().decode()
    assert 'tensor:a'in native and 'tensor:b'not in native and 'tensor:c'not in native and 'tensor:d'not in native
    assert report['scope']=='entire-ancestor-subgraph-before-target'
    assert report['training_admission']is False
    with pytest.raises(ValueError):native_operation_prefix(source,'a')


def test_loader_rejects_symlink_and_missing_manifest(tmp_path):
    (tmp_path/'manifest.json').symlink_to(FIXTURES/'manifest.json')
    with pytest.raises(ValueError):load_pinned_trace(tmp_path,'photonic-0')


@pytest.mark.parametrize('dtype,value',(('float32',1e40),('complex64',1e40),('int64',2**63),('bool',1),('float64',10**400)))
def test_dtype_overflow_or_coercion_is_never_silent(dtype,value):
    d=tiny();tensor=d['nodes'][0]['tensor'];tensor['dtype']=dtype
    if dtype.startswith('complex'):
        tensor.pop('values');tensor.update(real=value,imag=0.)
    else:tensor['values']=value
    with pytest.raises(ValueError):encoded(d)


def test_reference_import_cannot_authorize_training():
    trace=encoded(tiny())
    assert trace.training_admitted is False
    with pytest.raises(ValueError,match='evaluation-only'):trace.require_training_admission()
