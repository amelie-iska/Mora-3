import json
from dataclasses import replace
import pytest
import torch
from mora3.data.uniprot_annotations import AnnotationPredictionView
from mora3.model import uniprot_go as go
from mora3.model.payload import PayloadEncoder
from mora3.tokenization.shortest_path import ByteTokenizer


def view(target='secret'):
    p=dict(split='train',license='CC-BY-4.0',source_ref=dict(source_release='2026_03',raw_file_sha256='a'*64))
    return AnnotationPredictionView('P12345','MAG',json.dumps({'annotation_packet':{'text':target}}).encode(),json.dumps(p).encode())


def test_bridge_preserves_full_target_but_encodes_sequence_only():
    assert hasattr(go,'annotation_graph_task'), 'shared graph/task bridge missing'
    a,t=go.annotation_graph_task(view())
    b,u=go.annotation_graph_task(view('changed'*10000))
    assert t.canonical_bytes()==u.canonical_bytes()
    assert t.input_fields(a)==frozenset({('node',0,'sequence')})
    assert json.loads(a.nodes[0].measurements[1].value)==view().targets()
    enc=PayloadEncoder(ByteTokenizer(controls=('pad','node','edge')),8)
    x=enc.forward_task(a,task=t,identifiers=torch.eye(1))
    y=enc.forward_task(b,task=u,identifiers=torch.eye(1))
    assert torch.equal(x['features'],y['features'])
    # Missing targets at inference cannot affect features or task mask.
    missing=replace(a,nodes=(replace(a.nodes[0],measurements=(a.nodes[0].measurements[0],replace(a.nodes[0].measurements[1],value=None,status='missing',available=False))),))
    assert torch.equal(x['features'],enc.forward_task(missing,task=t,identifiers=torch.eye(1))['features'])
    assert t.available_targets(missing)==()


@pytest.mark.parametrize('key,value',[('split','test'),('license','unknown')])
def test_bridge_fails_closed_on_unsupported_scope(key,value):
    assert hasattr(go,'annotation_graph_task'), 'shared graph/task bridge missing'
    v=view();p=v.provenance();p[key]=value
    with pytest.raises(ValueError):go.annotation_graph_task(replace(v,provenance_bytes=json.dumps(p).encode()))
