import copy
from dataclasses import replace
import pytest
import torch
from test_uniprot_ec_objective import view
from mora3.model import annotation_vocabulary as av
from mora3.model.uniprot_go import annotation_graph_task


def admitted(sequence='MAGMAG'):
    return replace(view({}),sequence=sequence,provenance_bytes=b'{"split":"train","license":"CC-BY-4.0","source_ref":{"source_release":"2026_03","raw_file_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}')


def test_fit_ignores_all_targets_and_roundtrips_full_sequence():
    a=admitted();b=replace(a,target_bytes=b'{"totally":"different"}')
    x=av.fit_annotation_vocabulary([a],learned_budget=2,max_length=3)
    y=av.fit_annotation_vocabulary([b],learned_budget=2,max_length=3)
    assert x==y
    tok=x.tokenizer();ids=tok.encode(a.sequence.encode())
    assert tok.decode(ids)==a.sequence.encode()
    assert len(ids)<len(a.sequence)
    assert x.certificate['max_constraint_violation']<1e-6


@pytest.mark.parametrize('change',[{'split':'test'},{'license':'unknown'}])
def test_fit_requires_train_rights(change):
    import json
    a=admitted();p=a.provenance();p.update(change)
    with pytest.raises(ValueError):av.fit_annotation_vocabulary([replace(a,provenance_bytes=json.dumps(p).encode())],learned_budget=2,max_length=3)


def test_bounds_reject_without_truncation():
    with pytest.raises(ValueError,match='byte'):
        av.fit_annotation_vocabulary([admitted()],learned_budget=2,max_length=3,max_bytes=2)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_bound_payload_excludes_targets_and_restores_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    a=admitted();b=replace(a,target_bytes=b'{}')
    spec=av.fit_annotation_vocabulary([a],learned_budget=2,max_length=3)
    model=av.AnnotationPayloadEncoder(spec,4).to(device)
    def encode(m,v):
        r,t=annotation_graph_task(v)
        return m.forward_task(r,task=t,identifiers=torch.eye(1,device=device))['features']
    assert torch.equal(encode(model,a),encode(model,b))
    clone=av.AnnotationPayloadEncoder(spec,4).to(device);clone.load_state_dict(copy.deepcopy(model.state_dict()))
    encode(model,a).sum().backward();encode(clone,a).sum().backward()
    for p,q in zip(model.parameters(),clone.parameters()):assert torch.equal(p.grad,q.grad)
    other=av.fit_annotation_vocabulary([admitted('GAMGAM')],learned_budget=2,max_length=3)
    with pytest.raises(ValueError,match='vocabulary'):
        av.AnnotationPayloadEncoder(other,4).load_state_dict(model.state_dict())
