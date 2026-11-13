"""Fixed finite controls and byte-exact typed forest serialization."""
from dataclasses import replace
import json
import pytest
import torch
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.tokenization.forest_tokens import ForestTokenAxis,FOREST_CONTROLS,serialize_token_forest,parse_token_forest
from test_forest_flow import skeleton


def base():
    return AnnotationVocabulary(json.dumps(dict(version=1,learned_hex=[b'abc'.hex(),b'\x00\xff'.hex()],
        controls=['pad','node','edge'],certificate={'fixture':True})))


def test_explicit_extension_preserves_payload_and_existing_control_ids():
    original=base();axis=ForestTokenAxis.from_annotation(original)
    old=original.tokenizer();new=axis.tokenizer()
    assert len(FOREST_CONTROLS)==40 and len(set(FOREST_CONTROLS))==40
    assert new.size==old.size+40 and new.vocabulary==old.vocabulary
    assert all(new.control_id(k)==old.control_id(k)for k in ('pad','node','edge'))
    payload=bytes(range(256))+b'abc\x00\xff<MORA:FOREST_BEGIN>'
    assert old.encode(payload)==new.encode(payload)
    assert new.decode(new.encode(payload))==payload
    assert not set(new.encode(b'<MORA:FINISH>'))&set(axis.control_ids)
    extended=axis.annotation_vocabulary()
    assert extended.tokenizer().size==new.size and json.loads(original.specification_json).get('forest_extension')is None
    assert json.loads(extended.specification_json)['forest_extension']['base_sha256']==axis.base_sha256
    assert extended.certificate==original.certificate
    assert axis.sha256!=axis.base_sha256
    with pytest.raises(ValueError):ForestTokenAxis.from_annotation(extended)


def test_all_branches_parenthood_direction_roles_and_bytes_roundtrip():
    axis=ForestTokenAxis.from_annotation(base());f=skeleton()
    payloads=(b'known context',b'abc\x00\xff',b'one conclusion',b'another route',b'two conclusion')
    tokens=serialize_token_forest(axis,f,payloads)
    decoded=parse_token_forest(axis,tokens)
    assert tuple(x['payload']for x in decoded)==payloads
    assert tuple(x['parent']for x in decoded)==f.parents
    assert tuple(x['direction']for x in decoded)==f.directions
    assert tuple(x['status']for x in decoded)==f.statuses
    assert [x['role']for x in decoded]==['condition']+['predicted']*4
    assert tokens[0]==axis.tokenizer().control_id('<MORA:FOREST_BEGIN>')
    assert tokens[-1]==axis.tokenizer().control_id('<MORA:FOREST_END>')


def test_multiple_roots_are_separate_complete_trees():
    axis=ForestTokenAxis.from_annotation(base());f=replace(skeleton(),
        node_ids=('a','b','c','d','e'),parents=(-1,0,1,-1,3),clamped=(0,3),
        directions=('root','computational','computational','root','temporal'))
    tokens=serialize_token_forest(axis,f,(b'A',b'B',b'C',b'D',b'E'))
    decoded=parse_token_forest(axis,tokens)
    assert tuple(x['parent']for x in decoded)==(-1,0,1,-1,3)
    assert tokens.count(axis.tokenizer().control_id('<MORA:TREE_BEGIN>'))==2


@pytest.mark.parametrize('fault',['missing-end','trailing','bad-parent','control-in-payload','huge','censored','payload-count'])
def test_structural_or_resource_errors_never_silently_repair(fault):
    axis=ForestTokenAxis.from_annotation(base());f=skeleton();tokens=serialize_token_forest(axis,f,(b'x',)*5);tok=axis.tokenizer()
    if fault=='missing-end':tokens=tokens[:-1]
    if fault=='trailing':tokens=tokens+(0,)
    if fault=='bad-parent':
        x=list(tokens);start=x.index(tok.control_id('<MORA:PARENT>'))+1
        end=x.index(tok.control_id('<MORA:RELATION>'));x[start:end]=tok.encode(b'123');tokens=tuple(x)
    if fault=='control-in-payload':
        x=list(tokens);pos=x.index(tok.control_id('<MORA:PREDICTED>'))+1;x.insert(pos,tok.control_id('<MORA:PASS>'));tokens=tuple(x)
    with pytest.raises(ValueError):
        if fault=='huge':parse_token_forest(axis,tokens,max_tokens=1)
        elif fault=='censored':serialize_token_forest(axis,replace(f,statuses=('open','open','censored','open','terminal')),(b'x',)*5)
        elif fault=='payload-count':serialize_token_forest(axis,f,(b'x',))
        else:parse_token_forest(axis,tokens)


def test_control_namespace_and_axis_tampering_refuse():
    axis=ForestTokenAxis.from_annotation(base())
    with pytest.raises(ValueError):replace(axis,controls=axis.controls[:-1])
    with pytest.raises(ValueError):replace(axis,learned_hex=('00',))
    with pytest.raises(ValueError):replace(axis,base_sha256='x')
