"""Exact source-text reference math, synthetic content, no source admission."""
import copy

import pytest
import torch
from torch.nn import functional as F
from test_pubchem_admet_mup import pubchem


def api():
    from mora3.model.source_text import SharedSourceTextModel,text_condition_graph
    return SharedSourceTextModel,text_condition_graph


@pytest.mark.parametrize('extra',[{'answer':'gold'},{'source_id':'identifier'}, {'target_length':9}, {'audit':{}}])
def test_text_conditions_refuse_target_or_audit_fields(extra):
    _,graph=api()
    with pytest.raises(ValueError):graph(dict(task='gsm8k_answer',question='Known question',**extra))


def test_text_condition_contains_only_full_known_question():
    _,graph=api();question='שאלה\n  known prompt\t'
    record,roles=graph(dict(task='gsm8k_answer',question=question))
    assert question in record.nodes[0].measurements[0].value or '\\n' in record.nodes[0].measurements[0].value
    assert record.nodes[0].measurements[1].value is None


@pytest.mark.parametrize('chunk,recompute',[(1,False),(3,False),(3,True),(1000,True)])
def test_all_native_bytes_eos_loss_and_full_history_gradients_match_unchunked(chunk,recompute):
    model_type,_=api();decoder=pubchem().eval();model=model_type(decoder).eval()
    conditions=({'task':'source_language_prior','language':'he','domain':'biblical'},
                {'task':'gsm8k_answer','question':'A source question\nwith complete whitespace'})
    targets=('בְּרֵאשִׁית\n  retained'.encode(),b'\nreasoning\t\n#### 42')
    assert {id(p) for p in model.parameters()}=={id(p) for p in decoder.parameters()}
    rng=torch.get_rng_state().clone()
    actual=model.objective(conditions,targets,chunk_size=chunk,recompute=recompute)
    context=model.condition(conditions);total=context.sum()*0;count=0
    for row,raw in enumerate(targets):
        ids=[decoder.codec.bos,*decoder.codec.tokenizer.encode(raw),decoder.codec.eos]
        assert decoder.codec.decode(ids)==raw
        tokens=torch.tensor(ids,dtype=torch.long)
        values,_=decoder.rnn(decoder.embedding(tokens[:-1][None]),context[row:row+1][None].contiguous())
        logits=decoder._readout(values)
        total+=F.cross_entropy(logits.reshape(-1,decoder.codec.size),tokens[1:],reduction='sum')
        count+=len(ids)-1
    expected=total/count
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    params=tuple(model.parameters())
    a=torch.autograd.grad(actual,params,allow_unused=True)
    b=torch.autograd.grad(expected,params,allow_unused=True)
    for x,y in zip(a,b):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-11)
    assert torch.equal(rng,torch.get_rng_state())
    by_identity={id(p):gradient for p,gradient in zip(params,a)}
    for p in (decoder.embedding.weight,decoder.rnn.weight_hh_l0,decoder.output.weight):
        assert by_identity[id(p)] is not None
        assert bool(by_identity[id(p)].abs().sum()>0)


def test_humaneval_is_evaluation_only_and_empty_native_text_keeps_eos():
    model_type,_=api();model=model_type(pubchem().eval())
    condition=({'task':'humaneval_solution','prompt':'def untouched_source_code():\n'},)
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='evaluation-only'):model.objective(condition,(b'return 1',))
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.isfinite(model.evaluation_objective(condition,(b'',),chunk_size=1,recompute=True))
    state=copy.deepcopy(model.state_dict());clone=copy.deepcopy(model)
    clone.load_state_dict(state)
    assert clone.decoder.shared is not model.decoder.shared
    assert clone.get_extra_state()==model.get_extra_state()


def test_resource_and_utf8_refusals_do_not_crop_targets_or_consume_rng():
    model_type,_=api();model=model_type(pubchem().eval(),max_total_target_bytes=2)
    conditions=({'task':'source_language_prior','language':'en','domain':'biomedical'},)
    for target in (b'abc',b'\xff'):
        rng=torch.get_rng_state().clone()
        with pytest.raises((ValueError,UnicodeDecodeError)):model.objective(conditions,(target,))
        assert torch.equal(rng,torch.get_rng_state())



def test_observed_native_root_list_opt_in_preserves_repeats_and_excludes_all_gold_context():
    import json
    model_type,graph=api();decoder=pubchem().eval()
    condition={'task':'oshb_source_root_list','word':'בְּרֵאשִׁית'}
    roots=['ראש','ברא','ראש'];target=json.dumps(roots,ensure_ascii=False,separators=(',',':')).encode()
    legacy=model_type(decoder)
    assert legacy.get_extra_state()['schema']=='Mora3.SharedSourceText/v1'
    with pytest.raises(ValueError,match='opt-in'):legacy.objective((condition,),(target,))
    model=model_type(decoder,root_list_policy='observed-native-root-list-v1')
    assert model.get_extra_state()['schema']=='Mora3.SharedSourceText/v2'
    assert {id(p)for p in model.parameters()}=={id(p)for p in decoder.parameters()}
    known=model.condition((condition,))
    loss=model.objective((condition,),(target,),chunk_size=3,recompute=True)
    assert torch.isfinite(loss)
    assert torch.equal(known,model.condition((condition,)))
    ids=[decoder.codec.bos,*decoder.codec.tokenizer.encode(target),decoder.codec.eos]
    assert json.loads(decoder.codec.decode(ids))==roots
    for key in ('roots','lemma','morph','lexical_entry_id','native_aug_code','audit'):
        with pytest.raises(ValueError):graph(dict(condition,**{key:'excluded-gold'}))


@pytest.mark.parametrize('raw',[b'[]',b'{}',b'[null]',b'[1]',b'[""]'])
def test_unlisted_or_nonliteral_root_targets_never_become_empty_eos_negatives(raw):
    model_type,_=api();model=model_type(pubchem().eval(),root_list_policy='observed-native-root-list-v1')
    condition=({'task':'oshb_source_root_list','word':'native-word'},)
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='root-string'):model.objective(condition,(raw,))
    assert torch.equal(rng,torch.get_rng_state())
