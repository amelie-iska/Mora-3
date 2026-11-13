"""Actual learned token-slot projection; source-frozen test codebooks are synthetic."""
from dataclasses import replace
import copy
import pytest
import torch
from test_forest_flow import make,problem,skeleton,state,SOURCE,COORD
from test_forest_tokens import base
from mora3.tokenization.forest_tokens import ForestTokenAxis,parse_token_forest
from mora3.model.forest_token_decoder import (ForestTokenCodebook,ForestDecoderSpec,NativeForestTokenDecoder,
    KnownForestPayloads,prepare_forest_token_targets,forest_token_loss,forest_decoder_parameter_roles)
from mora3.reasoning.flow_forest_refinement import DecodedForest,DecodeRefusal
from mora3.model.parameterization import parameter_ledger,width_family


def build(width=8,slots=8,cap=1000000):
    field=make(width);axis=ForestTokenAxis.from_annotation(base());g=torch.Generator().manual_seed(331)
    values=torch.randn(axis.tokenizer().size,3,dtype=torch.float64,generator=g)
    codebook=ForestTokenCodebook(axis,values,SOURCE,COORD,'c'*64,'d'*64,'e'*64)
    return NativeForestTokenDecoder(field,codebook,ForestDecoderSpec(slots,max_score_elements=cap)).double()


def known(f=None):
    f=f or skeleton()
    return KnownForestPayloads(f.context_sha256,COORD,((0,b'exact source condition'),),'f'*64)


def test_neural_slot_projection_has_exact_nearest_codebook_scores_and_gradients():
    d=build();r,t=problem();f=skeleton();x=state(2).requires_grad_()
    embeddings=d.token_embeddings(r,t,x,f);logits=d(r,t,x,f)
    assert embeddings.shape==(2,5,8,3)and logits.shape==(2,5,8,d.axis.tokenizer().size)
    allowed=list(d.allowed_token_ids)
    reference=-(embeddings[...,None,:]-d.codebook[allowed]).square().sum(-1)/d.spec.temperature
    # A token-independent ||query||² shift is omitted from production logits.
    observed=logits[...,allowed]-embeddings.square().sum(-1)[...,None]/d.spec.temperature
    assert torch.allclose(observed,reference,atol=1e-12,rtol=1e-12)
    assert torch.equal(logits[...,allowed].argmax(-1),reference.argmax(-1))
    loss=embeddings[:,1:].square().mean();loss.backward()
    for p in(d.slot_embedding.weight,d.readout.weight,d.field.state_projection.weight,d.field.geometry.graph.feature.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all()and p.grad.abs().sum()>0
    assert d.codebook.grad is None and x.grad[:,1:].abs().sum()>0


def test_targets_are_complete_byte_sequences_plus_stop_not_control_injection():
    d=build(slots=64);f=skeleton();payloads=((None,b'abc',b'answer',b'<MORA:PASS>',None),)
    targets,mask=prepare_forest_token_targets(d.axis,(f,),payloads,max_slots=64)
    assert not mask[:,0].any()and not mask[:,4].any()
    assert set(targets[mask].tolist())<=set(d.allowed_token_ids)
    assert d.axis.tokenizer().control_id('<MORA:PASS>')not in targets[mask]
    assert targets[0,1,int(mask[0,1].sum())-1]==d.stop_token
    with pytest.raises(ValueError):prepare_forest_token_targets(d.axis,(f,),((None,b'abcde'*100,b'x',b'x',b'x'),),max_slots=2)


def test_masked_decoder_loss_uses_real_labels_without_input_leakage():
    d=build(slots=8);r,t=problem();f=skeleton();x=state(1)
    target,mask=prepare_forest_token_targets(d.axis,(f,),((None,b'abc',b'abc',b'x',None),),max_slots=8)
    loss=forest_token_loss(d,r,t,x,f,target,mask)
    expected=torch.nn.functional.cross_entropy(d(r,t,x,f)[mask],target[mask])
    assert torch.equal(loss['loss'],expected)
    loss['loss'].backward();assert d.readout.weight.grad.abs().sum()>0
    assert d.codebook.grad is None
    r2,t2=problem(target=12345.)
    assert torch.equal(d(r,t,x,f),d(r2,t2,x,f))
    corrupted=target.clone();corrupted[~mask]=999999
    assert torch.equal(forest_token_loss(d,r,t,x,f,corrupted,mask)['loss'],expected)


@pytest.mark.parametrize('fault',['clamped','hole','not-stopped','control','after-stop'])
def test_bad_supervision_refuses(fault):
    d=build();r,t=problem();f=skeleton();x=state(1)
    target,mask=prepare_forest_token_targets(d.axis,(f,),((None,b'abc',b'x',b'x',b'x'),),max_slots=8)
    if fault=='clamped':mask[0,0,0]=True;target[0,0,0]=d.stop_token
    if fault=='hole':mask[0,1,0]=False
    if fault=='not-stopped':target[0,1,int(mask[0,1].sum())-1]=ord('x')
    if fault=='control':target[0,1,0]=d.axis.tokenizer().control_id('<MORA:SOURCE_OBSERVED>')
    if fault=='after-stop':mask[0,1,2]=True;target[0,1,2]=ord('x')
    with pytest.raises(ValueError):forest_token_loss(d,r,t,x,f,target,mask)


def test_bound_decoder_preserves_clamped_source_payloads_and_complete_grammar():
    d=build();r,t=problem();f=skeleton()
    # Deliberately set an untrained test readout to a known stop embedding. This
    # is a deterministic equation fixture, not an optimizer update or quality.
    with torch.no_grad():d.readout.weight.zero_();d.readout.bias.copy_(d.codebook[d.stop_token])
    decoder=d.bind_context(r,t,known(f));out=decoder(f)
    assert type(out)is DecodedForest and out.decoder_sha256==decoder.sha256
    rows=parse_token_forest(d.axis,out.tokens)
    assert rows[0]['payload']==b'exact source condition'
    assert all(row['payload']==b''for row in rows[1:])
    assert [row['role']for row in rows]==['condition']+['predicted']*4
    assert tuple(row['parent']for row in rows)==f.parents


def test_missing_stop_is_explicit_repairable_decode_refusal_not_fake_completion():
    d=build();r,t=problem();f=skeleton()
    with torch.no_grad():d.readout.weight.zero_();d.readout.bias.copy_(d.codebook[ord('x')])
    bound=d.bind_context(r,t,known(f));out=bound(f)
    assert type(out)is DecodeRefusal and out.reason=='token-slot-budget-without-stop'
    assert out.repairable is True and out.decoder_sha256==bound.sha256


def test_codebook_cannot_drift_or_load_different_values_under_same_axis():
    d=build();saved=copy.deepcopy(d.state_dict());before={n:p.clone()for n,p in d.named_parameters()}
    saved['readout.weight'].add_(10);saved['codebook'][0,0]+=1
    with pytest.raises(ValueError):d.load_state_dict(saved)
    assert all(torch.equal(before[n],p)for n,p in d.named_parameters())
    d.codebook[0,0]+=1
    with pytest.raises(ValueError,match='codebook'):d.get_extra_state()


def test_source_coordinate_and_known_payload_roles_are_bound():
    d=build();r,t=problem();f=skeleton()
    with pytest.raises(ValueError):d.bind_context(r,t,replace(known(),context_sha256='0'*64))
    wrong=replace(known(),payloads=((1,b'future target not a condition'),))
    with pytest.raises(ValueError):d.bind_context(r,t,wrong)(f)
    wrongbook=replace(d.source_codebook,source_snapshot_sha256='f'*64)
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):NativeForestTokenDecoder(d.field,wrongbook,d.spec)
    assert torch.equal(rng,torch.get_rng_state())


def test_actual_decoder_axis_roles_form_one_native_width_family():
    models=[build(w)for w in(8,12,16)];ledgers=[]
    for d in models:
        roles=forest_decoder_parameter_roles(d,identifier_axis=tuple('geometry:'+str(i)for i in range(8)))
        assert set(roles)==set(dict(d.named_parameters(remove_duplicate=False)))
        ledgers.append(parameter_ledger(d,roles))
    assert len(set(width_family(*ledgers).total_numel))==3


def test_missing_labels_are_inactive_and_known_context_never_becomes_supervision(monkeypatch):
    d=build();r,t=problem();f=skeleton();x=state(1)
    target=torch.full((1,5,8),-100,dtype=torch.long);mask=torch.zeros_like(target,dtype=torch.bool)
    monkeypatch.setattr(d,'forward',lambda *a:pytest.fail('inactive supervision must not run decoder'))
    assert forest_token_loss(d,r,t,x,f,target,mask)['loss']is None


def test_bounded_token_scoring_matches_dense_argmax_without_forest_truncation():
    dense=build();chunked=build(cap=1800);r,t=problem();f=skeleton();x=state(3)
    expected=dense(r,t,x,f).argmax(-1)
    assert torch.equal(chunked.nearest_token_ids(r,t,x,f),expected)
    assert chunked.nearest_token_ids(r,t,x,f).shape==(3,5,8)
    with pytest.raises(ValueError,match='score'):chunked(r,t,x,f)


def test_chunked_complete_token_loss_and_gradients_match_dense_reference():
    dense=build();chunked=build(cap=1800);r,t=problem();f=skeleton();x=state(2)
    targets,mask=prepare_forest_token_targets(dense.axis,(f,f),((None,b'abc',b'hi',b'x',b'y'),)*2,max_slots=8)
    a=forest_token_loss(dense,r,t,x,f,targets,mask)['loss'];b=forest_token_loss(chunked,r,t,x,f,targets,mask)['loss']
    assert torch.allclose(a,b,rtol=1e-13,atol=1e-13)
    a.backward();b.backward()
    for (_,p),(_,q)in zip(dense.named_parameters(),chunked.named_parameters()):
        if p.grad is not None:assert torch.allclose(p.grad,q.grad,atol=1e-12,rtol=1e-10)


def test_impossible_one_row_score_budget_refuses_before_initialization_rng():
    d=build();rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):NativeForestTokenDecoder(d.field,d.source_codebook,replace(d.spec,max_score_elements=1))
    assert torch.equal(rng,torch.get_rng_state())


def test_chunked_loss_recomputes_logits_in_backward_and_preserves_rng(monkeypatch):
    d=build(cap=1800);r,t=problem();f=skeleton();x=state(2);calls=[];original=d._scores
    def score(q,**kwargs):calls.append(q.shape[0]*len(d.codebook));return original(q,**kwargs)
    monkeypatch.setattr(d,'_scores',score)
    targets,mask=prepare_forest_token_targets(d.axis,(f,f),((None,b'abc',b'hi',b'x',b'y'),)*2,max_slots=8)
    rng=torch.get_rng_state().clone();loss=forest_token_loss(d,r,t,x,f,targets,mask)['loss'];before=len(calls)
    loss.backward()
    assert before>1 and len(calls)>before and max(calls)<=1800
    assert torch.equal(rng,torch.get_rng_state())


def test_concrete_neural_field_editor_scorer_decoder_execute_controller_together():
    from mora3.model.forest_flow import NativeForestSubtreeEditor
    from mora3.model.forest_quality import NativeForestQualityHead,ForestQualitySpec,ForestConstraintDecision
    from mora3.reasoning.flow_forest_refinement import (RefinementPolicy,ForestAssessment,DecodedAssessment,
        forest_fingerprint,decoded_fingerprint,run_flow_seeded_refinement)
    d=build();q=NativeForestQualityHead(d.field,ForestQualitySpec(('fixture','order'),'a'*64,(1.,1.)))
    d.eval();q.eval();r,t=problem();f=replace(skeleton(),directions=('root',)+('computational',)*4)
    # Same direction stratum: a temporal->computational revision is correctly a
    # different typed output, so it must not be called a duplicate in this test.
    # Deterministic empty-payload token fixture. This directly sets test readout
    # parameters, performs NO optimizer update and makes NO semantic-quality claim.
    with torch.no_grad():d.readout.weight.zero_();d.readout.bias.copy_(d.codebook[d.stop_token])
    field=d.field.bind_context(r,t);decoder=d.bind_context(r,t,known(f))
    def structural(candidate):
        return ForestConstraintDecision(forest_fingerprint(candidate),candidate.context_sha256,'e'*64,'pass','f'*64,True)
    scorer=q.bind_context(r,t,structural,constraint_policy_sha256='e'*64)
    editor=NativeForestSubtreeEditor(field,steps=2,generator=torch.Generator().manual_seed(71))
    def final(candidate,decoded):
        rows=parse_token_forest(d.axis,decoded.tokens)
        assert len(rows)==5 and rows[0]['payload']==b'exact source condition'
        return DecodedAssessment(decoded_fingerprint(decoded),candidate.context_sha256,'d'*64,'pass',1.,'f'*64)
    policy=RefinementPolicy(initial_candidates=4,active_candidates=2,return_outputs=2,
        edit_rounds=1,edits_per_candidate=2,total_candidates=8,minimum_score=0.,minimum_decoded_score=0.,
        correction_threshold=1.,minimum_distance=0.)
    out=run_flow_seeded_refinement(f,field,steps=2,generator=torch.Generator().manual_seed(72),policy=policy,
        assess=scorer,edit=editor,decode=decoder,assess_decoded=final,field_sha256=field.sha256,
        evaluator_sha256=scorer.sha256,decoder_sha256=decoder.sha256,decoded_evaluator_sha256='d'*64)
    assert len(out['initial_candidate_ids'])==4 and out['candidate_attempts']==8
    assert out['edit_rounds_completed']==1 and editor.last_receipt['velocity_evaluations']==2
    # All empty payloads have one semantic output identity: do not fabricate K.
    assert len(out['outputs'])==1 and out['status']=='insufficient-qualified-outputs'
    assert all(len(scorer.evidence(row['assessment'].evidence_sha256)['node_criterion_scores'])==5 for row in out['candidates'])
    assert out['gfn_on_policy']is False and out['training_admission']is False


def test_chunked_backward_rejects_changed_decoder_configuration():
    d=build(cap=1800);r,t=problem();f=skeleton();x=state(2)
    targets,mask=prepare_forest_token_targets(d.axis,(f,f),((None,b'abc',b'hi',b'x',b'y'),)*2,max_slots=8)
    loss=forest_token_loss(d,r,t,x,f,targets,mask)['loss']
    d.spec=replace(d.spec,temperature=2.)
    with pytest.raises(ValueError,match='configuration'):loss.backward()
