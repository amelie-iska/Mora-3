"""Test-only tiny neural windows; no production source or optimizer updates."""
from dataclasses import replace
import copy
import pytest
import torch
from test_forest_token_decoder import build
from test_forest_flow import problem,skeleton,state
from mora3.model.forest_window_continuation import NativeForestWindowContinuation,WindowContinuationSpec


def setup(slots=4):
    d=build(slots=8); r,t=problem(); f=skeleton(); x=state(1)
    c=NativeForestWindowContinuation(d,WindowContinuationSpec(slots,32,128,100000)).double()
    return c,r,t,f,x


def test_true_causal_windows_and_unmodified_token_axis():
    c,r,t,f,x=setup(); stop=c.decoder.stop_token
    a=torch.tensor([ord('a'),ord('b'),ord('c'),ord('d'),ord('e'),ord('f'),ord('g'),ord('h'),stop])
    b=a.clone(); b[6]=ord('z')
    before=c.teacher_logits(r,t,x,f,1,a)
    changed=c.teacher_logits(r,t,x,f,1,b)
    assert before.shape==(9,c.decoder.codebook.shape[0])
    assert torch.allclose(before[:7],changed[:7],atol=0,rtol=0)
    assert not torch.allclose(before[7],changed[7])
    assert c.decoder.axis.sha256==c.decoder.source_codebook.axis.sha256
    assert c.last_window_lengths(a)==(4,4,1)


def test_complete_supervision_gradients_and_one_final_stop():
    c,r,t,f,x=setup(); stop=c.decoder.stop_token
    tokens=torch.tensor([ord('a'),ord('b'),ord('c'),ord('d'),stop])
    scores=c.teacher_logits(r,t,x,f,1,tokens)
    loss=c.teacher_loss(r,t,x,f,1,tokens)
    expected=torch.nn.functional.cross_entropy(scores,tokens)
    assert torch.allclose(loss,expected,atol=1e-13,rtol=1e-13)
    loss.backward()
    for p in (c.prefix_input.weight,c.causal_gru.weight_ih_l0,c.decoder.readout.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    with pytest.raises(ValueError,match='stop'):
        c.teacher_logits(r,t,x,f,1,torch.tensor([ord('a'),ord('b')]))
    with pytest.raises(ValueError,match='stop'):
        c.teacher_logits(r,t,x,f,1,torch.tensor([stop,ord('a'),stop]))


def test_window_lengths_and_exact_checkpoint_recovery_no_rng_draw():
    c,r,t,f,x=setup(); stop=c.decoder.stop_token
    a=torch.tensor([ord('a')]*11+[stop]); g=torch.get_rng_state().clone()
    expected=c.teacher_logits(r,t,x,f,1,a)
    assert c.last_window_lengths(a)==(4,4,4)
    assert torch.equal(g,torch.get_rng_state())
    sd=copy.deepcopy(c.state_dict()); c.load_state_dict(sd)
    assert torch.equal(expected,c.teacher_logits(r,t,x,f,1,a))
    bad=copy.deepcopy(sd); bad['decoder.codebook'][0,0]+=0.1
    with pytest.raises(ValueError): c.load_state_dict(bad)
    with pytest.raises(ValueError,match='capacity'):
        c.teacher_logits(r,t,x,f,1,torch.tensor([ord('a')]*128+[stop]))


def test_streamed_chunked_scores_and_long_cross_boundary_loss():
    c,r,t,f,x=setup(); stop=c.decoder.stop_token
    tokens=torch.tensor([ord('a')+i%12 for i in range(19)]+[stop])
    logits=c.teacher_logits(r,t,x,f,1,tokens)
    loss=c.teacher_loss(r,t,x,f,1,tokens)
    assert torch.allclose(loss,torch.nn.functional.cross_entropy(logits,tokens),atol=1e-12)
    from dataclasses import replace
    c.spec=replace(c.spec,max_score_elements=len(c.decoder.codebook))
    # A frozen declared budget cannot be silently changed to make a run pass.
    with pytest.raises(ValueError,match='drift'):c.teacher_loss(r,t,x,f,1,tokens)
    c.spec=WindowContinuationSpec(4,32,128,100000)
    assert torch.allclose(c.teacher_loss(r,t,x,f,1,tokens),loss,atol=1e-12)


def test_greedy_inference_matches_next_token_teacher_scores_and_shortfall():
    c,r,t,f,x=setup();stop=c.decoder.stop_token
    with torch.no_grad():
        c.decoder.readout.weight.zero_()
        c.decoder.readout.bias.copy_(c.decoder.codebook[ord('x')])
    rng=torch.get_rng_state().clone()
    result=c.generate_token_ids(r,t,x,f,1,max_steps=7)
    assert result['complete'] is False and result['reason']=='token-budget-exhausted'
    assert result['tokens']==(ord('x'),)*7
    prefix=torch.tensor(result['tokens']+(stop,))
    scores=c.teacher_logits(r,t,x,f,1,prefix)
    assert tuple(scores[:7].argmax(-1).tolist())==result['tokens']
    assert torch.equal(rng,torch.get_rng_state())
    with torch.no_grad():c.decoder.readout.bias.copy_(c.decoder.codebook[stop])
    done=c.generate_token_ids(r,t,x,f,1,max_steps=7)
    assert done['complete'] is True and done['tokens']==(stop,)
    assert done['reason']=='final-stop'
    assert c.teacher_logits(r,t,x,f,1,torch.tensor([stop]))[0].argmax()==stop
