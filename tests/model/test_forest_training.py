"""Synthetic, zero-update integration of all four real native forest heads."""
from dataclasses import replace
import copy
import pytest
import torch
from test_forest_flow import make, problem, skeleton, state, SOURCE, COORD
from test_forest_tokens import base
from mora3.tokenization.forest_tokens import ForestTokenAxis
from mora3.model.forest_token_decoder import (ForestTokenCodebook, ForestDecoderSpec,
    NativeForestTokenDecoder, prepare_forest_token_targets, forest_token_loss)
from mora3.model.forest_quality import (ForestQualitySpec, NativeForestQualityHead,
    ForestQualityLabels, forest_quality_loss)
from mora3.model.forest_analogy import (ForestAnalogySpec, NativeForestAnalogyHead,
    ForestAnalogyItem, forest_analogy_loss)
from mora3.model.forest_flow import forest_flow_matching_loss
from mora3.model.forest_training import (ForestTrainingHeads, ForestTrainingSpec,
    ForestBatchIdentity, ForestTrainingBatch, FlowSupervision, TokenSupervision,
    QualitySupervision, AnalogySupervision, preflight_forest_batch,
    forest_joint_loss, forest_batch_sha256)


def fixture(width=8):
    flow=make(width);axis=ForestTokenAxis.from_annotation(base())
    values=torch.randn(axis.tokenizer().size,3,dtype=torch.float64,
                       generator=torch.Generator().manual_seed(331))
    book=ForestTokenCodebook(axis,values,SOURCE,COORD,'c'*64,'d'*64,'e'*64)
    decoder=NativeForestTokenDecoder(flow,book,ForestDecoderSpec(8,max_score_elements=1800))
    quality=NativeForestQualityHead(flow,ForestQualitySpec(('order','constraint'),'f'*64,(1.,2.)))
    analogy=NativeForestAnalogyHead(flow,ForestAnalogySpec(4,'d'*64,max_memory=16))
    heads=ForestTrainingHeads(flow,decoder,quality,analogy)
    spec=ForestTrainingSpec(SOURCE,COORD,'c'*64,'d'*64,'e'*64,heads.sha256,(1.,.3,.2,.1))
    r,t=problem();f=skeleton();x=state(2)
    mask=torch.ones_like(x,dtype=torch.bool);mask[:,0]=False
    token,tm=prepare_forest_token_targets(axis,(f,f),((None,b'abc',b'hi',b'x',None),)*2,max_slots=8)
    whole=torch.tensor([[1.,0.],[.5,1.]],dtype=x.dtype)
    node=torch.zeros(2,5,2,dtype=x.dtype);nm=torch.ones_like(node,dtype=torch.bool);nm[:,0]=False
    labels=ForestQualityLabels(whole,node,torch.ones_like(whole,dtype=torch.bool),nm,SOURCE,'f'*64,'9'*64,'train')
    queries=(ForestAnalogyItem('q0',r,t,None,('q0',),'train','d'*64,'1'*64),)
    memories=tuple(ForestAnalogyItem('m'+str(i),r,t,replace(f,states=state(1)[0]+i),
        ('m'+str(i),),'train','d'*64,'2'*64)for i in range(2))
    pairs=AnalogySupervision(queries,memories,torch.tensor([[True,False]]),torch.tensor([[True,True]]),'8'*64)
    identity=ForestBatchIdentity(SOURCE,COORD,'c'*64,'d'*64,'e'*64,'synthetic-v1',
        ('example0','example1'),(17,29),(('group0',),('group1',)),'train','synthetic_fixture','7'*64)
    batch=ForestTrainingBatch(identity,r,t,f,FlowSupervision(x+.2,mask),
        TokenSupervision(x.clone(),token,tm,'6'*64),QualitySupervision(x.clone(),labels),pairs)
    return heads,spec,batch


def test_four_objective_equations_and_shared_gradients():
    h,s,b=fixture();noise=state();time=torch.tensor([.2,.8],dtype=noise.dtype)
    actual=forest_joint_loss(h,s,b,noise=noise,time=time)
    refs=(forest_flow_matching_loss(h.flow,b.record,b.task,b.template,b.flow.values,noise,time,b.flow.available)['loss'],
        forest_token_loss(h.decoder,b.record,b.task,b.tokens.states,b.template,b.tokens.values,b.tokens.available)['loss'],
        forest_quality_loss(h.quality,b.record,b.task,b.quality.states,b.template,b.quality.labels)['loss'],
        forest_analogy_loss(h.analogy,b.analogy.queries,b.analogy.memories,b.analogy.positive,b.analogy.known)['loss'])
    expected=sum(w*x for w,x in zip(s.weights,refs))
    torch.testing.assert_close(actual['loss'],expected,atol=0,rtol=0)
    assert actual['counts']=={'flow':16,'tokens':14,'quality':20,'analogy':2}
    params=(h.flow.readout.weight,h.decoder.readout.weight,h.quality.forest_readout.weight,
            h.analogy.query_readout.weight,h.analogy.memory_readout.weight,h.flow.geometry.graph.feature.weight)
    actual_grads=torch.autograd.grad(actual['loss'],params,retain_graph=True)
    reference_grads=torch.autograd.grad(expected,params)
    for a,e in zip(actual_grads,reference_grads):
        torch.testing.assert_close(a,e,atol=1e-12,rtol=1e-10)
        assert torch.isfinite(a).all()and a.abs().sum()>0
    assert all(p.grad is None for p in h.flow.parameters())
    assert actual['training_admission']is False


@pytest.mark.parametrize('fault',('heldout','source','coordinate','encoder','group-policy','skeleton-policy',
    'generated-skeleton','root-supervision','token-stop','token-prefix','quality-policy','quality-heldout',
    'quality-nan','analogy-heldout','analogy-unknown-positive','decoder-gradient','wrong-batch','same-group-positive','quality-dtype'))
def test_all_inputs_refuse_before_any_neural_call(fault):
    h,s,b=fixture()
    if fault=='quality-dtype':h.quality.node_readout.float()
    if fault=='heldout':b=replace(b,identity=replace(b.identity,split='test'))
    if fault=='source':b=replace(b,identity=replace(b.identity,source_snapshot_sha256='0'*64))
    if fault=='coordinate':b=replace(b,identity=replace(b.identity,coordinate_sha256='0'*64))
    if fault=='encoder':b=replace(b,identity=replace(b.identity,encoder_sha256='0'*64))
    if fault=='group-policy':b=replace(b,identity=replace(b.identity,group_policy_sha256='0'*64))
    if fault=='skeleton-policy':b=replace(b,identity=replace(b.identity,skeleton_policy_sha256='0'*64))
    if fault=='generated-skeleton':b=replace(b,template=replace(b.template,skeleton_origin='generated'))
    if fault=='root-supervision':b.flow.available[:,0]=True
    if fault=='token-stop':b.tokens.values[0,1,int(b.tokens.available[0,1].sum())-1]=0
    if fault=='token-prefix':b.tokens.available[0,1,0]=False
    if fault=='quality-policy':b=replace(b,quality=replace(b.quality,labels=replace(b.quality.labels,label_policy_sha256='0'*64)))
    if fault=='quality-heldout':b=replace(b,quality=replace(b.quality,labels=replace(b.quality.labels,split='validation')))
    if fault=='quality-nan':b.quality.labels.whole_values[0,0]=float('nan')
    if fault=='analogy-heldout':b=replace(b,analogy=replace(b.analogy,memories=(replace(b.analogy.memories[0],split='test'),b.analogy.memories[1])))
    if fault=='analogy-unknown-positive':b.analogy.known[0,0]=False
    if fault=='decoder-gradient':b.tokens.states.requires_grad_()
    if fault=='wrong-batch':b=replace(b,flow=replace(b.flow,values=b.flow.values[:1]))
    if fault=='same-group-positive':b=replace(b,analogy=replace(b.analogy,memories=(replace(b.analogy.memories[0],groups=('q0',)),b.analogy.memories[1])))
    calls=[]
    handles=[m.register_forward_pre_hook(lambda *args:calls.append(True))for m in(h.flow,h.decoder,h.quality,h.analogy,h.flow.geometry)]
    rng=torch.get_rng_state().clone()
    try:
        with pytest.raises(ValueError):forest_joint_loss(h,s,b,noise=state(),time=torch.tensor([.2,.8],dtype=torch.float64))
    finally:
        for handle in handles:handle.remove()
    assert calls==[]and torch.equal(rng,torch.get_rng_state())


def test_unknown_labels_are_not_zeros_or_mixture_renormalization():
    h,s,b=fixture();b=replace(b,flow=None,quality=None,analogy=None)
    reference=forest_token_loss(h.decoder,b.record,b.task,b.tokens.states,b.template,b.tokens.values,b.tokens.available)['loss']
    actual=forest_joint_loss(h,s,b)
    torch.testing.assert_close(actual['loss'],s.weights[1]*reference,atol=0,rtol=0)
    assert actual['counts']=={'flow':0,'tokens':14,'quality':0,'analogy':0}
    b.tokens.available.zero_();b.tokens.values.fill_(999999)
    assert forest_joint_loss(h,s,b)['loss']is None


def test_missing_nonfinite_labels_mask_before_arithmetic_and_do_not_receive_gradients():
    h,s,b=fixture();b.flow.values[:,0]=float('nan');b.flow.values.requires_grad_()
    b.quality.labels.node_values[:,0]=float('nan');b.quality.labels.node_values.requires_grad_()
    out=forest_joint_loss(h,s,b,noise=state(),time=torch.tensor([.2,.8],dtype=torch.float64))
    out['loss'].backward()
    assert b.flow.values.grad is None and b.quality.labels.node_values.grad is None
    assert torch.isfinite(out['loss'])


def test_hidden_graph_targets_do_not_change_loss_or_input_identity():
    h,s,b=fixture();r,t=problem(target=9000.)
    other=replace(b,record=r,task=t)
    a=forest_joint_loss(h,s,b,noise=state(),time=torch.tensor([.3,.7],dtype=torch.float64))
    z=forest_joint_loss(h,s,other,noise=state(),time=torch.tensor([.3,.7],dtype=torch.float64))
    torch.testing.assert_close(a['loss'],z['loss'],atol=0,rtol=0)
    assert forest_batch_sha256(b)!=forest_batch_sha256(other)


def test_head_alias_drift_or_changed_codebook_refuses():
    h,s,b=fixture();alien=copy.deepcopy(h.decoder)
    with pytest.raises(ValueError):ForestTrainingHeads(h.flow,alien,h.quality,h.analogy).binding()
    h.decoder.codebook[0,0]+=1
    with pytest.raises(ValueError):preflight_forest_batch(h,s,b)


def test_no_objectives_is_unavailable_and_spec_weights_are_exact():
    h,s,b=fixture();empty=replace(b,flow=None,tokens=None,quality=None,analogy=None)
    assert forest_joint_loss(h,s,empty)['loss']is None
    for weights in ((0.,0.,0.,0.),(True,1.,1.,1.),(float('nan'),1.,1.,1.),(-1.,1.,1.,1.)):
        with pytest.raises(ValueError):replace(s,weights=weights).__post_init__()
