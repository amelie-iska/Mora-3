"""Native learned complete-forest flow: no optimizer steps or training admission."""
import copy
from dataclasses import replace
import pytest
import torch
from mora3.data.schema import GraphRecord,Node,Edge,Measurement
from mora3.data.provenance import Provenance
from mora3.data.tasks import ConditionalTask
from mora3.model.tokengt import TokenGT
from mora3.model.geometry_conditioning import GeometryConditioner
from mora3.model.parameterization import parameter_ledger,width_family
from mora3.reasoning.flow_forest_refinement import ForestState,sample_flow_forests
from mora3.model.forest_flow import (ForestFlowSpec,NativeForestFlowHead,
    forest_context_sha256,forest_flow_matching_loss,forest_flow_parameter_roles,
    native_forest_flow_objective)

SOURCE='a'*64
COORD='b'*64


def problem(target=5.,dose=1.):
    record=GraphRecord('fixture-not-training',(
        Node('molecule',(Measurement('dose',dose,'dimensionless','observed',True),
                         Measurement('heldout',target,'dimensionless','observed',True))),
        Node('cell',(Measurement('cell_kind','fixture',None,'observed',True),))),
        (Edge(0,1,'assay-context',True),),
        (Provenance('synthetic','v1','fixture','c'*64,'test-only'),))
    task=ConditionalTask(((('node',0,'dose'),'condition'),(('node',0,'heldout'),'target'),
                          (('node',1,'cell_kind'),'condition')),topology_known=True)
    return record,task


def skeleton(record=None,task=None):
    if record is None:record,task=problem()
    return ForestState(('root','left','left-end','right','right-end'),(-1,0,1,0,3),
        ('open','open','terminal','open','terminal'),torch.tensor([[.2,.4],[9.,9.],[9.,9.],[9.,9.],[9.,9.]],dtype=torch.float64),
        (0,),('root','computational','computational','temporal','temporal'),
        forest_context_sha256(record,task,source_snapshot_sha256=SOURCE),COORD)


def make(width=8,*,legacy_tropical=False,spec=None):
    torch.manual_seed(20261113)
    kwargs=dict(attention='tropical-blockwise')if legacy_tropical else dict(
        scaling_policy='mup-v1',initialization_policy='mup-transformer-v1',
        parameterization_policy='mup-tokengt-v1',base_width=8,base_ffn_width=8)
    backbone=TokenGT(width,2,width=width,heads=2,layers=1,ffn_width=width,**kwargs)
    geometry=GeometryConditioner(backbone,id_width=8,**({}if legacy_tropical else dict(
        parameterization_policy='mup-geometry-v1',base_width=8)))
    spec=spec or ForestFlowSpec(dimension=2,source_snapshot_sha256=SOURCE,coordinate_sha256=COORD,
                                max_nodes=8,max_batch_size=8,max_attention_elements=100000)
    return NativeForestFlowHead(geometry,spec).double()


def state(batch=2):
    return torch.randn(batch,5,2,generator=torch.Generator().manual_seed(7),dtype=torch.float64)


def test_missing_native_head_is_real_trainable_and_shared():
    head=make();record,task=problem();x=state().requires_grad_();time=torch.tensor([.2,.7],dtype=x.dtype)
    y=head(record,task,x,time,skeleton());assert y.shape==x.shape
    assert torch.equal(y[:,0],torch.zeros_like(y[:,0]))
    (y[:,1:].square().sum()).backward()
    for p in(head.state_projection.weight,head.time_projection.weight,head.structure_projection.weight,
              head.readout.weight,head.geometry.payload.embedding.weight,head.geometry.graph.feature.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert x.grad[:,1:].abs().sum()>0 and torch.equal(x.grad[:,0],torch.zeros_like(x.grad[:,0]))
    source=head.geometry._source_backbone_reference[0]
    assert all(getattr(head.geometry.graph,k)is getattr(source,k)for k in('feature','role','layers','final_norm','graph_token'))


def test_hidden_targets_and_unclamped_template_values_do_not_enter_condition():
    h=make();r,t=problem();r2,t2=problem(target=1234567.)
    f=skeleton(r,t);f2=skeleton(r2,t2);f2.states[1:]=-777
    assert f.context_sha256==f2.context_sha256
    x=state();time=torch.full((2,),.3,dtype=x.dtype)
    assert torch.equal(h(r,t,x,time,f),h(r2,t2,x,time,f2))
    r3,t3=problem(dose=2.);f3=skeleton(r3,t3)
    assert f3.context_sha256!=f.context_sha256
    assert not torch.equal(h(r,t,x,time,f),h(r3,t3,x,time,f3))


def test_batched_matches_individual_and_depends_on_all_noisy_branches():
    h=make();r,t=problem();f=skeleton();x=state(3);ts=torch.tensor([.1,.4,.8],dtype=x.dtype)
    out=h(r,t,x,ts,f)
    assert torch.allclose(out,torch.cat([h(r,t,x[i:i+1],ts[i:i+1],f)for i in range(3)]),atol=1e-12,rtol=1e-12)
    altered=x.clone();altered[:,3:]+=2
    assert not torch.equal(out[:,1:3],h(r,t,altered,ts,f)[:,1:3])
    reverse=replace(f,directions=('root','temporal','temporal','computational','computational'))
    assert not torch.equal(out,h(r,t,x,ts,reverse))


def test_full_noisy_state_input_gradient_matches_finite_difference():
    h=make();r,t=problem();f=skeleton();x=state(1).requires_grad_();ts=torch.tensor([.4],dtype=x.dtype)
    v=h(r,t,x,ts,f)[0,2,0];g=torch.autograd.grad(v,x)[0][0,4,1]
    eps=1e-6;plus=x.detach().clone();minus=plus.clone();plus[0,4,1]+=eps;minus[0,4,1]-=eps
    ref=(h(r,t,plus,ts,f)[0,2,0]-h(r,t,minus,ts,f)[0,2,0])/(2*eps)
    assert torch.allclose(g,ref,atol=2e-8,rtol=2e-5)


def test_masked_cfm_equation_uses_no_zero_labels_and_no_target_gradients():
    h=make();r,t=problem();f=skeleton();noise=state();target=(state()+.3).requires_grad_()
    mask=torch.ones_like(noise,dtype=torch.bool);mask[:,0]=False;mask[0,3,1]=False
    with torch.no_grad():target[0,3,1]=float('nan')
    ts=torch.tensor([.25,.75],dtype=noise.dtype)
    result=forest_flow_matching_loss(h,r,t,f,target,noise,ts,mask)
    rate=1-h.spec.sigma_min;safe=torch.where(mask,target.detach(),torch.zeros_like(target))
    xt=torch.where(mask,(1-rate*ts[:,None,None])*noise+ts[:,None,None]*safe,noise)
    xt[:,0]=f.states[0]
    pred=h(r,t,xt,ts,f);velocity=safe-rate*noise
    oracle=(pred[mask]-velocity[mask]).square().mean()
    assert torch.equal(result['loss'],oracle)and result['supervised_coordinates']==15
    result['loss'].backward();assert target.grad is None
    assert result['training_admission']is False


def test_no_supervision_is_inactive_and_clamped_supervision_refuses(monkeypatch):
    h=make();r,t=problem();f=skeleton();x=state();time=torch.tensor([.2,.6],dtype=x.dtype)
    monkeypatch.setattr(h,'forward',lambda *a:pytest.fail('no supervised target must not execute model'))
    result=forest_flow_matching_loss(h,r,t,f,torch.full_like(x,float('nan')),x,time,torch.zeros_like(x,dtype=torch.bool))
    assert result['loss']is None and result['supervised_coordinates']==0
    with pytest.raises(ValueError,match='clamped'):
        forest_flow_matching_loss(h,r,t,f,x,x,time,torch.ones_like(x,dtype=torch.bool))


@pytest.mark.parametrize('change',[{'dimension':True},{'dimension':0},{'dimension':257},{'max_nodes':0},
    {'max_batch_size':False},{'max_attention_elements':0},{'sigma_min':1.},
    {'sigma_min':float('nan')},{'coordinate_sha256':'bad'},{'source_snapshot_sha256':'bad'}])
def test_bad_specification_refuses(change):
    kw=dict(dimension=2,source_snapshot_sha256=SOURCE,coordinate_sha256=COORD);kw.update(change)
    with pytest.raises(ValueError):ForestFlowSpec(**kw)


def test_rank_overflow_refuses_before_constructor_rng():
    h=make();rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='rank'):
        NativeForestFlowHead(h.geometry,replace(h.spec,max_nodes=9))
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('kind',['coordinate','context','capacity','dtype','time','nan','incomplete'])
def test_current_boundaries_refuse_before_forward(kind):
    h=make();r,t=problem();f=skeleton();x=state();ts=torch.tensor([.2,.4],dtype=x.dtype)
    if kind=='coordinate':f=replace(f,coordinate_sha256='d'*64)
    if kind=='context':f=replace(f,context_sha256='e'*64)
    if kind=='capacity':x=state(9);ts=torch.zeros(9,dtype=x.dtype)
    if kind=='dtype':x=x.float();ts=ts.float()
    if kind=='time':ts[0]=1.1
    if kind=='nan':x[0,2,0]=float('nan')
    if kind=='incomplete':f=replace(f,statuses=('open','open','censored','open','terminal'))
    with pytest.raises(ValueError):h(r,t,x,ts,f)


def test_bound_native_sampler_is_reproducible_and_rejects_model_drift():
    h=make();r,t=problem();f=skeleton();bound=h.bind_context(r,t)
    rng=torch.get_rng_state().clone()
    a=sample_flow_forests(f,bound,count=4,steps=3,generator=torch.Generator().manual_seed(90),field_sha256=bound.sha256)
    b=sample_flow_forests(f,bound,count=4,steps=3,generator=torch.Generator().manual_seed(90),field_sha256=bound.sha256)
    assert len(a['forests'])==4 and all(torch.equal(x.states,y.states)for x,y in zip(a['forests'],b['forests']))
    assert all(torch.equal(x.states[0],f.states[0])for x in a['forests'])
    assert not torch.equal(a['forests'][0].states,a['forests'][1].states)
    assert torch.equal(rng,torch.get_rng_state())
    with torch.no_grad():h.readout.weight.add_(.01)
    with pytest.raises(ValueError,match='changed|drift'):
        bound(state(),torch.tensor([.2,.4],dtype=torch.float64),f)


def test_full_head_roles_and_finite_axes_across_widths():
    heads=[make(w)for w in(8,12,16)];ledgers=[]
    for h in heads:
        roles=forest_flow_parameter_roles(h,identifier_axis=tuple('geometry:'+str(i)for i in range(8)))
        assert set(roles)==set(dict(h.named_parameters(remove_duplicate=False)))
        ledgers.append(parameter_ledger(h,roles))
        extra=sum(p.numel()for p in h.parameters())-sum(p.numel()for p in h.geometry.parameters())
        assert extra==2*h.spec.dimension*h.width+15*h.width+h.spec.dimension
    family=width_family(*ledgers);assert len(set(family.total_numel))==3
    assert all(h.readout.out_features==2 for h in heads)


def test_config_and_tensor_checkpoint_rejection_precedes_any_copy():
    h=make();state_dict=copy.deepcopy(h.state_dict());before={n:p.detach().clone()for n,p in h.named_parameters()}
    clone=copy.deepcopy(h);clone.load_state_dict(state_dict)
    r,t=problem();x=state();ts=torch.tensor([.2,.8],dtype=x.dtype)
    assert torch.equal(h(r,t,x,ts,skeleton()),clone(r,t,x,ts,skeleton()))
    for kind in('sidecar','late-shape','alias'):
        broken=copy.deepcopy(state_dict)
        broken['readout.weight']=broken['readout.weight']+1
        if kind=='sidecar':broken['_extra_state']['spec']['coordinate_sha256']='e'*64
        elif kind=='late-shape':broken['readout.bias']=torch.zeros(3,dtype=torch.float64)
        else:broken['unrecognized']=torch.zeros(1)
        with pytest.raises(ValueError):h.load_state_dict(broken)
        assert all(torch.equal(before[n],p)for n,p in h.named_parameters())


def test_legacy_tropical_operator_is_shared_not_silently_replaced():
    h=make(legacy_tropical=True);r,t=problem();f=skeleton();x=state().requires_grad_()
    y=h(r,t,x,torch.tensor([.2,.4],dtype=x.dtype),f)
    y.square().sum().backward();assert x.grad is not None and torch.isfinite(x.grad).all()
    assert h.get_extra_state()['parameterization']=='legacy-no-mup-qualification'
    assert h.geometry.graph.layers[0].attention is h.geometry._source_backbone_reference[0].layers[0].attention
    assert 'Tropical' in type(h.geometry.graph.layers[0].attention).__name__
    with pytest.raises(ValueError,match='muP|mup'):
        forest_flow_parameter_roles(h,identifier_axis=tuple('g:'+str(i)for i in range(8)))


def test_readout_uses_selected_master_multiplier_once():
    h=make(16);r,t=problem();f=skeleton();x=state();time=torch.tensor([.2,.7],dtype=x.dtype)
    calls=[]
    hook=h.readout.register_forward_pre_hook(lambda m,args:calls.append(args[0].detach().clone()))
    result=h(r,t,x,time,f);hook.remove()
    assert len(calls)==1
    # Rebuild the complete graph computation independently from native primitives.
    rows=[];depth=[]
    statuses=('open','terminal','censored','invalid');directions=('root','computational','temporal','observational','interventional')
    for i,p in enumerate(f.parents):
        depth.append(0 if p==-1 else depth[p]+1)
        rows.append([depth[i]/h.spec.max_nodes,float(i in f.clamped)]+[float(f.statuses[i]==s)for s in statuses]+[float(f.directions[i]==s)for s in directions])
    y=x.clone();y[:,0]=f.states[0]
    inp=torch.nn.functional.silu((torch.nn.functional.linear(y,h.state_projection.weight,h.state_projection.bias)
        +torch.nn.functional.linear(time[:,None],h.time_projection.weight,h.time_projection.bias)[:,None]
        +torch.nn.functional.linear(x.new_tensor(rows),h.structure_projection.weight,h.structure_projection.bias)[None]
        +h.geometry(r,t)[:,None])/2)
    edges=tuple((p,i)for i,p in enumerate(f.parents)if p!=-1)
    features=torch.cat((inp,torch.stack([(inp[:,p]+inp[:,i])/2 for p,i in edges],1)),1)
    endpoint=torch.tensor([(i,i)for i in range(5)]+list(edges),dtype=torch.long)
    role=torch.tensor([0]*5+[1]*4,dtype=torch.long);ids=torch.eye(5,8,dtype=x.dtype)
    hidden=h.geometry.graph([dict(features=row,endpoints=endpoint,roles=role,identifiers=ids)for row in features])[0][:,1:6]
    expected_input=torch.nn.functional.silu(hidden)*(h.readout_mult/(h.width/h.base_width))
    assert torch.equal(calls[0],expected_input)
    expected=torch.nn.functional.linear(expected_input,h.readout.weight,h.readout.bias);expected[:,0]=0
    assert torch.equal(result,expected)


def test_fresh_process_tensor_checkpoint_and_sampler_rng_reconstruct_exactly(tmp_path):
    import subprocess,sys,os
    h=make();r,t=problem();f=skeleton();generator=torch.Generator().manual_seed(876)
    bound=h.bind_context(r,t)
    sample_flow_forests(f,bound,count=3,steps=2,generator=generator,field_sha256=bound.sha256)
    # Our own temporary untrained tensor state, not a loaded external pickle or a
    # substantive training checkpoint. Persist the next sampler state explicitly.
    torch.save(dict(model=h.state_dict(),generator=generator.get_state()),tmp_path/'state.pt')
    expected=sample_flow_forests(f,bound,count=3,steps=2,generator=generator,field_sha256=bound.sha256)
    script='''import sys,torch
from pathlib import Path
from test_forest_flow import make,problem,skeleton
from mora3.reasoning.flow_forest_refinement import sample_flow_forests
p=Path(sys.argv[1]);saved=torch.load(p/'state.pt',weights_only=True,map_location='cpu')
h=make();h.load_state_dict(saved['model']);r,t=problem();field=h.bind_context(r,t)
g=torch.Generator();g.set_state(saved['generator'])
result=sample_flow_forests(skeleton(),field,count=3,steps=2,generator=g,field_sha256=field.sha256)
torch.save(dict(states=torch.stack([f.states for f in result['forests']]),generator=result['generator_after'],field=field.sha256),p/'replay.pt')
'''
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    child=subprocess.run([sys.executable,'-B','-c',script,str(tmp_path)],env=env,capture_output=True,text=True,timeout=45)
    assert child.returncode==0,child.stderr
    actual=torch.load(tmp_path/'replay.pt',weights_only=True,map_location='cpu')
    assert torch.equal(actual['states'],torch.stack([f.states for f in expected['forests']]))
    assert torch.equal(actual['generator'],expected['generator_after'])and actual['field']==bound.sha256


def test_actual_neural_field_reaches_existing_multi_output_controller():
    from mora3.reasoning.flow_forest_refinement import run_flow_seeded_refinement
    from test_flow_forest_refinement import policy,assessment,decoder,decoded_assessment
    h=make();r,t=problem();field=h.bind_context(r,t)
    # Scorer/decoder remain labeled software fixtures; only the proposal field is
    # the actual new untrained neural head, not an analytic velocity callback.
    p=policy(edit_rounds=0,minimum_score=0.,minimum_decoded_score=0.,minimum_distance=0.)
    out=run_flow_seeded_refinement(skeleton(),field,steps=2,generator=torch.Generator().manual_seed(99),policy=p,
        assess=lambda f:replace(assessment(f,1.),context_sha256=f.context_sha256),
        edit=lambda *args,**kwargs:(),decode=lambda f:replace(decoder(f),context_sha256=f.context_sha256),
        assess_decoded=lambda f,d:replace(decoded_assessment(f,d,1.),context_sha256=f.context_sha256),field_sha256=field.sha256,
        evaluator_sha256='c'*64,decoder_sha256='d'*64,decoded_evaluator_sha256='e'*64)
    assert len(out['initial_candidate_ids'])==p.initial_candidates
    assert len(out['outputs'])==p.return_outputs and out['gfn_on_policy']is False


def test_native_subtree_editor_uses_actual_field_and_preserves_siblings():
    from mora3.model.forest_flow import NativeForestSubtreeEditor
    from mora3.reasoning.flow_forest_refinement import ForestAssessment,forest_fingerprint,_checked_revision
    h=make();r,t=problem();f=skeleton();field=h.bind_context(r,t)
    a=ForestAssessment(forest_fingerprint(f),f.context_sha256,'c'*64,'pass',.5,(1.,.1,.4,.8,.9),'d'*64,True)
    gen=torch.Generator().manual_seed(123);global_rng=torch.get_rng_state().clone();parent=f.states.clone()
    editor=NativeForestSubtreeEditor(field,steps=3,generator=gen)
    result=editor(f,a,count=2,reason='correct',feedback=None)
    assert len(result)==2 and all(p.cut_node_id=='left'for p in result)
    for proposal in result:
        _checked_revision(f,proposal)
        assert torch.equal(proposal.forest.states[[0,3,4]],parent[[0,3,4]])
    assert not torch.equal(result[0].forest.states[1:3],result[1].forest.states[1:3])
    assert torch.equal(parent,f.states)and torch.equal(global_rng,torch.get_rng_state())
    assert editor.last_receipt['velocity_evaluations']==3
    assert editor.last_receipt['generated_preserved_nodes_are_observations']is False
    replay=NativeForestSubtreeEditor(field,steps=3,generator=torch.Generator().manual_seed(123))(f,a,count=2,reason='correct',feedback=None)
    assert all(torch.equal(x.forest.states,y.forest.states)for x,y in zip(result,replay))


def test_editor_invalid_assessment_or_stale_weights_does_not_advance_rng():
    from mora3.model.forest_flow import NativeForestSubtreeEditor
    from mora3.reasoning.flow_forest_refinement import ForestAssessment,forest_fingerprint
    h=make();r,t=problem();f=skeleton();field=h.bind_context(r,t);g=torch.Generator().manual_seed(3)
    edit=NativeForestSubtreeEditor(field,steps=2,generator=g)
    a=ForestAssessment(forest_fingerprint(f),f.context_sha256,'c'*64,'pass',.5,(1.,.1,.4,.8,.9),'d'*64)
    before=g.get_state().clone()
    with pytest.raises(ValueError):edit(f,replace(a,candidate_sha256='0'*64),count=2,reason='correct',feedback=None)
    assert torch.equal(before,g.get_state())
    with torch.no_grad():h.readout.bias.add_(1)
    with pytest.raises(ValueError):edit(f,a,count=2,reason='correct',feedback=None)
    assert torch.equal(before,g.get_state())
