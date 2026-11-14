"""Source compiler tests; all train examples are explicitly synthetic, zero updates."""
from dataclasses import replace
from hashlib import sha256
import copy
import json
from pathlib import Path
import pytest
import torch
from test_forest_flow import make, SOURCE
from test_forest_tokens import base
from mora3.model.annotation_vocabulary import AnnotationPayloadEncoder
from mora3.tokenization.forest_tokens import ForestTokenAxis
from mora3.reasoning.trajectory_contracts import ReasoningTrace, TraceNode, TraceStep, TraceFact
from mora3.data.forest_source_exports import FrozenForestSourceEncoder, ForestSourceExport, compile_source_forest


def trace(split='train'):
    def fact(name, value, role, origin=None, deps=()):
        return TraceFact(name, value, role, origin, deps, 'synthetic_fixture', sha256(value.encode()).hexdigest())
    facts=(fact('condition','2.0','condition'),fact('left','3.00','derived','left',('condition',)),
           fact('right','4e0','derived','right',('condition',)),fact('hidden','999','evaluation'))
    nodes=(TraceNode('root',0,('condition',),'open'),
           TraceNode('left',1,('condition','left'),'terminal'),
           TraceNode('right',1,('condition','right'),'terminal'))
    steps=(TraceStep('L','root','left','compute-left','computational',('condition',),'c'*64),
           TraceStep('R','root','right','compute-right','temporal',('condition',),'d'*64))
    return ReasoningTrace('source-record-7','e'*64,'f'*64,'a'*64,split,('source-group-7',),nodes,steps,facts,())


def encoder():
    geometry=make().geometry
    axis=ForestTokenAxis.from_annotation(base())
    payload=AnnotationPayloadEncoder(axis.annotation_vocabulary(),8).double()
    return FrozenForestSourceEncoder(geometry,payload,axis,source_snapshot_sha256=SOURCE,
        encoder_evidence_sha256='9'*64,encoder_status='untrained-diagnostic')


def exported(e=None,t=None,**kwargs):
    return compile_source_forest(e or encoder(),t or trace(),source_revision='synthetic-v1',
        original_ordinal=7,source_evidence_sha256='8'*64,**kwargs)


def test_all_branches_exact_source_bytes_and_frozen_native_prefix_values():
    e=encoder();t=trace();x=exported(e,t)
    d=x.document();assert d['split']=='train' and d['evidence_kind']=='synthetic_fixture'
    assert d['node_ids']==['root','left','right'] and d['parents']==[-1,0,0]
    assert d['directions']==['root','computational','temporal']
    assert d['original_ordinal']==7 and d['source_revision']=='synthetic-v1'
    assert d['source_trace']['facts'][1]['value_json']=='3.00'
    assert d['source_trace']['facts'][2]['value_json']=='4e0'
    assert [f['name']for f in d['source_trace']['facts']][-1]=='hidden'
    payloads=x.payloads();assert all(b'hidden'not in p for p in payloads)
    assert b'right'not in payloads[1] and b'left'not in payloads[2]
    from mora3.reasoning.trajectory_contracts import prefix_native_graph
    expected=[]
    for path in ((),('L',),('R',)):
        r,task,_=prefix_native_graph(t,path)
        expected.append(e.geometry(r,task).detach()[0])
    torch.testing.assert_close(x.latents(),torch.stack(expected),atol=0,rtol=0)
    assert d['training_admission']is False and not x.latents().requires_grad


def test_future_sibling_and_evaluation_values_cannot_change_earlier_exports():
    e=encoder();t=trace();x=exported(e,t)
    other=replace(t,facts=(t.facts[0],t.facts[1],replace(t.facts[2],value_json='1000'),replace(t.facts[3],value_json='0')))
    y=exported(e,other)
    assert torch.equal(x.latents()[:2],y.latents()[:2])
    assert x.document()['context_sha256']==y.document()['context_sha256']
    assert x.sha256!=y.sha256
    assert not torch.equal(x.latents()[2],y.latents()[2])


def test_source_encoder_snapshots_owner_without_changing_modes_rng_or_alias_values():
    g=make().geometry;axis=ForestTokenAxis.from_annotation(base())
    p=AnnotationPayloadEncoder(axis.annotation_vocabulary(),8).double();g.train();p.train()
    before=[v.detach().clone()for v in (*g.parameters(),*p.parameters())];rng=torch.get_rng_state().clone()
    e=FrozenForestSourceEncoder(g,p,axis,source_snapshot_sha256=SOURCE,encoder_evidence_sha256='9'*64,encoder_status='untrained-diagnostic')
    assert g.training and p.training and all(v.requires_grad for v in (*g.parameters(),*p.parameters()))
    assert all(not m.training for root in(e.geometry,e.payload)for m in root.modules())
    assert all(not v.requires_grad for root in(e.geometry,e.payload)for v in root.parameters())
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(a,b)for a,b in zip(before,(*g.parameters(),*p.parameters())))
    assert {v.data_ptr()for v in g.parameters()}.isdisjoint(v.data_ptr()for v in e.geometry.parameters())


def test_exact_extended_embedding_rows_not_old_metadata_or_random_codebook():
    e=encoder();book=e.codebook();n=e.axis.tokenizer().size
    assert torch.equal(book.values,e.payload.embedding.weight[:n])
    assert book.encoder_sha256==e.encoder_sha256 and book.coordinate_sha256==e.coordinate_sha256
    assert book.axis.sha256==e.axis.sha256 and book.values.requires_grad is False
    old=AnnotationPayloadEncoder(base(),8).double()
    with pytest.raises(ValueError,match='extended'):
        FrozenForestSourceEncoder(e.geometry,old,e.axis,source_snapshot_sha256=SOURCE,encoder_evidence_sha256='9'*64,encoder_status='untrained-diagnostic')


@pytest.mark.parametrize('fault',('weights','bufferless-config','mode','grad','axis'))
def test_encoder_changes_refuse_before_new_neural_evaluation(fault):
    e=encoder()
    if fault=='weights':e.geometry.graph.feature.weight.data.add_(1)
    if fault=='bufferless-config':e.geometry.payload.max_tokens+=1
    if fault=='mode':e.geometry.train()
    if fault=='grad':e.payload.embedding.weight.requires_grad_(True)
    if fault=='axis':
        from mora3.model.annotation_vocabulary import AnnotationVocabulary
        data=json.loads(e.axis.base_specification_json);data['different_source']='test-only'
        e.axis=ForestTokenAxis.from_annotation(AnnotationVocabulary(json.dumps(data)))
    calls=[];handle=e.geometry.register_forward_pre_hook(lambda *a:calls.append(True))
    try:
        with pytest.raises(ValueError):exported(e)
        assert not calls
    finally:handle.remove()


def test_safe_disk_roundtrip_and_mutable_readbacks_do_not_change_artifact(tmp_path):
    e=encoder();x=exported(e);path=tmp_path/'source.json';x.save(path)
    y=ForestSourceExport.load(path,expected_sha256=x.sha256)
    assert y.raw==x.raw and y.payloads()==x.payloads()
    assert torch.equal(y.latents(),x.latents())
    y.latents().fill_(123);y.document()['split']='test'
    assert torch.equal(y.latents(),x.latents()) and y.document()['split']=='train'
    with pytest.raises(FileExistsError):x.save(path)
    path.write_bytes(x.raw+b' ')
    with pytest.raises(ValueError):ForestSourceExport.load(path,expected_sha256=x.sha256)
    link=tmp_path/'link.json';link.symlink_to(path)
    with pytest.raises(ValueError):ForestSourceExport.load(link,expected_sha256=x.sha256)


def heads_for(e):
    from mora3.model.forest_flow import NativeForestFlowHead,ForestFlowSpec
    from mora3.model.forest_token_decoder import NativeForestTokenDecoder,ForestDecoderSpec
    from mora3.model.forest_quality import NativeForestQualityHead,ForestQualitySpec
    from mora3.model.forest_analogy import NativeForestAnalogyHead,ForestAnalogySpec
    from mora3.model.forest_training import ForestTrainingHeads,ForestTrainingSpec
    flow=NativeForestFlowHead(make().geometry,ForestFlowSpec(8,SOURCE,e.coordinate_sha256,max_nodes=8)).double()
    decoder=NativeForestTokenDecoder(flow,e.codebook(),ForestDecoderSpec(2048,max_score_elements=100000))
    quality=NativeForestQualityHead(flow,ForestQualitySpec(('source-supported',),'c'*64,(1.,)))
    analogy=NativeForestAnalogyHead(flow,ForestAnalogySpec(8,'a'*64))
    h=ForestTrainingHeads(flow,decoder,quality,analogy)
    spec=ForestTrainingSpec(SOURCE,e.coordinate_sha256,e.encoder_sha256,'a'*64,e.skeleton_policy_sha256,h.sha256,(1.,1.,1.,1.))
    return h,spec


def test_actual_export_compiles_into_existing_route_and_four_head_loss_with_gradients():
    from mora3.model.forest_training import forest_joint_loss
    from mora3.research.forest_training_routes import ForestTrainingRoute
    e=encoder();x=exported(e);h,s=heads_for(e);batch=x.training_batch(h,s)
    assert batch.quality is None and batch.analogy is None
    assert batch.identity.original_ordinals==(7,) and batch.identity.export_sha256==x.sha256
    assert not batch.flow.available[:,0].any() and batch.flow.available[:,1:].all()
    assert torch.equal(batch.template.states[1:],torch.zeros_like(batch.template.states[1:]))
    route=ForestTrainingRoute(h,s,(batch,),task_key='forest',trial_id='synthetic-test-only')
    selected=route.select(h,route.initial_cursor(),max_windows=1)
    assert selected['original_source_rows']==1 and selected['status']=='prepared'
    noise=torch.randn(batch.flow.values.shape,dtype=torch.float64,generator=torch.Generator().manual_seed(101))
    result=forest_joint_loss(h,s,batch,noise=noise,time=torch.tensor([.4],dtype=torch.float64))
    assert set(result['component_losses'])=={'flow','tokens'} and result['counts']['quality']==result['counts']['analogy']==0
    result['loss'].backward()
    assert h.flow.readout.weight.grad.abs().sum()>0 and h.decoder.readout.weight.grad.abs().sum()>0
    assert all(v.grad is None for root in(e.geometry,e.payload)for v in root.parameters())


@pytest.mark.parametrize('split',('validation','test'))
def test_heldout_exports_cannot_be_training_batches(split):
    e=encoder();x=exported(e,trace(split));h,s=heads_for(e)
    with pytest.raises(ValueError,match='train'):x.training_batch(h,s)


@pytest.mark.parametrize('fault',('censored','node-budget','token-budget','byte-budget','ordinal-bool'))
def test_capacity_and_incomplete_traces_refuse_before_forward(fault):
    e=encoder();t=trace();kwargs={}
    if fault=='censored':t=replace(t,nodes=(*t.nodes[:2],replace(t.nodes[2],status='censored')))
    if fault=='node-budget':kwargs['max_nodes']=2
    if fault=='token-budget':kwargs['max_total_prefix_tokens']=1
    if fault=='byte-budget':kwargs['max_export_bytes']=1
    calls=[];handle=e.geometry.register_forward_pre_hook(lambda *a:calls.append(True))
    try:
        with pytest.raises(ValueError):
            if fault=='ordinal-bool':compile_source_forest(e,t,source_revision='s',original_ordinal=True,source_evidence_sha256='8'*64)
            else:exported(e,t,**kwargs)
        assert not calls
    finally:handle.remove()


def test_real_original_grh_trace_stays_computed_evaluation_only(tmp_path):
    from mora3.data.grh_research_traces import load_pinned_trace,as_reasoning_trace
    root=Path(__file__).resolve().parents[1]/'fixtures/grh_research'
    source=load_pinned_trace(root,'photonic-0');t=as_reasoning_trace(source)
    e=encoder();x=compile_source_forest(e,t,source_revision=source.revision,original_ordinal=0,source_evidence_sha256=source.sha256)
    d=x.document();assert d['split']=='test' and d['evidence_kind']=='computed'
    assert len(d['node_ids'])==4 and d['source_trace']['source_sha256']==source.sha256
    assert len(d['source_trace']['facts'])==len(source.document()['nodes'])+1
    assert d['encoder']['encoder_status']=='untrained-diagnostic'
    x.save(tmp_path/'actual-reference.json')
    assert ForestSourceExport.load(tmp_path/'actual-reference.json',expected_sha256=x.sha256).raw==x.raw
    h,s=heads_for(e)
    with pytest.raises(ValueError,match='train'):x.training_batch(h,s)


def test_same_size_mutated_native_payload_tokenizer_is_not_the_committed_axis():
    from mora3.tokenization.shortest_path import ByteTokenizer
    e=encoder();old=e.payload.codec.tokenizer
    e.payload.codec.tokenizer=ByteTokenizer((b'xyz',b'12'),controls=e.axis.base_controls+e.axis.controls)
    assert e.payload.codec.tokenizer.size==old.size
    with pytest.raises(ValueError,match='tokenizer|axis'):e.codebook()


@pytest.mark.parametrize('fault',('extra-field','missing-field','boolean-counter','zero-counter',
    'encoder-commitment','coordinate-commitment','codebook-schema','codebook-dtype','codebook-export','nonfinite-latent'))
def test_rehashed_malformed_export_refuses_complete_schema_and_internal_bindings(fault):
    x=exported();d=x.document()
    if fault=='extra-field':d['unregistered_override']=True
    if fault=='missing-field':d.pop('statuses')
    if fault=='boolean-counter':d['total_native_prefix_tokens']=True
    if fault=='zero-counter':d['total_native_prefix_tokens']=0
    if fault=='encoder-commitment':
        d['encoder']['encoder_sha256']='0'*64;d['codebook']['encoder_sha256']='0'*64
    if fault=='coordinate-commitment':
        d['encoder']['coordinate_sha256']='0'*64;d['codebook']['coordinate_sha256']='0'*64
    if fault=='codebook-schema':d['codebook']['schema']='different'
    if fault=='codebook-dtype':d['codebook']['dtype']='torch.float32'
    if fault=='codebook-export':d['codebook']['export_sha256']='0'*64
    if fault=='nonfinite-latent':d['latent_hex']='000000000000f87f'+d['latent_hex'][16:]
    raw=json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    with pytest.raises(ValueError):ForestSourceExport(raw,sha256(raw).hexdigest())


def test_equivalence_merge_is_not_silently_duplicated_or_dropped():
    t=trace();nodes=(t.nodes[0],replace(t.nodes[1],status='open'),replace(t.nodes[2],status='open'),
        TraceNode('merge',2,('condition',),'terminal','2'*64))
    steps=t.steps+(TraceStep('LM','left','merge','merge','computational',('condition',),'3'*64),
        TraceStep('RM','right','merge','merge','computational',('condition',),'4'*64))
    merged=replace(t,nodes=nodes,steps=steps)
    with pytest.raises(ValueError,match='Merged'):exported(encoder(),merged)


def test_native_complete_prefix_export_is_deterministic_without_rng_changes():
    e=encoder();rng=torch.get_rng_state().clone();first=exported(e);second=exported(e)
    assert first.raw==second.raw and first.sha256==second.sha256
    assert torch.equal(rng,torch.get_rng_state())


def test_source_codebook_or_dtype_mismatch_cannot_be_reinterpreted_by_consumer():
    e=encoder();x=exported(e);h,s=heads_for(e)
    with pytest.raises(ValueError):x.training_batch(h,replace(s,encoder_sha256='0'*64))
    with pytest.raises(ValueError):x.training_batch(h,replace(s,coordinate_sha256='0'*64))
    with pytest.raises(ValueError):x.training_batch(h,replace(s,group_policy_sha256='0'*64))


def test_export_readback_in_a_fresh_process(tmp_path):
    import os,subprocess,sys
    x=exported();p=tmp_path/'forest.json';x.save(p)
    code='''import json,sys
from mora3.data.forest_source_exports import ForestSourceExport
x=ForestSourceExport.load(sys.argv[1],expected_sha256=sys.argv[2]);d=x.document()
print(json.dumps(dict(sha256=x.sha256,shape=list(x.latents().shape),split=d['split'],ordinal=d['original_ordinal'],training_admission=d['training_admission'])))
'''
    child=subprocess.run([sys.executable,'-c',code,str(p),x.sha256],text=True,capture_output=True,check=True,timeout=30,env=os.environ.copy())
    assert json.loads(child.stdout)==dict(sha256=x.sha256,shape=[3,8],split='train',ordinal=7,training_admission=False)


def test_export_encoder_commitment_includes_implementation_and_runtime_versions():
    import mora3.data.forest_source_exports as impl
    e=encoder();d=json.loads(e._frozen)
    assert d['runtime']['torch']==torch.__version__
    assert d['runtime']['byteorder']=='little'
    code=d['implementation_sha256']
    for name in ('mora3.data.forest_source_exports','mora3.model.geometry_conditioning',
                 'mora3.model.payload','mora3.model.tokengt','mora3.tokenization.shortest_path'):
        assert name in code and len(code[name])==64
    assert code[impl.__name__]==sha256(Path(impl.__file__).read_bytes()).hexdigest()


@pytest.mark.parametrize('dtype',(torch.float32,torch.float64))
def test_original_native_equations_from_compiled_source_match_joint_loss_and_gradients(dtype):
    from mora3.model.forest_flow import forest_flow_matching_loss
    from mora3.model.forest_token_decoder import forest_token_loss
    from mora3.model.forest_training import forest_joint_loss
    if dtype==torch.float64:e=encoder()
    else:
        g=make().geometry.float();axis=ForestTokenAxis.from_annotation(base())
        p=AnnotationPayloadEncoder(axis.annotation_vocabulary(),8).float()
        e=FrozenForestSourceEncoder(g,p,axis,source_snapshot_sha256=SOURCE,encoder_evidence_sha256='9'*64,encoder_status='untrained-diagnostic')
    # The common helper returns float64. Match dtype before constructing the frozen objective.
    if dtype==torch.float64:h,s=heads_for(e)
    else:
        from mora3.model.forest_flow import ForestFlowSpec,NativeForestFlowHead
        from mora3.model.forest_token_decoder import NativeForestTokenDecoder,ForestDecoderSpec
        from mora3.model.forest_quality import NativeForestQualityHead,ForestQualitySpec
        from mora3.model.forest_analogy import NativeForestAnalogyHead,ForestAnalogySpec
        from mora3.model.forest_training import ForestTrainingHeads,ForestTrainingSpec
        flow=NativeForestFlowHead(make().geometry.float(),ForestFlowSpec(8,SOURCE,e.coordinate_sha256,max_nodes=8))
        h=ForestTrainingHeads(flow,NativeForestTokenDecoder(flow,e.codebook(),ForestDecoderSpec(2048,max_score_elements=100000)),
            NativeForestQualityHead(flow,ForestQualitySpec(('source-supported',),'c'*64,(1.,))),
            NativeForestAnalogyHead(flow,ForestAnalogySpec(8,'a'*64)))
        s=ForestTrainingSpec(SOURCE,e.coordinate_sha256,e.encoder_sha256,'a'*64,e.skeleton_policy_sha256,h.sha256,(1.,1.,1.,1.))
    b=exported(e).training_batch(h,s);g=torch.Generator().manual_seed(910)
    noise=torch.randn(b.flow.values.shape,generator=g,dtype=dtype);time=torch.tensor([.7],dtype=dtype)
    joint=forest_joint_loss(h,s,b,noise=noise,time=time)['loss']
    direct=forest_flow_matching_loss(h.flow,b.record,b.task,b.template,b.flow.values,noise,time,b.flow.available)['loss']
    direct=direct+forest_token_loss(h.decoder,b.record,b.task,b.tokens.states,b.template,b.tokens.values,b.tokens.available)['loss']
    torch.testing.assert_close(joint,direct,atol=0,rtol=0)
    params=(h.flow.readout.weight,h.decoder.readout.weight,h.flow.geometry.graph.feature.weight)
    a=torch.autograd.grad(joint,params);other=torch.autograd.grad(direct,params)
    for x,y in zip(a,other):torch.testing.assert_close(x,y,atol=0,rtol=0)


def test_boolean_parent_reference_in_rehashed_wire_is_not_an_integer_parent():
    x=exported();d=x.document();d['parents'][1]=False
    raw=json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    with pytest.raises(ValueError):ForestSourceExport(raw,sha256(raw).hexdigest())


def test_frozen_codebook_safe_disk_export_reconstructs_exact_native_decoder_asset(tmp_path):
    from mora3.data.forest_source_exports import load_source_codebook
    e=encoder();path=tmp_path/'codebook.json';digest=e.save_codebook(path)
    book=load_source_codebook(path,expected_sha256=digest)
    assert book.binding()==e.codebook().binding()
    assert book.axis==e.axis and torch.equal(book.values,e.codebook().values)
    h,s=heads_for(e);x=exported(e)
    from mora3.model.forest_token_decoder import NativeForestTokenDecoder
    h=replace(h,decoder=NativeForestTokenDecoder(h.flow,book,h.decoder.spec))
    s=replace(s,heads_sha256=h.sha256)
    assert x.training_batch(h,s).identity.export_sha256==x.sha256
    with pytest.raises(FileExistsError):e.save_codebook(path)
    data=json.loads(path.read_bytes());data['values_hex']=format(int(data['values_hex'][:2],16)^1,'02x')+data['values_hex'][2:]
    raw=json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    other=tmp_path/'changed.json';other.write_bytes(raw)
    with pytest.raises(ValueError):load_source_codebook(other,expected_sha256=sha256(raw).hexdigest())


def test_source_codebook_load_in_fresh_process_requires_no_model_or_pickle(tmp_path):
    import os,subprocess,sys
    e=encoder();path=tmp_path/'codebook.json';digest=e.save_codebook(path)
    child=subprocess.run([sys.executable,'-c',
        'import sys,json;from mora3.data.forest_source_exports import load_source_codebook;print(json.dumps(load_source_codebook(sys.argv[1],expected_sha256=sys.argv[2]).binding(),sort_keys=True))',
        str(path),digest],check=True,capture_output=True,text=True,env=os.environ.copy(),timeout=30)
    assert json.loads(child.stdout)==e.codebook().binding()
