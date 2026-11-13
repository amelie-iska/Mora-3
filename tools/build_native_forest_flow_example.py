"""Build an explicitly untrained neural fixture and bounded CPU component profile.

No optimizer, source-corpus admission, weights download or training run. The
fixture's biological inputs are fabricated scalar test records, not experiments.
"""
from pathlib import Path
from dataclasses import replace
import hashlib,json,time,statistics,resource
import torch
from mora3.data.schema import GraphRecord,Node,Edge,Measurement
from mora3.data.tasks import ConditionalTask
from mora3.data.provenance import Provenance
from mora3.model.tokengt import TokenGT
from mora3.model.geometry_conditioning import GeometryConditioner
from mora3.model.forest_flow import ForestFlowSpec,NativeForestFlowHead,forest_context_sha256
from mora3.reasoning.flow_forest_refinement import ForestState,sample_flow_forests


def build(root):
    torch.set_num_threads(1);torch.manual_seed(20261113)
    backbone=TokenGT(16,2,width=16,heads=2,layers=2,ffn_width=16,
        scaling_policy='mup-v1',initialization_policy='mup-transformer-v1',
        parameterization_policy='mup-tokengt-v1',base_width=8,base_ffn_width=8)
    geometry=GeometryConditioner(backbone,id_width=8,parameterization_policy='mup-geometry-v1',base_width=8)
    spec=ForestFlowSpec(2,'a'*64,'b'*64,max_nodes=8,max_batch_size=8,max_attention_elements=100000)
    model=NativeForestFlowHead(geometry,spec).double().eval()
    task=ConditionalTask(((('node',0,'condition'),'condition'),(('node',0,'target'),'target')),topology_known=True)
    def problem(condition=1.,target=7.):
        return GraphRecord('synthetic-field-illustration',(Node('test-context',(
            Measurement('condition',condition,'dimensionless','observed',True),
            Measurement('target',target,'dimensionless','observed',True))),),(),
            (Provenance('synthetic-software-fixture','v1','field-illustration','c'*64,'test-only'),))
    state=torch.tensor([[[.2,.4],[-.7,.9],[-.3,.2],[.6,-.8],[.1,-.5]]],dtype=torch.float64)
    rows=[];bound=None;template=None
    for condition in(1.,2.):
        record=problem(condition);hidden=problem(condition,999999.)
        template=ForestState(('condition','left','left-terminal','right','right-terminal'),
            (-1,0,1,0,3),('open','open','terminal','open','terminal'),state[0].clone(),(0,),
            ('root','computational','computational','temporal','temporal'),
            forest_context_sha256(record,task,source_snapshot_sha256=spec.source_snapshot_sha256),spec.coordinate_sha256)
        bound=model.bind_context(record,task)
        for time_value in(0.,.25,.5,.75,1.):
            ts=torch.tensor([time_value],dtype=state.dtype)
            with torch.no_grad():
                value=model(record,task,state,ts,template);excluded=model(hidden,task,state,ts,template)
            assert torch.equal(value,excluded)and torch.count_nonzero(value[:,0])==0
            rows.append(dict(condition=condition,time=time_value,states=state[0].tolist(),
                velocity=value[0].tolist(),hidden_target_max_difference=float((value-excluded).abs().max()),
                context_sha256=template.context_sha256,field_sha256=bound.sha256))
    samples=sample_flow_forests(template,bound,count=4,steps=4,generator=torch.Generator().manual_seed(17),field_sha256=bound.sha256)
    timings=[]
    with torch.no_grad():
        for i in range(9):
            start=time.perf_counter();bound(state,torch.tensor([.5],dtype=state.dtype),template);elapsed=time.perf_counter()-start
            if i>=2:timings.append(elapsed)
    from mora3.model.annotation_vocabulary import AnnotationVocabulary
    from mora3.tokenization.forest_tokens import ForestTokenAxis
    from mora3.model.forest_token_decoder import ForestTokenCodebook,ForestDecoderSpec,NativeForestTokenDecoder
    from mora3.model.forest_quality import ForestQualitySpec,NativeForestQualityHead
    from mora3.model.forest_analogy import ForestAnalogySpec,NativeForestAnalogyHead,ForestAnalogyItem,ForestAnalogyIndex
    axis=ForestTokenAxis.from_annotation(AnnotationVocabulary(json.dumps(dict(version=1,
        learned_hex=[b'abc'.hex()],controls=['pad','node','edge'],certificate={'synthetic_fixture':True}))))
    book=ForestTokenCodebook(axis,torch.randn(axis.tokenizer().size,3,generator=torch.Generator().manual_seed(446),dtype=state.dtype),
        spec.source_snapshot_sha256,spec.coordinate_sha256,'c'*64,'d'*64,'e'*64)
    decoder=NativeForestTokenDecoder(model,book,ForestDecoderSpec(4,max_score_elements=1000000)).eval()
    quality=NativeForestQualityHead(model,ForestQualitySpec(('fixture-criterion-A','fixture-criterion-B'),'d'*64,(1.,1.))).eval()
    analogy=NativeForestAnalogyHead(model,ForestAnalogySpec(4,'e'*64,max_memory=16,max_centers=4)).eval()
    with torch.no_grad():
        embedding=decoder.token_embeddings(problem(2.),task,state,template)[0];tokenizer=axis.tokenizer()
        nearest=[]
        for node in range(1,5):
            for slot in range(4):
                allowed=torch.tensor(decoder.allowed_token_ids);dist=torch.linalg.vector_norm(decoder.codebook[allowed]-embedding[node,slot],dim=1)
                ids=torch.argsort(dist,stable=True)[:6]
                nearest.append(dict(node=node,slot=slot,dimension=3,candidates=[dict(token=int(allowed[i]),distance=float(dist[i]),
                    label='NODE_END'if int(allowed[i])==decoder.stop_token else repr(tokenizer.decode((int(allowed[i]),))))for i in ids]))
        quality_values=quality(problem(2.),task,torch.stack([f.states for f in samples['forests']]),template)
        quality_rows=[dict(candidate=i,criteria=row.tolist(),ranking_score=float(quality_values['ranking_score'][i]))
            for i,row in enumerate(quality_values['criterion_logits'].sigmoid())]
        memories=tuple(ForestAnalogyItem(str(i),problem(2.),task,f,('memory:'+str(i),),'train','e'*64,'f'*64)
            for i,f in enumerate(samples['forests']))
        index=ForestAnalogyIndex(analogy,memories,centers=4)
        query=ForestAnalogyItem('query',problem(1.),task,None,('memory:0',),'validation','e'*64,'f'*64)
        exact=index.search(query,k=3,exact=True);approx=index.search(query,k=3,probes=1)
    fixture=dict(schema='Mora3.UntrainedNativeForestFieldFixture/v1',synthetic=True,trained=False,
        seed=20261113,width=16,base_width=8,dimension=2,nodes=5,attention='actual shared vanilla TokenGT',
        unique_component_parameters=sum(p.numel()for p in model.parameters()),
        new_parameters_excluding_shared_geometry=sum(p.numel()for p in model.parameters())-sum(p.numel()for p in geometry.parameters()),
        parameters_are_component_not_full80M_count=True,rows=rows,nearest_token_examples=nearest,
        control_count=len(axis.controls),vocabulary_size=axis.tokenizer().size,
        quality_examples=quality_rows,quality_criteria=quality.spec.criteria,
        analogy_example=dict(exact=exact,approximate=approx,excluded_shared_group='memory:0',
            semantic_correctness_measured=False),
        candidates=[f.states.tolist()for f in samples['forests']],solver_steps=4,batched_candidates=4,
        field_evaluations=4,native_transformer_invocations=8,
        cpu_forward_profile=dict(threads=1,dtype='float64',batch=1,nodes=5,repetitions=7,median_seconds=statistics.median(timings),
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        source_sha256=hashlib.sha256((root/'src/mora3/model/forest_flow.py').read_bytes()).hexdigest(),
        component_source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()for name in(
            'src/mora3/model/forest_flow.py','src/mora3/model/forest_analogy.py',
            'src/mora3/model/forest_token_decoder.py','src/mora3/model/forest_quality.py',
            'src/mora3/tokenization/forest_tokens.py','src/mora3/reasoning/flow_forest_refinement.py')},
        optimizer_updates=0,training_runs=0,heldout_performance=False,
        caveat='Untrained neural component on fabricated scalar inputs; no biological accuracy or whole-model speed claim')
    out=root/'tests/project/fixtures/native-forest-flow-example.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(fixture,indent=2)+'\n')
    print(json.dumps(dict(path=str(out.relative_to(root)),rows=len(rows),component_parameters=fixture['unique_component_parameters'],
        added_parameters=fixture['new_parameters_excluding_shared_geometry'],profile=fixture['cpu_forward_profile']),indent=2))


if __name__=='__main__':build(Path(__file__).resolve().parents[1])
