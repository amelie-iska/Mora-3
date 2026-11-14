"""Export retained real numerical references with an explicitly untrained encoder.

No optimizer, corpus admission, network access, repository code execution,
checkpoint download or dependency installation. Outputs are evaluation-only.
The three originals are user-owned public exports already pinned in this repo.
"""
from pathlib import Path
import argparse
from hashlib import sha256
import json
import resource
import time
import torch
from mora3.data.grh_research_traces import load_pinned_trace, as_reasoning_trace, REVISION, MANIFEST_SHA256
from mora3.data.forest_source_exports import FrozenForestSourceEncoder, ForestSourceExport, compile_source_forest, load_source_codebook
from mora3.model.tokengt import TokenGT
from mora3.model.geometry_conditioning import GeometryConditioner
from mora3.model.annotation_vocabulary import AnnotationVocabulary, AnnotationPayloadEncoder
from mora3.tokenization.forest_tokens import ForestTokenAxis


def wire(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def run(destination, *, payload_max_tokens=8192, max_total_prefix_tokens=2_000_000):
    if (type(payload_max_tokens) is not int or not 1 <= payload_max_tokens <= 262144
            or type(max_total_prefix_tokens) is not int or not 1 <= max_total_prefix_tokens <= 64_000_000):
        raise ValueError('Explicit bounded source payload and aggregate prefix capacity required')
    root=Path(__file__).resolve().parents[1]
    destination=Path(destination).absolute()
    if any(p.is_symlink()for p in(destination,*destination.parents)):
        raise ValueError('A fresh nonsymlink output directory is required')
    destination.mkdir(parents=False,exist_ok=False)
    recipe=dict(schema='Mora3.SourceForestDiagnosticRecipe/v1',seed=20261113,width=8,heads=2,
        layers=1,identifier_rank=64,dtype='float64',encoder_status='untrained-diagnostic',
        vocabulary='byte-only-unfitted-plus-forty-typed-controls',
        source_revision=REVISION,source_manifest_sha256=MANIFEST_SHA256,
        scope='real-computed-evaluation-source-with-untrained-native-encoder',optimizer_updates=0)
    if payload_max_tokens != 8192 or max_total_prefix_tokens != 2_000_000:
        recipe.update(schema='Mora3.SourceForestDiagnosticRecipe/v2',
            payload_max_tokens=payload_max_tokens,max_total_prefix_tokens=max_total_prefix_tokens,
            capacity_semantics='complete-per-entity-scalar-payload-and-total-prefix-work-not-attention-context')
    (destination/'encoder-recipe.json').write_bytes(wire(recipe))
    snapshot=sha256(wire(['GRH-reference-only-source-snapshot',REVISION,MANIFEST_SHA256])).hexdigest()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(recipe['seed'])
        backbone=TokenGT(8,2,width=8,heads=2,layers=1,ffn_width=8,
            scaling_policy='mup-v1',initialization_policy='mup-transformer-v1',
            parameterization_policy='mup-tokengt-v1',base_width=8,base_ffn_width=8)
        geometry=GeometryConditioner(backbone,id_width=64,parameterization_policy='mup-geometry-v1',base_width=8,
            payload_max_tokens=payload_max_tokens).double()
        vocabulary=AnnotationVocabulary(wire(dict(version=1,method='diagnostic-byte-only-unfitted',
            learned_hex=[],controls=['pad','node','edge'],sources=[],certificate={'fit_performed':False})).decode())
        axis=ForestTokenAxis.from_annotation(vocabulary)
        payload=AnnotationPayloadEncoder(axis.annotation_vocabulary(),8).double()
        encoder=FrozenForestSourceEncoder(geometry,payload,axis,source_snapshot_sha256=snapshot,
            encoder_evidence_sha256=sha256(wire(recipe)).hexdigest(),encoder_status='untrained-diagnostic')
    book_path=destination/'source-codebook.json';book_sha=encoder.save_codebook(book_path)
    assert load_source_codebook(book_path,expected_sha256=book_sha).binding()==encoder.codebook().binding()
    (destination/'encoder-configuration.json').write_bytes(encoder._frozen)
    rows=[];examples=[]
    source_root=root/'tests/fixtures/grh_research'
    for ordinal,name in enumerate(('photonic-0','tropical-attention','causal-soft-moe')):
        started=time.monotonic();source=load_pinned_trace(source_root,name);trace=as_reasoning_trace(source)
        row=dict(trace_id=name,original_sha256=source.sha256,source_revision=source.revision,
            original_dataflow_nodes=len(source.document()['nodes']),reasoning_states=len(trace.nodes),
            source_facts=len(trace.facts),source_split=trace.split,evidence_kind='computed',training_admission=False)
        try:
            result=compile_source_forest(encoder,trace,source_revision=source.revision,
                original_ordinal=ordinal,source_evidence_sha256=source.sha256,
                max_nodes=64,max_total_prefix_tokens=max_total_prefix_tokens)
        except ValueError as error:
            row.update(status='refused-without-truncation',reason=str(error))
        else:
            path=destination/(name+'.forest.json');result.save(path)
            replay=ForestSourceExport.load(path,expected_sha256=result.sha256)
            assert replay.raw==result.raw and torch.equal(replay.latents(),result.latents())
            document=result.document()
            row.update(status='exported-evaluation-only',path=path.name,sha256=result.sha256,
                bytes=len(result.raw),latent_shape=list(result.latents().shape),
                total_native_prefix_tokens=document['total_native_prefix_tokens'],exact_disk_readback=True)
            examples.append(dict(trace_id=name,source_sha256=source.sha256,export_sha256=result.sha256,
                node_ids=document['node_ids'],parents=document['parents'],directions=document['directions'],
                statuses=document['statuses'],latents=result.latents().tolist(),
                payloads=[json.loads(raw)for raw in result.payloads()],
                original_source_facts=[dict(name=f.name,origin_node=f.origin_node,role=f.role)for f in trace.facts]))
        row['wall_seconds']=time.monotonic()-started;rows.append(row)
        assert load_pinned_trace(source_root,name).sha256==source.sha256
    report=dict(schema='Mora3.RealSourceForestExportDiagnostic/v1',encoder=encoder.binding(),
        codebook_path=book_path.name,codebook_file_sha256=book_sha,rows=rows,
        original_files_unchanged=True,optimizer_updates=0,substantive_training_runs_started=0,
        new_training_examples_admitted=0,synthetic_source_records=False,
        encoder_trained=False,device='cpu',threads=1,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        resource_scope='whole-process-cumulative-not-isolated-model-memory',
        source_program_executed=False,complete_project=False)
    (destination/'verification.json').write_bytes(wire(report))
    (destination/'explorer-data.json').write_bytes(wire(dict(schema='Mora3.SourceForestExplorer/v1',
        diagnostic=True,trained=False,optimizer_updates=0,encoder=encoder.binding(),rows=rows,examples=examples)))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,help='New directory; existing paths are never overwritten.')
    parser.add_argument('--payload-max-tokens',type=int,default=8192)
    parser.add_argument('--max-total-prefix-tokens',type=int,default=2_000_000)
    args=parser.parse_args()
    print(json.dumps(run(args.output_dir,payload_max_tokens=args.payload_max_tokens,
        max_total_prefix_tokens=args.max_total_prefix_tokens),indent=2))
