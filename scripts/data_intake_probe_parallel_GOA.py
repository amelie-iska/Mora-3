"""Closed-manifest query census and complete corpus-reader probe of the new null-code batch."""
from pathlib import Path
import collections,csv,datetime,gzip,hashlib,json,sys
import pyarrow as pa
import pyarrow.parquet as pq
R=Path.cwd();sys.path.insert(0,str(R/'scripts'))
from data_intake_goa_manifest_discovery import candidate_source_manifests
from data_intake_curate_goa_snapshot_v3 import completed_manifest
from data_intake_goa_projection import annotation_projection
from data_sources_goa_v2 import crosscheck_exports
B=Path('outputs/data-intake/20261101-GOA-parallel-manifest-readiness-v1');B.mkdir(exist_ok=False);S=Path('outputs/data-sources/20261031-goa-companion-v1')
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def check(e):
 p=Path(e['path']);b=p.read_bytes();assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'];return p
with gzip.open('outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz','rt') as f:baseline={r['Entry'] for r in csv.DictReader(f,delimiter='\t')}
queries=set();inputs=[];available=0;parallel=0
for p in candidate_source_manifests(S):
 m=completed_manifest(json.loads(p.read_text()));assert m['source_snapshot']=='2026-07-28 10:08' and m['full_batch_scan'] and m['license']=='CC-BY-4.0';q=set(m['query_canonical_accessions']);assert len(q)==len(m['query_canonical_accessions']) and q<=baseline and not q&queries;queries.update(q);inputs.append(ref(p));available+=m['available_annotations'];parallel+=p.name.startswith('manifest-quickgo-parallel-v2-annotation-')
p=S/'curated/manifest-quickgo-parallel-v2-annotation-0-batch-0000.json';m=json.loads(p.read_text());annotations=[];gaf=None;gpad=None
for e in m['all_raw_payloads']:
 p=check(e)
 if p.name.endswith('complete.gaf'):gaf=p.read_text()
 elif p.name.endswith('complete.gpad'):gpad=p.read_text()
 elif (p.name.startswith('page-') or '-json-page-' in p.name) and p.suffix=='.json':annotations+=json.loads(p.read_text())['results']
assert len(annotations)==2985 and gaf is not None and gpad is not None;proof=crosscheck_exports(annotations,m['query_canonical_accessions'],gaf,gpad);assert proof['GPAD_full_native_semantic_multiset_equal'] and proof['GAF_unrepresentable_native_rows']==1;projections=[annotation_projection(x) for x in annotations];missing=[x for x in projections if x['GO_evidence'] is None];assert len(missing)==1 and missing[0]['ECO_evidence']=='ECO:0000312' and missing[0]['evidence_class']=='source_inferred_or_other_evidence';assert all(x['missing_is_negative'] is False for x in projections)
rows=[dict(annotation_id=p['source_annotation_id'],projection_json=json.dumps(p,sort_keys=True)) for p in projections];q=B/'new-manifest-complete-assertion-probe.parquet';pq.write_table(pa.Table.from_pylist(rows),q,compression='zstd');assert pq.read_table(q).to_pylist()==rows
result=dict(schema='mora3.GOAParallelCorpusReaderReadiness/v1',status='passed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_source_manifest_metadata_count=len(inputs),parallel_manifest_count=parallel,closed_source_query_count=len(queries),closed_source_available_annotation_count=available,exact_query_union_subset_of_107778_baseline=True,duplicate_canonical_queries=0,remaining_baseline_queries=sorted(baseline-queries),full_scope=len(queries)==107778,input_manifests=inputs,census_quality_level='manifest-and-query-level-only;notnewfullrawQCofallbatches',complete_actual_newstyle_probe_annotations=len(annotations),newstyle_probe_full_native_GPAD_and_GAF_representable_proof=proof,source_null_shortcodes_preserved=True,null_ECO0000312_not_invented_as_experimental_code=True,projection=ref(q),new_task_admissions=0,new_corpus_task_shards=0,new_model_heads=0,next_complete_curation_command='/home/iska/miniconda3/envs/tokengt/bin/python scripts/data_intake_curate_goa_snapshot_v3.py --output-name GOA-typed-complete-v5 --max-batches 10000',next_command_gate='Fullsourcequeryunionmustbeverified;do notclaimfull107778completionfromthispartialcensus',producer_code=ref(Path('scripts/data_intake_curate_goa_snapshot_v3.py')),discovery_code=ref(Path('scripts/data_intake_goa_manifest_discovery.py')))
(B/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');print('PASS',len(inputs),parallel,len(queries),available,len(annotations))
