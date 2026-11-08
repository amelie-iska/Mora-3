"""Exact separately versioned current GO reference binding; historical annotations unchanged."""
from pathlib import Path
import collections,datetime,hashlib,json,time
import pyarrow as pa
import pyarrow.parquet as pq
B=Path('outputs/data-intake/20261101-GO-ontology-reference-binding-v2');B.mkdir(exist_ok=False);start=time.perf_counter();S=Path('outputs/data-sources/20261031-goa-companion-v1/curated/quickgo-parallel-v2-ontology/manifest.json');m=json.loads(S.read_text());A=Path('outputs/data-intake/20261101-annotation-target-axes-v1/GO.parquet');axis=pq.read_table(A).to_pylist();rawrefs={r['source']['path']:r['source'] for r in m['source_pages']};cache=collections.OrderedDict();terms={};expected_edges=collections.Counter();actual_edges=collections.Counter();history=0;sourcefiles=0

def ref(p,**kw):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),**kw)
def check(e):
 p=Path(e['path']);assert p.stat().st_size==e['bytes'] and ref(p)['sha256']==e['sha256'];return p
try:
 for e in m['typed_payloads']:
  p=check(e)
  if e['kind']=='edges':
   for r in pq.read_table(p).to_pylist():actual_edges[(r['child_id'],r['relation'],r['parent_id'])]+=1
   continue
  assert e['kind']=='nodes'
  for r in pq.read_table(p).to_pylist():
   key=r['source_raw_path']
   if key not in cache:
    cache[key]=json.loads(check(rawrefs[key]).read_text())['results'];sourcefiles+=1
    if len(cache)>4:cache.popitem(last=False)
   raw=cache[key][r['source_result_index']];native=json.loads(r['native_term_json']);assert raw==native and r['go_id']==raw['id'];assert hashlib.sha256(r['native_term_json'].encode()).hexdigest()==r['native_term_sha256'];assert r['name']==raw['name'] and r['aspect']==raw.get('aspect') and r['is_obsolete']==raw.get('isObsolete',False);assert r['go_id'] not in terms;terms[r['go_id']]=r;history+=len(raw.get('history',[]))
   for child in raw.get('children',[]):expected_edges[(child['id'],child['relation'],raw['id'])]+=1
 assert len(terms)==48392 and expected_edges==actual_edges and sum(actual_edges.values())==74005 and history==1004040
 rows=[];counts=collections.Counter();unmatched=[]
 for a in axis:
  t=terms.get(a['axis_id']);native=json.loads(t['native_term_json']) if t else None;counts['source_axis_labels']+=1;counts['exact_current_primary_ID_match']+=t is not None;counts['current_obsolete_label']+=bool(t and t['is_obsolete']);counts['current_name_missing']+=not(t and t['name']);counts['current_aspect_missing']+=not(t and t['aspect'])
  if t is None:unmatched.append(a['axis_id'])
  rows.append(dict(axis_id=a['axis_id'],source_axis_metadata_json=json.dumps(a,sort_keys=True),current_ontology_term_name=t['name'] if t else None,current_ontology_aspect=t['aspect'] if t else None,current_ontology_is_obsolete=t['is_obsolete'] if t else None,current_ontology_definition_json=json.dumps(native.get('definition'),sort_keys=True) if native is not None else None,current_ontology_synonyms_json=json.dumps(native.get('synonyms'),sort_keys=True) if native is not None else None,current_ontology_term_source_json=json.dumps(dict(raw_file=t['source_raw_path'],result_index=t['source_result_index'],native_term_sha256=t['native_term_sha256'])) if t else None,current_ontology_timestamp=m['ontology_timestamp'],current_ontology_version=m['ontology_version'],annotation_source_versions_separate=True,GOA_historical_source_goName_unchanged=True,raw_annotation_ID_not_remapped=True,obsolete_does_not_rewrite_historical_assertion=True,history_used_as_current_edges=False,names_are_separate_reference_metadata_not_hidden_target_predictors=True,new_semantic_or_numeric_task_admitted=False,missing_reference_kept_unknown=t is None))
 p=B/'GO-axis-current-reference-bindings.parquet';pq.write_table(pa.Table.from_pylist(rows),p,compression='zstd');assert pq.read_table(p).to_pylist()==rows
 endpoints={v for c,r,p in actual_edges for v in [c,p]};missing_endpoints=sorted(endpoints-set(terms));relations=collections.Counter()
 for (child_id,relation,parent_id),n in actual_edges.items():relations[relation]+=n
 result=dict(schema='mora3.GOCurrentReferenceAxisBinding/v1',status='complete_source_native_term_edge_and_exact_axis_bindings_verified',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),ontology_source=ref(S),ontology_timestamp=m['ontology_timestamp'],ontology_version=m['ontology_version'],source_annotation_snapshot='GOA2026-07-28;KB2026_03separate',same_snapshot=False,source_license=m['license'],all48392term_objects_against_original_raw_pages=True,all74005current_child_edges_exact_source_multiset=True,native_history_entries_preserved_in_source_refs=history,history_is_not_current_edges=True,current_relation_counts=dict(relations),current_edge_endpoint_IDs_absent_from_full_catalog=missing_endpoints,source_axis=ref(A),binding=ref(p,rows=len(rows)),counts=dict(counts),unmatched_axis_IDs=unmatched,source_typed_payload_refs=m['typed_payloads'],all_original_goNames_annotation_payloads_and_task_shards_unchanged=True,all_Parquet_binding_roundtrip=True,all_target_labels_original_IDs_preserved=True,no_automatic_obsolete_or_alt_ID_remapping=True,no_GO_ancestry_label_propagation=True,no_dense_negative_labels=True,names_and_definitions_not_injected_into_hidden_target_predictors=True,new_model_heads=0,new_training_examples=0,new_independent_experiments=0,optimizer_steps=0,neural_admission='referencecompaniononly;task-specificnativecodec/scope/evidencegatesremain',seconds=time.perf_counter()-start,code=ref(Path(__file__)))
 (B/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');print('COMPLETE',json.dumps(dict(counts)),json.dumps(dict(relations)))
except BaseException as exc:
 (B/'failure.json').open('x').write(json.dumps(dict(error=repr(exc),seconds=time.perf_counter()-start),indent=2)+'\n');raise
