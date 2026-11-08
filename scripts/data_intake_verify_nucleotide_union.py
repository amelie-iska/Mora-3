"""Finite independent all-record intake audit; source and old corpus stay immutable."""
from pathlib import Path
import collections, datetime, hashlib, json, os, sys, time
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[3] if False else Path.cwd()
sys.path.insert(0,str(ROOT/'scripts'))
from data_intake_corpus import CorpusTask,canonical,read_corpus_tasks
from data_intake_nucleotide_prior import prior_identity
B=Path('outputs/data-intake/20261101-nucleotide-corpus-v1')
O=B/'union-verification-v1'; O.mkdir(exist_ok=False)
start=time.perf_counter()
basepath=Path('outputs/data-intake/20261101-tox21-original-admission-v1/master-corpus-manifest-v12-tox21-source-protocol-v1.json')
npath=B/'full-v1/manifest.json'; base=json.loads(basepath.read_text()); n=json.loads(npath.read_text())
def ref(p):
 p=Path(p); h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*2**20),b''): h.update(block)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def check(e):
 got=ref(e['path']); assert got['bytes']==e['bytes'] and got['sha256']==e['sha256'],got
 if 'rows' in e: assert pq.ParquetFile(e['path']).metadata.num_rows==e['rows']
 return Path(e['path'])
def live(stage,**kwargs):
 (O/'live-verification.json').write_text(json.dumps(dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),stage=stage,elapsed_seconds=time.perf_counter()-start,**kwargs),indent=2)+'\n')
 print(stage,kwargs,round(time.perf_counter()-start,2),flush=True)
try:
 held={'validation':set(),'test':set()}; prior_ids=set(); prior_counts=collections.Counter(); schemas=collections.Counter(); adapters=collections.Counter(); allsplit=collections.Counter(); shardbytes=0
 for i,s in enumerate(base['shards']):
  p=check(s); shardbytes+=s['bytes']; g=base['effective_group_lookup'][s['path']]; check(g)
  rows=pq.read_table(p,columns=['task_id','split','source_family','native_schema']).to_pylist(); groups=pq.read_table(g['path'],columns=['task_id','effective_groups_json']).to_pylist(); gm={x['task_id']:x['effective_groups_json'] for x in groups};assert len(gm)==len(rows)==g['rows'];assert set(gm)=={r['task_id'] for r in rows}
  for r in rows:
   assert r['task_id'] not in prior_ids;prior_ids.add(r['task_id']);prior_counts[r['source_family']]+=1;allsplit[r['split']]+=1;schemas[r['native_schema']]+=1
   if r['split'] in held:held[r['split']].update(json.loads(gm[r['task_id']]))
  if i%500==0:live('prior master complete-byte and allocation verification',shards=i+1)
 assert len(prior_ids)==base['serialized_tasks'];assert shardbytes==base['total_shard_bytes'];assert dict(allsplit)==base['split_counts']; assert dict(schemas)==base['native_schema_counts'];assert not held['validation']&held['test']
 # New nucleotide tasks have complete original targets and minimal allowlisted conditions.
 ids={}; source=collections.Counter(); axes=collections.Counter(); newedges=0; newbytes=0; mixed=[]; payloadbytes=0
 for i,s in enumerate(n['shards']):
  p=check(s);newbytes+=s['bytes'];g=n['effective_group_lookup'][s['path']];check(g)
  rows=pq.read_table(p).to_pylist(); gr=pq.read_table(g['path']).to_pylist(); gm={x['task_id']:x['effective_groups_json'] for x in gr};assert len(gm)==len(rows)==g['rows'];assert set(gm)=={r['task_id'] for r in rows}
  if s.get('source_family') is None:mixed.append(s)
  for r in rows:
   tid=r['task_id']; assert tid not in ids and tid not in prior_ids;assert r['split']=='train' and r['admitted_for_task'] and r['model_adapter_supported'] is False;assert r['native_schema']=='mora3.CorpusTask/v1'
   raw=r['native_payload'];assert hashlib.sha256(raw).hexdigest()==r['native_payload_sha256'];t=CorpusTask.from_bytes(raw);assert t.canonical_bytes()==raw;d=json.loads(raw);assert d['task']=='source_native_nucleotide_sequence_prior';assert set(d['conditions'])=={'prior_domain','source_organism'};assert set(d['targets'])=={'source_native_sequence'};seq=d['targets']['source_native_sequence'];assert len(seq)==r['full_axis_values'];assert tid==prior_identity(seq,d['conditions']['prior_domain'],d['conditions']['source_organism']);digest=hashlib.sha256(seq.encode()).hexdigest();ids[tid]=(digest,len(seq));source[r['source_family']]+=1;axes[r['source_family']]+=len(seq);payloadbytes+=len(raw)
   gs=json.loads(gm[tid]);assert set(json.loads(r['group_ids_json']))<=set(gs);assert not set(gs)&held['validation'];assert not set(gs)&held['test'];newedges+=len(gs)
  if i%100==0:live('all new canonical payloads and complete effective groups',shards=i+1,tasks=len(ids))
 assert len(ids)==n['serialized_unique_tasks'];assert dict(source)==n['unique_tasks_by_source'];assert newbytes==n['task_shard_bytes'];assert newedges==n['task_group_edges']
 occurrences=collections.Counter(); first=collections.Counter(); native_axes=collections.Counter(); ledgerbytes=0
 for i,e in enumerate(n['source_occurrence_ledger']):
  p=check(e);ledgerbytes+=e['bytes']
  for r in pq.read_table(p).to_pylist():
   assert r['task_id'] in ids;assert ids[r['task_id']]==(r['native_sequence_sha256'],r['native_length']);occurrences[r['source_family']]+=1;native_axes[r['source_family']]+=r['native_length'];first[r['source_family']]+=int(r['new_unique_task'])
  if i%250==0:live('all source occurrence native identity conservation',ledger_shards=i+1,occurrences=sum(occurrences.values()))
 assert dict(occurrences)==n['source_occurrences'];assert dict(first)==dict(source);assert sum(occurrences.values())-len(ids)==n['duplicate_occurrences_retained'];assert dict(native_axes)==n['source_symbols_by_source']
 for e in n['input_native_payloads']: check({k:v for k,v in e.items() if k!='rows'})
 for e in n['code_refs']+[n['cross_source_overlap_receipt']]:check(e)
 # Exercise exact runtime filtering on the real mixed boundary shard. Whole rows/groups remain immutable.
 mixed_checks=[]
 for s in mixed:
  mp=O/'mixed-shard-probe-manifest.json';mp.write_text(json.dumps(dict(shards=[s],effective_group_lookup={s['path']:n['effective_group_lookup'][s['path']]})))
  expected=collections.Counter(r['source_family'] for r in pq.read_table(s['path'],columns=['source_family']).to_pylist())
  for family,count in expected.items():
   decoded=list(read_corpus_tasks(mp,root=ROOT,source=family,supported_only=False));assert len(decoded)==count and all(r['source_family']==family for r in decoded);assert not list(read_corpus_tasks(mp,root=ROOT,source=family));mixed_checks.append(dict(source_family=family,expected=count,read_count=len(decoded),supported_gate_preserved=True))
 master=json.loads(basepath.read_text());master.update(build_id='20261101-whole-native-nucleotide-corpus-v13',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supersedes_for_consumption=str(basepath),prior_manifest_sha256=ref(basepath)['sha256'],additional_whole_nucleotide_admission_manifest=str(npath),verification=str(O/'receipt.json'))
 master['shards']+=n['shards'];master['effective_group_lookup'].update(n['effective_group_lookup']);master['serialized_tasks']+=n['serialized_unique_tasks'];master['total_shard_bytes']+=newbytes;master['split_counts']['train']+=n['serialized_unique_tasks'];master['native_schema_counts']['mora3.CorpusTask/v1']+=n['serialized_unique_tasks'];master['adapter_support_counts']['mora3.CorpusTask/v1;model_adapter_supported=False']+=n['serialized_unique_tasks'];master['verification_layers'].append(str(O/'receipt.json'))
 for family,inv,lic,release in [('DNA-native-reference-prior','INV-001','CC-BY-4.0','GustavoHCruz/DNA_coding_regions@4e6860536e67a2799c206bb66b0ccce4931e78f1'),('Rfam15.1-native-reference-prior','INV-003','CC0-1.0','officialRfam15.1;January2026;fullrepresentative-genomeFASTA')]:
  master['source_ledger'].append(dict(source_family=family,inventory_id=inv,raw_available=n['source_occurrences'][family],quality_pass=n['source_occurrences'][family],admitted_source_occurrences=n['source_occurrences'][family],serialized_unique_native_context_tasks=source[family],duplicate_occurrences_preserved=n['source_occurrences'][family]-source[family],count_unit='native_occurrences_and_exact_sequence_context_task_priors_separate;not_independent_experiments',release=release,license=lic,model_adapter_supported=False,annotation_or_translation_supervision_admitted=False))
  master['serialized_distinct_record_counts_by_source'][family]=source[family]
  master['contexts'][family]=dict(rows=source[family],source_occurrences=n['source_occurrences'][family],context_units='complete_source_nucleotide_symbols;allcase_T_U_ambiguity_axes_preserved',source_native_length_stats=n['native_length_stats'][family],model_token_count=None,no_silent_truncation=True)
  master['source_task_context_policy'][family]=dict(evidence='released_reference_sequence_prior;not_independent_experimental_observation',conditions=['prior_domain','literal_source_organism_or_None'],targets=['full_native_sequence'],excluded_predictors=['source_ids_headers_coordinates_features_proteins','target_sequence_length_and_hash','audit_metadata'],split='train_only;complete_occurrence_identity_parent_groups_recorded;homology_alias_taxonomy_heldouts_unresolved',missing=['experimental_modality_pairs','verified_CDS_translation','Rfam_family_confidence_alignment_annotation_join','native_neural_reader'])
 master['mixture_proposal']['nucleotide_reference_prior_allocation']=dict(status='unactivated_provisional;no_sampling_weight_set',weight=None,requires='native variable-length character reader and masks;full-axis memory/throughput;independent family/parent/taxon-heldout study;deduplicated-observation-aware objective balancing')
 master['open_gates'].append('Nucleotide priors remain train-only;full effective occurrence/source-parent groups are authoritative;unknown accession aliases/homology/taxon/evaluation coverage and native decoder support unresolved;reference symbols are not new experiments or verified coding translations')
 assert len(master['shards'])==9364 and master['serialized_tasks']==9444588 and master['total_shard_bytes']==6103433868;assert sum(master['split_counts'].values())==master['serialized_tasks'];assert sum(master['native_schema_counts'].values())==master['serialized_tasks'];assert len(master['effective_group_lookup'])==len(master['shards'])
 mpath=B/'master-corpus-manifest-v13-whole-nucleotide-priors-v1.json'
 # Publication is the final write after the complete audit passes.
 receipt=dict(schema='mora3.IndependentWholeNucleotideUnionVerification/v1',status='passed',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_master=ref(basepath),new_component=ref(npath),all_prior_shards_complete_byte_hash_footer_and_task_group_allocation=True,prior_tasks=len(prior_ids),all_new_shards_complete_byte_hash_footer_and_full_payload_codec_roundtrip=True,new_unique_tasks=len(ids),new_tasks_by_source=dict(source),all_new_target_axes_and_allowlisted_predictors_checked=True,all_occurrence_native_sequence_hash_length_conservation=True,source_occurrences=dict(occurrences),duplicate_occurrences_retained=n['duplicate_occurrences_retained'],all228native_source_payload_hashes_rechecked=True,all_effective_group_shards_hash_taskset_and_known_heldout_intersection_checked=True,new_task_group_edges=newedges,new_known_validation_group_intersections=0,new_known_test_group_intersections=0,global_leakage_safe=False,homology_unknown_parent_aliases_and_PDB_nucleic_sequence_crosswalk_unresolved=True,source_context_not_paired_assays=True,mixed_source_runtime_checks=mixed_checks,new_task_shards=len(n['shards']),new_task_bytes=newbytes,new_full_canonical_payload_bytes=payloadbytes,occurrence_ledger_bytes=ledgerbytes,master_path=str(mpath),master_totals=dict(tasks=master['serialized_tasks'],shards=len(master['shards']),data_bytes=master['total_shard_bytes'],split_counts=master['split_counts']),model_adapter_supported_for_new_tasks=False,new_optimizer_steps=0,seconds=time.perf_counter()-start)
 (O/'receipt.json').open('x').write(json.dumps(receipt,indent=2)+'\n');mpath.open('x').write(json.dumps(master,indent=2)+'\n');live('verified v13 master published',master=str(mpath),tasks=master['serialized_tasks']);print('COMPLETE',json.dumps(receipt['master_totals']),flush=True)
except BaseException as exc:
 (O/'failure.json').open('x').write(json.dumps(dict(error=repr(exc),elapsed_seconds=time.perf_counter()-start),indent=2)+'\n');raise
