"""Exact native sequence-pair exposure guards; no accession/annotation transfer."""
from pathlib import Path
import json,hashlib,collections,time,datetime,functools,os,csv,gzip
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261101-Bernett-corpus-readiness-v1';O=B/'STRING-pair-exposure-v2';O.mkdir();start=time.perf_counter()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(record):
 p=Path(record['path']);p=p if p.is_absolute() else R/p;assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256'];return p
C=R/'outputs/data-sources/20261031-legacy-mint-v1/curated/STRING-qualified-sequence-SHA-candidates-v1/manifest.json';assert sha(C)=='84dfa46abc8f0fdf80740f142ca788b42977a6342d4a8e321648e24b6b4a0cef';c=json.loads(C.read_text());human={};native_cache={}
with gzip.open(R/'outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz','rt') as stream:qualifiedAA={r['Entry']:hashlib.sha256(r['Sequence'].encode()).hexdigest() for r in csv.DictReader(stream,delimiter='\t')}
for file in c['payloads']:
 for row in pq.read_table(check(file)).to_pylist():
  if row['taxon_id']!=9606:continue
  path=Path(row['source_native_payload'])
  if str(path) not in native_cache:
   assert sha(path)==row['source_native_payload_sha256'];native_cache[str(path)]=pq.read_table(path).to_pylist()
  native=native_cache[str(path)][row['source_local_row_index']];assert native['provider_id']==row['provider_id'] and native['native_sequence_sha256']==row['native_sequence_sha256'] and native['native_length']==row['native_length'];assert native['taxon_id']==row['taxon_id'] and native['source_record_index']==row['source_record_index'];assert native['qualified_exact_sequence_accessions']==row['qualified_exact_sequence_accessions'];assert all(qualifiedAA[accession]==row['native_sequence_sha256'] for accession in row['qualified_exact_sequence_accessions']);assert row['provider_id'] not in human;human[row['provider_id']]=row['native_sequence_sha256']
  assert row['annotation_or_function_transferred'] is False and row['source_provider_ID_is_qualified_UniProt_accession'] is False
A=B/'manifest.json';a=json.loads(A.read_text());bench={};records=[]
for file in a['outputs']:
 for row in pq.read_table(check(file)).to_pylist():
  records.append(row)
  if row['both_exact_qualified_native_sequences']:
   key=row['unordered_exact_sequence_pair_key'];assert key not in bench;bench[key]=row
M=R/'outputs/data-intake/20261101-RNAcentral-pilot-union-v1/master-corpus-manifest-v18-RNAcentral-native-pilot-v1.json';m=json.loads(M.read_text());counts=collections.Counter();exposures=collections.defaultdict(list);used=[]
(O/'live-audit.json').write_text(json.dumps(dict(pid=os.getpid(),status='scanningallcurrentadmittedSTRINGhumantrainpairpayloads',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
for sh in m['shards']:
 if sh.get('source_family')!='STRING-human-score-prior':continue
 path=check(sh);used.append(ref(path));assert sh['split']=='train'
 for row in pq.read_table(path).to_pylist():
  counts['current_admitted_STRINGhuman_train_task_rows']+=1;assert row['source_family']=='STRING-human-score-prior' and row['split']=='train' and row['admitted_for_task'];payload=row['native_payload'];assert hashlib.sha256(payload).hexdigest()==row['native_payload_sha256'];d=json.loads(payload);assert d['task']=='source_integrated_association_score_profile';cond=d['conditions'];p,q=cond['protein1_source_id'],cond['protein2_source_id'];assert cond['taxon']==9606
  if p not in human or q not in human:counts['rows_without_both_current_qualified_sequence_SHA_candidates']+=1;continue
  counts['rows_with_both_exact_sequence_SHA_candidates']+=1;key='|'.join(sorted([human[p],human[q]]))
  if key not in bench:continue
  t=d['targets'];values={k:v['value'] for k,v in t.items()};exposures[key].append(dict(current_STRING_task_id=row['task_id'],STRING_provider1=p,STRING_provider2=q,STRING_source_version='12.0',current_master_shard=sh['path'],native_payload_sha256=row['native_payload_sha256'],score_channels_json=json.dumps(values,sort_keys=True),score_role='sourceintegratedconfidenceprior;notindependentassay',source_experimental_channel_nonzero=values['experimental_channel']>0,source_database_channel_nonzero=values['database_channel']>0,source_textmining_channel_nonzero=values['textmining_channel']>0,direct_binding_proven=False,annotation_accession_or_function_transfer=False))
summary=collections.defaultdict(collections.Counter);out=[]
for row in records:
 key=row['unordered_exact_sequence_pair_key'];matches=exposures.get(key,[]) if key else [];split=row['source_native_split'];label='source_positive' if row['source_benchmark_label']==1 else 'source_random_sampled_negative';summary[split]['all_source_pairs']+=1;summary[split]['both_qualified_pairs']+=row['both_exact_qualified_native_sequences'];summary[split][label+'_with_existing_STRING_pair_prior']+=bool(matches);summary[split][label+'_with_nonzero_source_experimental_channel']+=any(x['source_experimental_channel_nonzero'] for x in matches);out.append(dict(source_record_id=row['source_record_id'],source_native_split=split,source_benchmark_label=row['source_benchmark_label'],both_qualified_native_sequences=row['both_exact_qualified_native_sequences'],unordered_exact_sequence_pair_key=key,current_admitted_STRING_pair_prior_match_count=len(matches),source_pair_prior_exposure_candidates_json=json.dumps(matches,sort_keys=True),same_exact_sequence_guard_not_accession_identity=True,actual_neural_optimizer_exposure_unknown=True,source_random_negative_not_relabelled_as_experimental_noninteraction=True,heldout_admission_review_required=True,new_training_task_admitted=False))
outputs=[]
for split in ['train','validation','test']:
 selected=[r for r in out if r['source_native_split']==split];path=O/('source-pair-exposure-'+split+'.parquet');pq.write_table(pa.Table.from_pylist(selected),path,compression='zstd');assert pq.read_table(path).to_pylist()==selected;outputs.append(ref(path,rows=len(selected)))
assert counts['current_admitted_STRINGhuman_train_task_rows']==738805;assert len(out)==274500
(O/'code.py').write_bytes(Path(__file__).read_bytes());result=dict(schema='mora3.BernettCurrentSTRINGExactSequencePairExposure/v1',status='all_current_admitted_STRINGhuman_tasks_and_all_Bernett_native_pairs_audited',source_pair_group_review=ref(A),STRING_provider_SHA_candidates=ref(C),current_master=ref(M),human_providerIDs_independently_bound_to_native_sequence_SHA_index=len(human),counts=dict(counts),source_protocol_exposure_counts={k:dict(v) for k,v in summary.items()},benchmark_qualified_sequence_pair_keys_with_admitted_STRING_prior=len(exposures),all_current_STRING_payload_hashes_and_source_target_semantics_rechecked=True,raw_provider_FASTA_AA_replay='inheritedwholeverifiedsourceQC;thisauditindependentlybindscompactcandidate_to_hashcheckednativeindex_and_currentqualifiedbaselineAA_SHA;notnew59Mrawreplay',same_sequence_pair_is_exposure_guard_not_forced_accession_or_annotation_identity=True,source_qualified_sequence_candidates_not_new_protein_admission=True,all_current_STRING_prior_rows_train_only=True,source_sampled_negatives_preserved_not_converted_to_experimental_negatives=True,current_admitted_pool_exposure_not_actual_optimizer_usage=True,original_source_splits_unchanged=True,new_eval_or_train_task_admissions=0,upstream_HIPPIE_perpair_rights_and_evidence_pending=True,other_interaction_text_pathway_prior_exposure_scope='notexhaustivelyresolved;noabsence/blindclaim',outputs=outputs,checked_current_STRING_shards=used,optimizer_steps=0,seconds=time.perf_counter()-start)
(O/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');(O/'live-audit.json').write_text(json.dumps(dict(pid=os.getpid(),status='complete_verified',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n');print(json.dumps({k:result[k] for k in ['human_providerIDs_independently_bound_to_native_sequence_SHA_index','counts','source_protocol_exposure_counts','benchmark_qualified_sequence_pair_keys_with_admitted_STRING_prior','seconds']},indent=2))
