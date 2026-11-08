"""Independent all-row RNAcentral pilot proof, prior RNA provenance overlay, immutable v18."""
from pathlib import Path
import sys,json,hashlib,collections,sqlite3,datetime,time,functools,shutil
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import CorpusTask
from data_intake_nucleotide_prior import prior_identity,sequence_groups
from data_intake_rnacentral_prior import ALPHABET
from data_intake_admit_uniprot_annotations import resolved_shard_rows
from data_intake_corpus_audit_metadata import adapter_support
P=R/'outputs/data-intake/20261101-RNAcentral-corpus-pilot-v1/manifest.json';M=R/'outputs/data-intake/20261101-full-GOA-corpus-union-v2/master-corpus-manifest-v17-full-canonical-GOA-v1.json';O=R/'outputs/data-intake/20261101-RNAcentral-pilot-union-v1'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  while chunk:=stream.read(4*2**20):h.update(chunk)
 return h.hexdigest()
def ref(path,**kw):return dict(path=str(path.relative_to(R)),bytes=path.stat().st_size,sha256=sha(path),**kw)
def check(record):
 path=Path(record['path']);path=path if path.is_absolute() else R/path;assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256'];return path
@functools.lru_cache(maxsize=3)
def source_rows(path,digest):
 p=R/path;assert sha(p)==digest;return pq.read_table(p).to_pylist()

def main():
 start=time.perf_counter();assert not O.exists() and shutil.disk_usage(R).free>202*2**30;O.mkdir();(O/'code').mkdir();(O/'Rfam-extra-source-groups').mkdir();(O/'code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes());p=json.loads(P.read_text());m=json.loads(M.read_text());master_hash=sha(M);index=check(p['new_identity_and_group_index']);db=sqlite3.connect('file:'+str(index)+'?mode=ro',uri=True);old_manifest=json.loads((R/'outputs/data-intake/20261101-nucleotide-corpus-v1/full-v1/manifest.json').read_text());old_index=check(old_manifest['generated_identity_index']);old=sqlite3.connect('file:'+str(old_index)+'?mode=ro',uri=True);native_seq={r[0]:(r[1],bool(r[2])) for r in db.execute('SELECT native_sha,task_id,reused_prior FROM sequences')};reused={tid for tid,reuse in native_seq.values() if reuse};allgroups=collections.defaultdict(set)
 for tid,group in db.execute('SELECT task_id,group_id FROM task_groups'):allgroups[tid].add(group)
 assert len(native_seq)==p['counts']['distinct_native_sequences'] and len(reused)==661
 observed=collections.Counter();seen_native=set();seen_source_records=set();new_tasks={};source_files=set();quarantine_count=0
 for ledger in p['occurrences']:
  path=check(ledger);table=pq.read_table(path).to_pylist();assert len(table)==ledger['rows']
  for row in table:
   sr=json.loads(row['source_ref_json']);src=source_rows(sr['native_payload'],sr['native_payload_sha256'])[sr['native_row_index']];seq=src['native_sequence'];digest=hashlib.sha256(seq.encode()).hexdigest();assert digest==src['native_sequence_sha256']==row['native_sequence_sha256'];assert len(seq)==src['native_length']==row['native_length'] and src['source_identifier']==sr['source_identifier'] and src['native_header']==sr['native_header'] and src['source_record_index']==sr['original_record_index'] and src['source_payload']==sr['original_FASTA'];assert src['source_record_id'] not in seen_source_records;seen_source_records.add(src['source_record_id']);assert set(seq)<=ALPHABET and src['sequence_alphabet_pass'];tid,reuse=native_seq[digest];assert tid==row['task_id'] and reuse==row['task_already_in_Rfam_corpus'];assert row['first_occurrence_in_pilot']==(digest not in seen_native);seen_native.add(digest);sg=sequence_groups(seq);expected={'nucleotide_native:'+digest,'nucleotide_upper:'+sg['uppercase_sha256'],'nucleotide_derived_RC_baseorder:'+sg['reverse_complement_base_order_group'],'rnacentral_native_identifier:'+src['source_identifier']};assert expected<=allgroups[tid] and row['uppercase_sequence_sha256']==sg['uppercase_sha256'] and row['derived_RC_baseorder_sha256']==sg['reverse_complement_base_order_group'];assert row['derived_RC_is_molecular_identity'] is False and row['source_status_is_not_negative_label'] is True
   if reuse:
    assert tid==prior_identity(seq,'RNA_family_reference_native_symbols',None);family=old.execute('SELECT source_family FROM tasks WHERE task_id=?',(tid,)).fetchone();assert family and family[0].startswith('Rfam');observed['reused_Rfam_occurrences']+=1
   else:
    assert tid==prior_identity(seq,'RNA_reference_native_symbols',None);assert tid not in new_tasks;new_tasks[tid]=(digest,src['source_record_id']);observed['new_tasks']+=1
   observed['quality_pass_occurrences']+=1;source_files.add(sr['native_payload']);observed['native_values']+=len(seq)
 for ledger in p['quarantines']:
  for row in pq.read_table(check(ledger)).to_pylist():
   sr=json.loads(row['source_ref_json']);src=source_rows(sr['native_payload'],sr['native_payload_sha256'])[sr['native_row_index']];seq=src['native_sequence'];assert set(seq)-ALPHABET and not src['sequence_alphabet_pass'];assert hashlib.sha256(seq.encode()).hexdigest()==row['native_sequence_sha256'];assert src['source_record_id'] not in seen_source_records;seen_source_records.add(src['source_record_id']);observed['native_values']+=len(seq);quarantine_count+=1
 assert observed['quality_pass_occurrences']==p['counts']['quality_pass_occurrences'] and quarantine_count==p['counts']['quarantined_occurrences'] and len(seen_source_records)==p['counts']['source_rows'] and observed['native_values']==p['counts']['source_native_values']
 payload_count=0
 for shard in p['shards']:
  sp=check(shard);gp=check(p['effective_group_lookup'][shard['path']]);groups=resolved_shard_rows(sp,gp,shard)
  for row,gr in zip(pq.read_table(sp).to_pylist(),groups):
   tid=row['task_id'];assert tid in new_tasks and tid not in reused;digest,record=new_tasks.pop(tid);assert record==row['record_id'];payload=row['native_payload'];assert hashlib.sha256(payload).hexdigest()==row['native_payload_sha256'];assert CorpusTask.from_bytes(payload).canonical_bytes()==payload;d=json.loads(payload);assert d['task']=='source_native_nucleotide_sequence_prior' and d['conditions']==dict(prior_domain='RNA_reference_native_symbols',source_organism=None);sr=d['audit']['source_ref'];src=source_rows(sr['native_payload'],sr['native_payload_sha256'])[sr['native_row_index']];assert d['targets']==dict(source_native_sequence=src['native_sequence']) and hashlib.sha256(src['native_sequence'].encode()).hexdigest()==digest;assert row['full_axis_values']==src['native_length'];assert row['admitted_for_task'] and not row['model_adapter_supported'] and row['split']=='train';assert set(json.loads(gr['effective_groups_json']))==allgroups[tid];payload_count+=1
 assert not new_tasks and payload_count==p['counts']['distinct_new_task_views']
 # New URS/source links augment existing exact Rfam tasks, with no payload edits.
 touched=0;overlaid=0
 for shard in m['shards']:
  if shard.get('source_family') not in [None,'Rfam15.1-native-reference-prior']:continue
  taskids=[r['task_id'] for r in pq.read_table(R/shard['path'],columns=['task_id']).to_pylist()]
  if not reused.intersection(taskids):continue
  gr=pq.read_table(check(m['effective_group_lookup'][shard['path']])).to_pylist()
  for row in gr:
   if row['task_id'] in reused:row['effective_groups_json']=json.dumps(sorted(set(json.loads(row['effective_groups_json']))|allgroups[row['task_id']]));overlaid+=1
  target=O/'Rfam-extra-source-groups'/f'{touched:04d}.parquet';pq.write_table(pa.Table.from_pylist(gr),target,compression='zstd');assert pq.read_table(target).to_pylist()==gr;m['effective_group_lookup'][shard['path']]=ref(target,rows=len(gr));touched+=1
 assert overlaid==len(reused);m['shards']+=p['shards'];m['effective_group_lookup'].update(p['effective_group_lookup']);n=p['counts']['distinct_new_task_views'];m['serialized_tasks']+=n;m['total_shard_bytes']+=p['task_shard_bytes'];m['split_counts']['train']+=n;m['native_schema_counts']['mora3.CorpusTask/v1']+=n;m['adapter_support_counts']['mora3.CorpusTask/v1;model_adapter_supported=False']+=n;m['serialized_distinct_record_counts_by_source']['RNAcentral-reference-priors']=n
 ids=set();splits=collections.Counter();schemas=collections.Counter();adapters=collections.Counter();source_counts=collections.Counter();source_records=collections.defaultdict(set);all_split_groups=collections.defaultdict(set);data_bytes=group_bytes=0
 for i,shard in enumerate(m['shards']):
  sp=check(shard);gp=check(m['effective_group_lookup'][shard['path']]);data_bytes+=sp.stat().st_size;group_bytes+=gp.stat().st_size;group_rows=resolved_shard_rows(sp,gp,shard);cols=['task_id','record_id','source_family','split','native_schema'];
  if 'model_adapter_supported' in pq.ParquetFile(sp).schema_arrow.names:cols.append('model_adapter_supported')
  rows=pq.read_table(sp,columns=cols).to_pylist();assert len(rows)==shard['rows']==len(group_rows)
  for row,gr in zip(rows,group_rows):
   tid=row['task_id'];assert tid==gr['task_id'] and tid not in ids;ids.add(tid);splits[row['split']]+=1;schemas[row['native_schema']]+=1;adapters[row['native_schema']+';model_adapter_supported='+str(adapter_support(row))]+=1;source_counts[row['source_family']]+=1;source_records[row['source_family']].add(row['record_id']);all_split_groups[row['split']].update(json.loads(gr['effective_groups_json']))
  if i%1000==0:print('RNA_UNION',i+1,'tasks',len(ids),'seconds',round(time.perf_counter()-start,2),flush=True)
 assert len(ids)==m['serialized_tasks'] and dict(splits)==m['split_counts'] and dict(schemas)==m['native_schema_counts'] and dict(adapters)==m['adapter_support_counts'] and data_bytes==m['total_shard_bytes'];assert {k:len(v) for k,v in source_records.items()}==m['serialized_distinct_record_counts_by_source'];intersection={a+'__'+b:len(all_split_groups[a]&all_split_groups[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]};assert not any(intersection.values());assert sha(M)==master_hash;old.close();db.close()
 receipt=dict(schema='mora3.RNAcentralPilotIndependentMasterAdmission/v1',status='verified',prior_master=ref(M),component=ref(P),all_selected_native_source_occurrences_independently_checked=len(seen_source_records),all_new_task_payloads_source_SHA_target_and_fullaxis_roundtrip=payload_count,all_nonstandard_quarantines_independently_preserved=quarantine_count,old_Rfam_exact_native_task_identity_reused_without_serialized_duplicates=overlaid,old_Rfam_effective_sidecars_augmented=touched,all_old_and_new_data_group_hashes_and_envelope_metadata_read=True,source_raw_FASTA_QC_scope='whole61Mrawnativeproofinheritedsourcehandoff;selectedtypedsourcewholebytes+allrowliteralrereadhere',prior_Rfam_reuse_scope='priorwholeverifiedindexhashandliteralnativeidentityrecipe;notnewoldtaskpayloaddecode',known_group_split_intersections=intersection,serialized_tasks=len(ids),shards=len(m['shards']),data_bytes=data_bytes,effective_group_bytes=group_bytes,split_counts=dict(splits),source_tasks=dict(source_counts),all_global_homology_or_broad_heldout_asserted=False,new_independent_experiments=0,optimizer_steps=0,seconds=time.perf_counter()-start)
 rp=O/'receipt.json';rp.open('x').write(json.dumps(receipt,indent=2)+'\n');m.update(build_id='20261101-RNAcentral-native-prior-pilot-corpus-v18',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_manifest_sha256=master_hash,supersedes_for_consumption=str(M.relative_to(R)),source_components_base_manifest=str(M.relative_to(R)),additional_RNAcentral_native_prior_manifest=str(P.relative_to(R)),verification=str(rp.relative_to(R)));m['verification_layers'].append(str(rp.relative_to(R)));m['source_curation_manifests'].append(ref(P));m['source_ledger'].append(dict(source_family='RNAcentral-reference-priors',inventory_id='INV-004',release='officialRNAcentral28.0active+inactive',license='CC0-1.0',raw_available=61026468,quality_pass_source_alphabet_candidates=61006261,source_nonstandard_symbol_quarantines=20207,retained_pilot_source_occurrences=p['counts']['quality_pass_occurrences'],pilot_source_rows=p['counts']['source_rows'],pilot_source_quarantines=quarantine_count,admitted_records=n,serialized_tasks=n,reused_existing_Rfam_tasks=overlaid,count_unit='sourceFASTAoccurrencesvsexactnativecontexttaskviews;notindependentexperiments',note='boundedquantilepluslongestnativepilot;full61Mnotadmitted;sourceactive/inactivestatusnotaresponseannotation;fullreleasedcount13discrepancyretained'));m['contexts']['RNAcentral-reference-priors']=dict(rows=n,full_native_length_range=p['native_length_range'],native_axis_values='allselectednativesequences;no8192crop',source_status='auditonlynotpredictorornegative',headers_and_URS='audit/groupsonlynotfunctiontargets',model_token_count=None);m['open_gates'].append('RNAcentralnativepilotonly;full61Midentity/URSconsistency/fullbuildbudgetandfinalmixtureunresolved;13releasecountdiscrepancypreserved;nativeRNAreader/domainandlongcontextsupportunvalidated;allnewpriorstrainonly;homology/taxonomyheldoutsunresolved');m['mixture_proposal']['RNAcentral_reference_prior_allocation']=dict(status='provisional_unactivated;nativepriornotpairedresponse;overlapRfamdedupbyliteralnativeidentity',weight=None,requires='model-sourceadapterandfullaxisthroughput/quality;wholepoolbudgetandgroupaudit;genuineheldoutprotocol');target=O/'master-corpus-manifest-v18-RNAcentral-native-pilot-v1.json';target.open('x').write(json.dumps(m,indent=2)+'\n');print('COMPLETE',str(target.relative_to(R)),len(ids),len(m['shards']),data_bytes,flush=True)
if __name__=='__main__':main()
