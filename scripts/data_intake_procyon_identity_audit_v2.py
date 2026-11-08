"""Reviewed streaming continuation of the completed entity prefix, no replay.

The immutable v1 entity payload/source/code evidence are retained. This phase
streams its readback and original relation split membership with named guards.
"""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,os,resource,sys,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_procyon_stream_guards import threshold_failures,validate_entity_row,stream_row_digest
B=R/'outputs/data-intake/20261101-ProCyon-native-corpus-v1';P=B/'streaming-identity-protocol-audit-plan-v2.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while x:=f.read(2**20):h.update(x)
 return h.hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def checked(v):
 p=Path(v['path']);p=p if p.is_absolute() else R/p
 if p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:raise ValueError('immutable corrective input changed: '+str(p))
 return p
def approved(a,plan_ref,script_ref):
 if a.get('approved') is not True or a.get('scope')!='ProCyon-streaming-completed-prefix-and-original-protocol-audit-only':raise ValueError('exact corrective streaming scope approval required')
 if a.get('plan_sha256')!=plan_ref['sha256'] or a.get('script_sha256')!=script_ref['sha256']:raise ValueError('exact corrective plan/script binding required')
 if a.get('task_admission') is not False or a.get('model_use') is not False or a.get('master_publication') is not False or a.get('model_training') is not False:raise ValueError('no admission, publication, model use or training is authorized')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--approval',required=True);args=ap.parse_args();start=time.monotonic();started=time.time();os.sched_setaffinity(0,{26,27});os.nice(10);pa.set_cpu_count(1);pa.set_io_thread_count(1);plan=json.loads(P.read_text());approval=R/args.approval;approved(json.loads(approval.read_text()),ref(P),ref(Path(__file__).resolve()))
 for v in list(plan['inputs'].values())+plan['dependencies']:checked(v)
 O=R/plan['output_namespace'];assert not O.exists();O.mkdir();samples=[];stage='initial';rows_done=0
 def guard():
  own_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;own_current=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE');other=Path('/proc/3501176/statm');rna=int(other.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') if other.exists() else 0;st=os.statvfs(R);sample=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,rows_done=rows_done,elapsed_seconds=time.monotonic()-start,deadline_utc=datetime.datetime.fromtimestamp(started+120,datetime.timezone.utc).isoformat(),own_current_RSS_bytes=own_current,own_peak_RSS_bytes=own_peak,current_RNA_RSS_bytes=rna,wrapper_reserve_bytes=128*2**20,aggregate_RSS_with_reserve_bytes=own_peak+rna+128*2**20,free_hardware_bytes=st.f_bavail*st.f_frsize);limits=dict(elapsed_seconds=120,own_peak_RSS_bytes=512*2**20,aggregate_RSS_with_reserve_bytes=8*2**30,free_hardware_bytes=64*2**30);failures=threshold_failures(sample,limits);sample['threshold_failures']=failures;samples.append(sample)
  if failures:
   p=O/'guard-stop-diagnostics.json';p.open('x').write(json.dumps(dict(sample=sample,limits=limits,thresholds_exceeded=failures,all_partial_outputs_retained=True),indent=2)+'\n');raise RuntimeError('named audit guard: '+','.join(failures))
 guard();h=json.loads(checked(plan['inputs']['source_handoff']).read_text());m=json.loads(checked(plan['inputs']['source_QC_manifest']).read_text());entity=checked(plan['inputs']['completed_v1_entity_payload']);native={};counts=collections.Counter();stage='stream_completed_entity_prefix'
 for batch in pq.ParquetFile(entity).iter_batches(batch_size=1024):
  for row in batch.to_pylist():
   status=validate_entity_row(row);key=(row['source_namespace'],row['native_ID']);assert key not in native;native[key]=(row['native_sequence_SHA256'],status);rows_done+=1;counts[key[0]+':native_AA_occurrence_symbols_not_model_tokens']+=row['native_length'];counts[key[0]+':native_rows_longer_than8192']+=row['native_length']>8192
   if key[0]=='protein':counts['protein:'+status]+=1
   else:
    q=json.loads(row['domain_parent_candidate_json']);counts['domain:native_records']+=1;counts['domain:prefix_candidate_source_protein_present']+=q['prefix_candidate_exists_in_source_whole_protein_FASTA'];counts['domain:prefix_candidate_fullprotein_current_qualified']+=q['prefix_candidate_whole_protein_current_qualified'];counts['domain:unique_literal_substring_candidate']+=q['unique_exact_substring_candidate'];counts['domain:multiple_literal_substring_spans']+=len(q['derived_literal_substring_spans_1based_closed'])>1;counts['domain:no_literal_substring_candidate']+=len(q['derived_literal_substring_spans_1based_closed'])==0
  guard()
 assert rows_done==59660
 for f in m['FASTA_files']:
  kind='protein' if '/protein/' in f['source_path'] else 'domain';assert counts[kind+':native_AA_occurrence_symbols_not_model_tokens']==f['whole_axis_AA_characters']
 stage='independent_completed_prefix_stream_readback';digest1=stream_row_digest(pq.ParquetFile(entity).iter_batches(batch_size=1024));guard();digest2=stream_row_digest(pq.ParquetFile(entity).iter_batches(batch_size=4096));guard();assert digest1==digest2 and digest1[0]==59660;relations=[]
 for f in m['CSV_files']:
  if '_relations.unified.csv' not in f['source_path']:continue
  stage='source_original_relation:'+f['source_path'];rows_done=0;guard();p=checked(f['payload']);checked(f['source']);splits=collections.defaultdict(lambda:dict(sequence_ID=set(),sequence_SHA=set(),text_ID=set(),exact_relation=set()));local=collections.Counter();raw_rel=collections.Counter();source_qualified=collections.Counter()
  for batch in pq.ParquetFile(p).iter_batches(batch_size=4096):
   for r in batch.to_pylist():
    raw=json.loads(r['native_fields_json']);kind=raw['seq_type'];seqid=raw['seq_id'];text=raw['text_id'];split=raw['split'];assert split==r['native_split'] and r['native_row_index']==rows_done;rows_done+=1;local[split]+=1;raw_rel[raw['relation']]+=1;key=(kind,seqid);s=splits[split];s['sequence_ID'].add(seqid);s['text_ID'].add(text);s['exact_relation'].add(hashlib.sha256(json.dumps([kind,seqid,raw['text_type'],text,raw['relation']],ensure_ascii=False,separators=(',',':')).encode()).digest())
    if key in native:s['sequence_SHA'].add(native[key][0]);source_qualified[native[key][1]]+=1
    else:source_qualified['native_sequence_ID_absent_from_source_FASTA']+=1
   guard()
  assert rows_done==f['rows'];train=splits.get('CL_train',{k:set() for k in ['sequence_ID','sequence_SHA','text_ID','exact_relation']});overlaps={s:{k:len(v[k]&train[k]) for k in train} for s,v in splits.items() if s!='CL_train'};relations.append(dict(source_path=f['source_path'],rows=rows_done,original_source=f['source'],native_payload=f['payload'],exact_native_split_counts=dict(local),source_native_relation_literal_counts=dict(raw_rel),native_source_protein_qualification_row_counts=dict(source_qualified),source_split_distinct_entity_counts={s:{k:len(v[k]) for k in v} for s,v in splits.items()},source_nontrain_overlap_with_source_CL_train=overlaps,source_CL_train_is_not_actual_Mora_model_training=True,original_relation_strings_authoritative_not_global_indexed_code_zero=True,source_historical_predicate_not_assumed_current_experimental_truth=True));print('PROCYON_STREAMING_RELATION_SPLIT_AUDIT',f['source_path'],rows_done,flush=True)
 assert sum(x['rows'] for x in relations)==433007;stage='final_hash_and_guard';rows_done=433007
 for v in list(plan['inputs'].values())+plan['dependencies']:checked(v)
 guard();receipt=dict(schema='mora3.ProCyonCurrentWholeSequenceAndNativeProtocolAudit/v2',status='complete_selected_open_native_entity_qualification_and_original_relation_split_audit;no_tasks_or_master_admission',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_plan=ref(P),approval=ref(approval),inputs=plan['inputs'],source_dataset_revision=h['dataset_revision'],current_qualification_release='2026_03;not_historical_ProCyon_release_equivalence',native_entity_count=59660,native_proteins=18174,native_domains=41486,current_reviewed_annotation4plus_baseline_entries=107778,counts=dict(counts),entity_qualification=plan['inputs']['completed_v1_entity_payload'],completed_entity_prefix_reused_not_recomputed=True,entity_prefix_all_field_stream_readback=dict(rows=digest1[0],framed_row_SHA256=digest1[1],independent_batch_sizes=[1024,4096]),original_relation_tables=relations,original_relation_view_rows=433007,all_native_source_relation_type_and_split_strings_preserved=True,domain_prefix_or_derived_substring_not_native_parent_coordinate_or_label_inheritance=True,exact_current_whole_protein_match_not_automatic_relation_or_text_admission=True,source_node_overlap_not_actual_model_optimizer_use=True,global_homology_and_phenotype_parent_leakage_safe=False,opaque_pickles_not_read_or_unpickled=True,new_task_views=0,new_global_admitted_records=0,model_use=False,full_dataset_ready=False,substantive_training=False,source_immutability=True,resource_samples=samples,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);(O/'manifest.json').open('x').write(json.dumps(receipt,indent=2)+'\n');print('COMPLETE_PROCYON_STREAMING_IDENTITY_PROTOCOL_AUDIT',json.dumps(dict(counts=dict(counts),seconds=receipt['seconds'],peak_RSS_bytes=receipt['peak_RSS_bytes'],manifest=ref(O/'manifest.json'))),flush=True)
if __name__=='__main__':main()
