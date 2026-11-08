"""Whole-source RNA prior materialization AFTER census and separate root review.

All old task bytes remain immutable. Reused prior identities are catalogued for
new effective-group sidecars during the later independent master union.
"""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,os,resource,sqlite3,sys,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import CorpusTask
from data_intake_full_rnacentral_state import NativeIdentityIndex
from data_intake_full_rnacentral_census import validate_native_row
from data_intake_full_rnacentral_tasks import full_native_prior_envelope,native_symbol_units
B=R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while x:=f.read(4*2**20):h.update(x)
 return h.hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p
 if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('immutable source hash/size mismatch')
 return p
def write(p,v):p.open('x').write(json.dumps(v,indent=2)+'\n')
def validate_materialization_approval(a,*,plan,census,clock,script,resource,dependencies):
 if a.get('approved') is not True or a.get('phase')!='materialize' or a.get('scope')!='whole-native-RNAcentral-materialization-only':raise ValueError('separate exact materialization approval required')
 for key,value in [('plan_sha256',plan),('census_manifest_sha256',census),('clock_sha256',clock),('script_sha256',script),('resource_coordination_sha256',resource)]:
  if a.get(key)!=value:raise ValueError('materialization review version mismatch: '+key)
 if a.get('dependency_refs')!=dependencies:raise ValueError('materialization dependency snapshot mismatch')


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--approval',type=Path,required=True);a=parser.parse_args();approval=a.approval if a.approval.is_absolute() else R/a.approval;planpath=B/'build-plan.json';plan=json.loads(planpath.read_text());cp=B/'census/manifest.json';c=json.loads(cp.read_text());clockpath=B/'build-clock.json';clock=json.loads(clockpath.read_text());dependencies=[ref(R/'scripts'/n) for n in ['data_intake_full_rnacentral_tasks.py','data_intake_full_rnacentral_state.py','data_intake_full_rnacentral_census.py','data_intake_rnacentral_prior.py','data_intake_nucleotide_prior.py','data_intake_corpus.py']];validate_materialization_approval(json.loads(approval.read_text()),plan=sha(planpath),census=sha(cp),clock=sha(clockpath),script=sha(Path(__file__)),resource=plan['resource_coordination']['sha256'],dependencies=dependencies);assert c['status']=='complete_full_source_census_pending_materialization_and_independent_proof' and c['source_counts']['source_occurrences']==61026468 and c['build_plan']==ref(planpath) and clock['plan_sha256']==sha(planpath);assert not (B/'materialized').exists();os.sched_setaffinity(0,{26,27});nice=os.getpriority(os.PRIO_PROCESS,0)
 if nice<10:os.nice(10-nice)
 pa.set_cpu_count(1);pa.set_io_thread_count(1);start=time.monotonic();O=B/'materialized';O.mkdir();(O/'shards/train').mkdir(parents=True);(O/'effective-groups').mkdir();(O/'code').mkdir()
 names=[Path(__file__).name,'data_intake_full_rnacentral_tasks.py','data_intake_full_rnacentral_state.py','data_intake_full_rnacentral_census.py','data_intake_rnacentral_prior.py','data_intake_nucleotide_prior.py','data_intake_corpus.py']
 for name in names:(O/'code'/name).open('xb').write((R/'scripts'/name).read_bytes())
 def guard(reserve=2**30):
  st=os.statvfs(R)
  if st.f_bavail*st.f_frsize-reserve<64*2**30:raise RuntimeError('hardware free buffer; retain partial outputs')
  if time.time()>clock['deadline_epoch']:raise RuntimeError('whole-build finite deadline; retain partial outputs')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30-128*2**20:raise RuntimeError('aggregate RSS guard; retain partial outputs')
 guard();statepath=check(c['native_identity_URS_index']);state=NativeIdentityIndex.open_readonly(statepath);state.db.execute('PRAGMA cache_size=-262144');indexpath=O/'task-identity-serialization-index.sqlite';db=sqlite3.connect(indexpath);db.executescript('PRAGMA cache_size=-262144;CREATE TABLE tasks(task_id BLOB PRIMARY KEY CHECK(length(task_id)=32),native_sha BLOB NOT NULL,source_file INTEGER NOT NULL,source_row INTEGER NOT NULL,reuse_kind TEXT NOT NULL,shard INTEGER,local_row INTEGER) WITHOUT ROWID;');db.commit();shards=[];lookup={};counts=collections.Counter();nativeunits=collections.Counter();minimum=None;maximum=0;buffer=[];metabuffer=[];bufferbytes=0;component_offset=collections.Counter()
 def flush():
  nonlocal bufferbytes
  if not buffer:return
  number=len(shards);p=O/'shards/train'/f'{number:06d}.parquet';g=O/'effective-groups'/f'{number:06d}.parquet';groups=[dict(task_id=r['task_id'],effective_groups_json=r['group_ids_json']) for r in buffer];pq.write_table(pa.Table.from_pylist(buffer),p,compression='zstd');pq.write_table(pa.Table.from_pylist(groups),g,compression='zstd');back=pq.read_table(p).to_pylist();assert back==buffer and pq.read_table(g).to_pylist()==groups
  for local,(row,meta) in enumerate(zip(back,metabuffer)):
   payload=row['native_payload'];assert hashlib.sha256(payload).hexdigest()==row['native_payload_sha256'];task=CorpusTask.from_bytes(payload);assert task.canonical_bytes()==payload;d=json.loads(payload);assert d['conditions']==dict(prior_domain='RNA_reference_native_symbols',source_organism=None) and len(d['targets']['source_native_sequence'])==row['full_axis_values'];assert hashlib.sha256(d['targets']['source_native_sequence'].encode()).digest()==meta['native_sha'];assert row['task_id']==meta['task_id'];db.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?,?)',(bytes.fromhex(row['task_id']),meta['native_sha'],meta['source_file'],meta['source_row'],'new',number,local))
  sh=ref(p,rows=len(back),source_family='RNAcentral-reference-priors',split='train');shards.append(sh);lookup[sh['path']]=ref(g,rows=len(groups));counts['new_task_shards']+=1;counts['new_task_views_serialized']+=len(back);buffer.clear();metabuffer.clear();bufferbytes=0;db.commit();guard()
 for file,f in enumerate(plan['source_files']):
  global_start=component_offset[f['component']];component_offset[f['component']]+=f['rows'];guard();selected={r['first_row']:r for r in state.source_rows(file) if state.db.execute('SELECT 1 FROM holdout_blocked WHERE native_sha=? LIMIT 1',(r['native_sha'],)).fetchone() is None};p=check(f);seen=0;matched=0
  for batch in pq.ParquetFile(p).iter_batches(batch_size=2048):
   for original in batch.to_pylist():
    row_index=seen;seen+=1;r=selected.get(row_index)
    if r is None:continue
    seq,native,upper,length,valid=validate_native_row(original,original_FASTA=f['original_FASTA'],expected_record_index=global_start+row_index);assert valid and native==r['native_sha'] and upper==r['uppercase_sha'];matched+=1;counts['eligible_distinct_native_sequences']+=1
    if r['reuse_kind']!='new':
     db.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?,?)',(bytes.fromhex(r['task_id']),native,file,row_index,r['reuse_kind'],None,None));counts['reused_'+r['reuse_kind']]+=1;continue
    sr=dict(native_payload=f['path'],native_payload_sha256=f['sha256'],native_row_index=row_index,original_record_index=original['source_record_index'],original_FASTA=original['source_payload'],source_identifier=original['source_identifier'],native_header=original['native_header'],source_status=f['component'],source_release='RNAcentral28.0',source_rights='CC0-1.0',original_source_record_id=original['source_record_id']);ids=state.identifiers(native);row=full_native_prior_envelope(seq,r,sr,ids);buffer.append(row);metabuffer.append(dict(task_id=r['task_id'],native_sha=native,source_file=file,source_row=row_index));bufferbytes+=len(row['native_payload']);nativeunits.update(native_symbol_units(seq));minimum=length if minimum is None else min(minimum,length);maximum=max(maximum,length)
    if len(buffer)>=1024 or bufferbytes>=8*2**20:flush()
   guard()
  assert seen==f['rows'] and matched==len(selected);db.commit();progress=dict(schema='mora3.FullRNAcentralMaterializationLiveCheckpoint/v1',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files_done=file+1,source_files_total=len(plan['source_files']),counts=dict(counts),native_units_not_model_tokens=dict(nativeunits),new_native_length_range=[minimum,maximum],elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,task_bytes=sum(x['bytes'] for x in shards),group_bytes=sum(x['bytes'] for x in lookup.values()),index_bytes=indexpath.stat().st_size);(B/'materialization-live-progress.json').write_text(json.dumps(progress,indent=2)+'\n');print('FULL_RNACENTRAL_MATERIALIZE',file+1,'newtasks',counts['new_task_views_serialized'],'reused',counts['eligible_distinct_native_sequences']-counts['new_task_views_serialized']-len(buffer),'seconds',round(time.monotonic()-start,1),flush=True)
 flush();guard();assert counts['new_task_views_serialized']==c['new_native_sequences_eligible_after_known_heldout_guards'];table_counts=dict(db.execute('SELECT reuse_kind,count(*) FROM tasks GROUP BY reuse_kind'));assert table_counts.get('new',0)==counts['new_task_views_serialized'] and sum(table_counts.values())==counts['eligible_distinct_native_sequences'];db.commit();db.close();state.close();assert ref(statepath)==c['native_identity_URS_index'];guard();result=dict(schema='mora3.FullRNAcentralNativeMaterializedComponent/v1',status='all_eligible_new_native_tasks_serialized_pending_independent_full_source_and_master_union',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),census=ref(cp),build_plan=ref(planpath),clock=ref(clockpath),phase_execution_approval=ref(approval),baseline_master=plan['baseline_master'],counts=dict(counts),task_index_counts=table_counts,task_identity_serialization_index=ref(indexpath),shards=shards,effective_group_lookup=lookup,task_shard_bytes=sum(f['bytes'] for f in shards),effective_group_bytes=sum(f['bytes'] for f in lookup.values()),native_units_not_model_tokens=dict(nativeunits),new_distinct_target_native_length_range=[minimum,maximum] if minimum is not None else None,all_new_task_full_byte_SHA_native_target_canonical_roundtrips=True,all_new_source_task_groups_include_complete_native_URS_ledger=True,preserved_old_Rfam_and_pilot_tasks_not_reserialized=True,preserved_old_effective_group_overlay_pending_union=True,source_native_values_not_model_tokens=True,actual_model_tokens=None,actual_optimizer_exposure=None,master_admitted_task_views_by_this_phase=0,model_adapter_supported=False,full_dataset_ready=False,substantive_training_authorized=False,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(O/'manifest.json',result);guard();print('COMPLETE_FULL_RNACENTRAL_MATERIALIZATION',json.dumps({k:result[k] for k in ['counts','task_shard_bytes','effective_group_bytes','seconds']}),flush=True)
if __name__=='__main__':main()
