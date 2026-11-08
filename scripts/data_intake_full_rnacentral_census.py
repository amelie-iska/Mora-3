"""Full immutable RNAcentral census, exact prior reuse and source/URS provenance.

No tasks/master are admitted by this phase. No source or old index is modified.
"""
from pathlib import Path
import collections,datetime,hashlib,json,os,resource,sqlite3,sys,time
import pyarrow as pa
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_full_rnacentral_state import NativeIdentityIndex,old_identity_resolver
from data_intake_rnacentral_prior import ALPHABET
B=R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while x:=f.read(4*2**20):h.update(x)
 return h.hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p
 if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('immutable source size/hash mismatch: '+str(p))
 return p
def write(p,v):p.open('x').write(json.dumps(v,indent=2)+'\n')
def usage():return sum(p.stat().st_size for p in B.rglob('*') if p.is_file())
def validate_native_row(row, *, original_FASTA, expected_record_index):
 seq=row['native_sequence'];native=hashlib.sha256(seq.encode()).digest();upper=hashlib.sha256(seq.upper().encode()).digest();length=len(seq)
 if native.hex()!=row['native_sequence_sha256'] or upper.hex()!=row['uppercase_sequence_sha256'] or length!=row['native_length']:
  raise ValueError('native sequence hash/full-axis conservation mismatch')
 if row['source_identifier']!=row['native_header'].split()[0] or row['source_record_index']!=expected_record_index or row['source_payload']!=original_FASTA:
  raise ValueError('native source identifier/order/original provenance mismatch')
 valid=bool(seq) and not(set(seq)-ALPHABET)
 if valid!=row['sequence_alphabet_pass'] or row['native_T_count']!=seq.upper().count('T') or row['native_U_count']!=seq.upper().count('U'):
  raise ValueError('source alphabet/T-U precision mismatch')
 return seq,native,upper,length,valid

def main():
 plan=json.loads((B/'build-plan.json').read_text());assert plan['schema']=='mora3.FullRNAcentralConstructionPlan/v1';assert plan['source_occurrences']==61026468 and plan['source_alphabet_pass']==61006261;os.sched_setaffinity(0,set(plan['CPU_affinity']));os.nice(10);pa.set_cpu_count(1);pa.set_io_thread_count(1);O=B/'census';assert not O.exists();O.mkdir();(O/'occurrences').mkdir();(O/'identifier-conflicts').mkdir();(O/'code').mkdir()
 for n in [Path(__file__).name,'data_intake_full_rnacentral_state.py','data_intake_rnacentral_prior.py','data_intake_nucleotide_prior.py']:(O/'code'/n).open('xb').write((R/'scripts'/n).read_bytes())
 start=time.monotonic();clock=json.loads((B/'build-clock.json').read_text());assert clock['plan_sha256']==sha(B/'build-plan.json');deadline=clock['deadline_epoch'];floor=plan['free_buffer_bytes'];maxrss=plan['max_peak_RSS_bytes']-128*2**20
 def guard(reserve=512*2**20):
  st=os.statvfs(R)
  if st.f_bavail*st.f_frsize-reserve<floor:raise RuntimeError('hardware free-space buffer; preserve partials')
  if time.time()>deadline:raise RuntimeError('whole-build finite deadline; preserve partials')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>maxrss:raise RuntimeError('RSS resource guard; preserve partials')
 guard();rfam=check(plan['old_Rfam_index']);pilot=check(plan['old_pilot_index']);old=sqlite3.connect('file:'+str(rfam)+'?mode=ro&immutable=1',uri=True);pi=sqlite3.connect('file:'+str(pilot)+'?mode=ro&immutable=1',uri=True);resolve=old_identity_resolver(old,pi);state=NativeIdentityIndex.create(O/'native-identity-URS.sqlite');state.db.execute('CREATE TABLE holdout_blocked(native_sha BLOB NOT NULL,group_id TEXT NOT NULL,PRIMARY KEY(native_sha,group_id)) WITHOUT ROWID');state.commit();held=set(json.loads(check(plan['heldout_group_snapshot']).read_text())['group_ids']);counts=collections.Counter();per=collections.defaultdict(collections.Counter);nextrecord=collections.Counter();ledgers=[];files=[];lo=None;hi=0;letters=longer=0
 schema=pa.schema([('source_file_ordinal',pa.int32()),('native_row_index',pa.int32()),('original_record_index',pa.int64()),('source_identifier',pa.string()),('native_sha',pa.binary(32)),('uppercase_sha',pa.binary(32)),('rc_guard_sha',pa.binary(32)),('native_length',pa.int64()),('source_alphabet_pass',pa.bool_()),('first_native_occurrence',pa.bool_()),('reuse_kind',pa.string()),('task_identity_digest',pa.binary(32)),('local_existing_heldout_group_match',pa.bool_()),('nonstandard_symbols',pa.string())])
 for file,source in enumerate(plan['source_files']):
  guard();p=check(source);pf=pq.ParquetFile(p);assert pf.metadata.num_rows==source['rows'];component=source['component'];ledger=[];file_rows=0
  for batch in pf.iter_batches(batch_size=2048):
   for row in batch.to_pylist():
    seq,native,upper,length,valid=validate_native_row(row,original_FASTA=source['original_FASTA'],expected_record_index=nextrecord[component]);counts['source_occurrences']+=1;per[component]['source_occurrences']+=1;nextrecord[component]+=1;letters+=length;lo=length if lo is None else min(lo,length);hi=max(hi,length);longer+=length>8192
    if valid:
     x=state.observe(seq,row['source_identifier'],file,file_rows,resolver=resolve);assert x['native_sha']==native and x['uppercase_sha']==upper;groups={'nucleotide_native:'+native.hex(),'nucleotide_upper:'+upper.hex(),'nucleotide_derived_RC_baseorder:'+x['rc_sha'].hex(),'rnacentral_native_identifier:'+row['source_identifier']};matches=groups&held
     for group in matches:state.db.execute('INSERT OR IGNORE INTO holdout_blocked VALUES (?,?)',(native,group))
     counts['quality_pass_source_occurrences']+=1;per[component]['quality_pass_source_occurrences']+=1;counts['first_native_occurrences']+=x['first_occurrence'];counts['local_heldout_group_matched_occurrences']+=bool(matches)
     if x['first_occurrence']:counts['first_'+x['reuse_kind']]+=1
     ledger.append(dict(source_file_ordinal=file,native_row_index=file_rows,original_record_index=row['source_record_index'],source_identifier=row['source_identifier'],native_sha=native,uppercase_sha=upper,rc_guard_sha=x['rc_sha'],native_length=length,source_alphabet_pass=True,first_native_occurrence=x['first_occurrence'],reuse_kind=x['reuse_kind'],task_identity_digest=bytes.fromhex(x['task_id']),local_existing_heldout_group_match=bool(matches),nonstandard_symbols=None))
    else:
     counts['source_nonstandard_quarantines']+=1;per[component]['source_nonstandard_quarantines']+=1;ledger.append(dict(source_file_ordinal=file,native_row_index=file_rows,original_record_index=row['source_record_index'],source_identifier=row['source_identifier'],native_sha=native,uppercase_sha=upper,rc_guard_sha=None,native_length=length,source_alphabet_pass=False,first_native_occurrence=False,reuse_kind=None,task_identity_digest=None,local_existing_heldout_group_match=False,nonstandard_symbols=json.dumps(sorted(set(seq)-ALPHABET))))
    file_rows+=1
   guard()
  assert file_rows==source['rows'];target=O/'occurrences'/f'{file:04d}.parquet';table=pa.Table.from_pylist(ledger,schema=schema);pq.write_table(table,target,compression='zstd');assert pq.read_table(target).to_pylist()==ledger;state.commit();lr=ref(target,rows=file_rows,source_file_ordinal=file);ledgers.append(lr);files.append(dict(source_file_ordinal=file,source=source,occurrences=lr))
  checkpoint=dict(schema='mora3.FullRNAcentralCensusLiveCheckpoint/v1',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files_done=file+1,source_files_total=len(plan['source_files']),counts=dict(counts),source_native_letters=letters,native_length_range=[lo,hi],source_longer_than8192=longer,elapsed_seconds=time.monotonic()-start,derived_bytes=usage(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  (B/'live-progress.json').write_text(json.dumps(checkpoint,indent=2)+'\n');write(O/f'file-{file:04d}-receipt.json',checkpoint);print('FULL_RNACENTRAL_CENSUS',file+1,'/',len(plan['source_files']),'rows',counts['source_occurrences'],'new',counts['first_new'],'seconds',round(time.monotonic()-start,1),flush=True);guard()
 assert counts['source_occurrences']==plan['source_occurrences'] and counts['quality_pass_source_occurrences']==plan['source_alphabet_pass'] and counts['source_nonstandard_quarantines']==20207 and letters==47369240265 and longer==280916 and hi==991425
 summary=state.summary();assert summary['source_occurrences']==counts['quality_pass_source_occurrences'] and summary['native_sequences']==counts['first_native_occurrences'];blocked=state.db.execute('SELECT count(DISTINCT native_sha) FROM holdout_blocked').fetchone()[0];eligible=state.db.execute('SELECT count(*) FROM sequences s WHERE reuse_kind=? AND NOT EXISTS (SELECT 1 FROM holdout_blocked b WHERE b.native_sha=s.native_sha)',('new',)).fetchone()[0];conflicts=[];conflict_files=[];conflict_count=0
 for identifier,n in state.identifier_conflicts():
  conflicts.append(dict(source_identifier=identifier,distinct_literal_native_sequences=n));conflict_count+=1
  if len(conflicts)>=4096:
   p=O/'identifier-conflicts'/f'{len(conflict_files):04d}.parquet';pq.write_table(pa.Table.from_pylist(conflicts),p,compression='zstd');assert pq.read_table(p).to_pylist()==conflicts;conflict_files.append(ref(p,rows=len(conflicts)));conflicts.clear();guard()
 if conflicts:
  p=O/'identifier-conflicts'/f'{len(conflict_files):04d}.parquet';pq.write_table(pa.Table.from_pylist(conflicts),p,compression='zstd');assert pq.read_table(p).to_pylist()==conflicts;conflict_files.append(ref(p,rows=len(conflicts)))
 state.commit();state.close();old.close();pi.close();assert ref(rfam)==plan['old_Rfam_index'] and ref(pilot)==plan['old_pilot_index'];guard()
 result=dict(schema='mora3.FullRNAcentralNativeCensus/v1',status='complete_full_source_census_pending_materialization_and_independent_proof',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),build_plan=ref(B/'build-plan.json'),source_handoff=plan['source_handoff'],baseline_master=plan['baseline_master'],source_counts=dict(counts),per_component={k:dict(v) for k,v in per.items()},identity_summary=summary,all_source_payload_SHA_checked_and_all_columns_decoded=True,native_axis_hash_header_order_verified=True,source_native_letters=letters,source_longer_than8192=longer,native_length_range=[lo,hi],source_file_dictionary=files,occurrence_ledgers=ledgers,native_identity_URS_index=ref(O/'native-identity-URS.sqlite'),old_readonly_indices_unchanged=True,existing_heldout_snapshot=plan['heldout_group_snapshot'],distinct_native_sequences_with_known_heldout_group=blocked,new_native_sequences_eligible_after_known_heldout_guards=eligible,source_identifier_multiple_native_sequence_groups=conflict_count,identifier_conflict_files=conflict_files,published_release_count_discrepancy=plan['published_release_count_discrepancy'],task_views_serialized_by_phase=0,new_master_admissions=0,model_tokens=None,substantive_training_authorized=False,source_status_not_negative=True,uppercase_T_U_RC_not_identity=True,source_missing_modification_material_family_taxon_context_not_fabricated=True,seconds=time.monotonic()-start,derived_bytes=usage(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 write(O/'manifest.json',result);guard();print('COMPLETE_FULL_RNACENTRAL_CENSUS',json.dumps({k:result[k] for k in ['identity_summary','new_native_sequences_eligible_after_known_heldout_guards','seconds','derived_bytes']}),flush=True)
if __name__=='__main__':main()
