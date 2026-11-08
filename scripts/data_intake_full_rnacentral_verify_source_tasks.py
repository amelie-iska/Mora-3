"""Independent ALL-source/ALL-new-task RNA proof; does not publish a master.

Requires a separately reviewed materialization and exact phase approval. Uses
new bounded-memory proof indexes; every original/census/source index is read-only.
"""
from pathlib import Path
import argparse,collections,datetime,functools,hashlib,json,os,resource,sqlite3,sys,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_full_rnacentral_state import NativeIdentityIndex,old_identity_resolver
from data_intake_full_rnacentral_replay_index import NativeReplayIndex
from data_intake_full_rnacentral_proof_helpers import verify_native_task
from data_intake_nucleotide_prior import sequence_groups
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
 if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('immutable source hash/size mismatch')
 return p
def write(p,v):p.open('x').write(json.dumps(v,indent=2)+'\n')
def validate_source_task_proof_approval(a,expected):
 if a.get('approved') is not True or a.get('phase')!='verify-source-tasks' or a.get('scope')!='whole-native-RNAcentral-source-task-proof-only':raise ValueError('separate full source/task proof approval required')
 for k,v in expected.items():
  if a.get(k)!=v:raise ValueError('source/task proof review mismatch: '+k)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--approval',type=Path,required=True);args=ap.parse_args();approval=args.approval if args.approval.is_absolute() else R/args.approval;pp=B/'build-plan.json';plan=json.loads(pp.read_text());cp=B/'census/manifest.json';c=json.loads(cp.read_text());mp=B/'materialized/manifest.json';m=json.loads(mp.read_text());clockpath=B/'build-clock.json';clock=json.loads(clockpath.read_text());names=[Path(__file__).name,'data_intake_full_rnacentral_replay_index.py','data_intake_full_rnacentral_proof_helpers.py','data_intake_full_rnacentral_state.py','data_intake_rnacentral_prior.py','data_intake_nucleotide_prior.py','data_intake_corpus.py'];expected=dict(plan_sha256=sha(pp),census_manifest_sha256=sha(cp),component_manifest_sha256=sha(mp),clock_sha256=sha(clockpath),script_sha256=sha(Path(__file__)),resource_coordination_sha256=plan['resource_coordination']['sha256'],dependency_refs=[ref(R/'scripts'/n) for n in names[1:]]);validate_source_task_proof_approval(json.loads(approval.read_text()),expected);assert c['status']=='complete_full_source_census_pending_materialization_and_independent_proof' and m['status']=='all_eligible_new_native_tasks_serialized_pending_independent_full_source_and_master_union';assert m['census']==ref(cp) and m['build_plan']==ref(pp) and m['clock']==ref(clockpath) and clock['plan_sha256']==sha(pp);O=B/'independent-source-task-proof-v1';assert not O.exists();os.sched_setaffinity(0,{26,27});nice=os.getpriority(os.PRIO_PROCESS,0)
 if nice<10:os.nice(10-nice)
 pa.set_cpu_count(1);pa.set_io_thread_count(1);O.mkdir();(O/'code').mkdir();(O/'source-checkpoints').mkdir()
 for n in names:(O/'code'/n).open('xb').write((R/'scripts'/n).read_bytes())
 start=time.monotonic()
 def guard(reserve=2**30):
  st=os.statvfs(R)
  if st.f_bavail*st.f_frsize-reserve<64*2**30 or time.time()>clock['deadline_epoch'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30-128*2**20:raise RuntimeError('whole deadline/hardware/RSS guard; retain partial proof')
 guard();sp=check(c['native_identity_URS_index']);state=NativeIdentityIndex.open_readonly(sp);state.db.execute('PRAGMA cache_size=-131072');ip=check(m['task_identity_serialization_index']);tasks=sqlite3.connect('file:'+str(ip)+'?mode=ro&immutable=1',uri=True);tasks.row_factory=sqlite3.Row;rfam_path=check(plan['old_Rfam_index']);pilot_path=check(plan['old_pilot_index']);rfam=sqlite3.connect('file:'+str(rfam_path)+'?mode=ro&immutable=1',uri=True);pilot=sqlite3.connect('file:'+str(pilot_path)+'?mode=ro&immutable=1',uri=True);resolver=old_identity_resolver(rfam,pilot);proof=NativeReplayIndex(O/'literal-source-replay-index.sqlite');held=set(json.loads(check(plan['heldout_group_snapshot']).read_text())['group_ids']);counts=collections.Counter();source_letters=0;nextrecord=collections.Counter();distinct_letters=0;blocked_first=0;ledgers={f['source_file_ordinal']:f for f in c['occurrence_ledgers']};assert len(ledgers)==len(plan['source_files'])
 for file,f in enumerate(plan['source_files']):
  guard();p=check(f);ledger=pq.read_table(check(ledgers[file])).to_pylist();assert len(ledger)==f['rows'];local=0
  for batch in pq.ParquetFile(p).iter_batches(batch_size=2048):
   for source in batch.to_pylist():
    r=ledger[local];seq=source['native_sequence'];native=hashlib.sha256(seq.encode()).digest();upper=hashlib.sha256(seq.upper().encode()).digest();valid=bool(seq) and not(set(seq)-ALPHABET);assert source['source_record_index']==nextrecord[f['component']] and source['source_payload']==f['original_FASTA'] and source['source_identifier']==source['native_header'].split()[0];assert native.hex()==source['native_sequence_sha256'] and upper.hex()==source['uppercase_sequence_sha256'] and len(seq)==source['native_length'] and valid==source['sequence_alphabet_pass'];assert source['native_T_count']==seq.upper().count('T') and source['native_U_count']==seq.upper().count('U');assert r['source_file_ordinal']==file and r['native_row_index']==local and r['original_record_index']==nextrecord[f['component']] and r['source_identifier']==source['source_identifier'] and r['native_sha']==native and r['uppercase_sha']==upper and r['native_length']==len(seq) and r['source_alphabet_pass']==valid;nextrecord[f['component']]+=1;counts['source_occurrences']+=1;source_letters+=len(seq)
    if valid:
     g=sequence_groups(seq);s=state.lookup(native);assert s is not None and s['uppercase_sha']==upper and s['rc_sha'].hex()==g['reverse_complement_base_order_group'];groups={'nucleotide_native:'+native.hex(),'nucleotide_upper:'+upper.hex(),'nucleotide_derived_RC_baseorder:'+g['reverse_complement_base_order_group'],'rnacentral_native_identifier:'+source['source_identifier']};matches=groups&held;first=proof.observe(native,source['source_identifier'],file,local,matches);assert first==r['first_native_occurrence']==(s['first_file']==file and s['first_row']==local);assert r['rc_guard_sha']==s['rc_sha'] and r['task_identity_digest'].hex()==s['task_id'] and r['reuse_kind']==s['reuse_kind'] and r['local_existing_heldout_group_match']==bool(matches) and r['nonstandard_symbols'] is None
     if first:
      old=resolver(seq);kind,tid=old if old is not None else ('new',s['task_id']);assert kind==s['reuse_kind'] and tid==s['task_id'];counts['first_'+kind]+=1;counts['distinct_native_sequences']+=1;distinct_letters+=len(seq)
     counts['quality_pass_source_occurrences']+=1
    else:
     assert r['task_identity_digest'] is None and r['rc_guard_sha'] is None and r['reuse_kind'] is None and not r['first_native_occurrence'] and not r['local_existing_heldout_group_match'] and r['nonstandard_symbols']==json.dumps(sorted(set(seq)-ALPHABET));counts['source_nonstandard_quarantines']+=1
    local+=1
   guard()
  assert local==f['rows'];proof.commit();progress=dict(source_files_done=file+1,source_files_total=len(plan['source_files']),counts=dict(counts),source_native_letters=source_letters,distinct_native_reference_letters=distinct_letters,elapsed_seconds=time.monotonic()-start);write(O/'source-checkpoints'/f'{file:04d}.json',progress);(B/'source-proof-live-progress.json').write_text(json.dumps(progress,indent=2)+'\n');print('FULL_RNA_SOURCE_PROOF',file+1,'rows',counts['source_occurrences'],'seconds',round(time.monotonic()-start,1),flush=True)
 assert counts['source_occurrences']==61026468 and counts['quality_pass_source_occurrences']==61006261 and counts['source_nonstandard_quarantines']==20207 and source_letters==47369240265;differences=proof.compare_readonly_census(sp);assert not any(differences.values());assert proof.summary()['native_sequences']==counts['distinct_native_sequences'];proof_summary=proof.summary();proof.close();rfam.close();pilot.close();assert ref(rfam_path)==plan['old_Rfam_index'] and ref(pilot_path)==plan['old_pilot_index'];source_receipt=dict(status='verified',counts=dict(counts),source_native_letters=source_letters,distinct_source_reference_symbols_not_additive_model_exposure=distinct_letters,census_exact_index_differences=differences,proof_index_summary=proof_summary,proof_index=ref(O/'literal-source-replay-index.sqlite'),all_source_rows_hashes_native_axes_order_original_refs_and_quarantines=True,all_literal_URS_multiplicities_first_refs_and_known_heldout_blocks=True,all_old_prior_recipe_classifications_recomputed=True);write(O/'source-replay-receipt.json',source_receipt)
 @functools.lru_cache(maxsize=2)
 def native_rows(file):return pq.read_table(check(plan['source_files'][file])).to_pylist()
 task_count=0;task_bytes=group_bytes=0;length_min=None;length_max=0;target_letters=0;taskpaths=set();source_task_groups_intersect_held=0
 for shard_index,f in enumerate(m['shards']):
  guard();assert f['path'] not in taskpaths;taskpaths.add(f['path']);p=check(f);gp=check(m['effective_group_lookup'][f['path']]);rows=pq.read_table(p).to_pylist();groups=pq.read_table(gp).to_pylist();assert len(rows)==len(groups)==f['rows'];task_bytes+=p.stat().st_size;group_bytes+=gp.stat().st_size
  for local,(row,group) in enumerate(zip(rows,groups)):
   idx=tasks.execute('SELECT * FROM tasks WHERE task_id=?',(bytes.fromhex(row['task_id']),)).fetchone();assert idx is not None;idx=dict(idx);assert idx['reuse_kind']=='new' and idx['shard']==shard_index and idx['local_row']==local;s=state.lookup(idx['native_sha']);source=native_rows(idx['source_file'])[idx['source_row']];v=verify_native_task(row,source,plan['source_files'][idx['source_file']],idx['source_file'],idx['source_row'],s,idx,state.identifiers(idx['native_sha']),group);values=json.loads(group['effective_groups_json']);source_task_groups_intersect_held+=bool(set(values)&held);task_count+=1;target_letters+=v['native_length'];length_min=v['native_length'] if length_min is None else min(length_min,v['native_length']);length_max=max(length_max,v['native_length'])
  if shard_index%100==0:print('FULL_RNA_NEW_TASK_PROOF',shard_index+1,'tasks',task_count,'seconds',round(time.monotonic()-start,1),flush=True)
  guard()
 assert task_count==m['counts']['new_task_views_serialized']==c['new_native_sequences_eligible_after_known_heldout_guards'] and task_bytes==m['task_shard_bytes'] and group_bytes==m['effective_group_bytes'] and source_task_groups_intersect_held==0;assert tasks.execute('SELECT count(*) FROM tasks WHERE reuse_kind=?',('new',)).fetchone()[0]==task_count;assert target_letters==m['native_units_not_model_tokens']['new_distinct_target_native_symbols'];tasks.close();state.close();assert ref(sp)==c['native_identity_URS_index'] and ref(ip)==m['task_identity_serialization_index'];guard();receipt=dict(schema='mora3.FullRNAcentralIndependentSourceAndTaskProof/v1',status='verified_source_and_new_tasks_pending_inherited_master_group_union',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),build_plan=ref(pp),census=ref(cp),component=ref(mp),baseline_master=plan['baseline_master'],clock=ref(clockpath),phase_execution_approval=ref(approval),source_replay=ref(O/'source-replay-receipt.json'),all_source_occurrences=counts['source_occurrences'],source_alphabet_pass=counts['quality_pass_source_occurrences'],source_quarantines=counts['source_nonstandard_quarantines'],all_new_task_payload_source_target_ID_location_axis_group_readback=task_count,new_task_bytes=task_bytes,new_effective_group_bytes=group_bytes,new_distinct_target_native_symbols=target_letters,new_native_length_range=[length_min,length_max] if length_min is not None else None,newtrain_known_heldout_group_intersections=source_task_groups_intersect_held,old_indices_and_completed_census_and_materializer_index_unchanged=True,no_master_publication_or_new_global_admission_by_this_phase=True,neural_tokens=None,actual_optimizer_use=None,full_dataset_ready=False,substantive_training_authorized=False,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(O/'receipt.json',receipt);guard();print('COMPLETE_FULL_RNA_SOURCE_TASK_PROOF',json.dumps(receipt),flush=True)
if __name__=='__main__':main()
