"""Pinned master-wide exact GO-ID exposure and NON-APPLYING mixture proposals."""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,os,resource,sqlite3,sys,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_master_GO_scan import row_exposure,subject_scope
from data_intake_RNA_phase_coexistence import owned_phase_RSS,normal_materialization,normal_source_proof,PhaseCoexistenceError
B=R/'outputs/data-intake/20261101-ProCyon-native-corpus-v1';P=B/'master-wide-heldout-GO-audit-plan-v2.json'
SCOPE='master-wide-native-heldout-GO-ID-exposure-and-nonapplying-mixture-proposal-only'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while chunk:=f.read(2**20):h.update(chunk)
 return h.hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def checked(value):
 p=Path(value['path']);p=p if p.is_absolute() else R/p
 if p.stat().st_size!=value['bytes'] or sha(p)!=value['sha256']:raise ValueError('pinned input/code changed: '+str(p))
 return p
def approve(a,plan,script):
 if a.get('approved') is not True or a.get('scope')!=SCOPE:raise ValueError('exact nonapplying audit approval required')
 if a.get('plan_sha256')!=plan['sha256'] or a.get('script_sha256')!=script['sha256']:raise ValueError('exact plan/script approval binding required')
 if any(a.get(k) is not False for k in ['task_admission','master_publication','model_use','model_training','mixture_policy_activation']):raise ValueError('audit creates no new admission/model/activated mixture policy')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--approval',required=True);args=parser.parse_args();started=time.time();start=time.monotonic();os.sched_setaffinity(0,{26,27});os.nice(10);pa.set_cpu_count(1);pa.set_io_thread_count(1)
 plan=json.loads(P.read_text());ap=Path(args.approval);ap=ap if ap.is_absolute() else R/ap;approve(json.loads(ap.read_text()),ref(P),ref(Path(__file__).resolve()))
 O=R/plan['output_namespace']
 if O.exists():raise ValueError('fresh audit namespace required; no resume/reset/retry')
 for v in list(plan['inputs'].values())+plan['dependencies']:checked(v)
 O.mkdir();stage='initial';counts=collections.Counter();rollups=collections.defaultdict(collections.Counter);GOA_subject_rollups=collections.defaultdict(collections.Counter);guard_samples=[];guard_total=0;last_owner_check=0;last_saved=0;owner_state=[];other=0;completed={};last_snapshot={}
 def snapshot(owner):
  expected=owner['expected_process'];pid=expected['pid'];p=Path('/proc')/str(pid)
  try:
   stat=(p/'stat').read_text();tail=stat[stat.rfind(')')+2:].split();return dict(pid=pid,start_ticks=int(tail[19]),state=tail[0],argv=(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'),script_sha256=sha(R/owner['script_ref']['path']),CPU_affinity=sorted(os.sched_getaffinity(pid)),nice=os.getpriority(os.PRIO_PROCESS,pid),RSS_bytes=int((p/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'))
  except (FileNotFoundError,ProcessLookupError):return None
 def owner_completion(owner):
  key=owner['phase']
  if completed.get(key):return True
  if key=='materialize':
   ok=normal_materialization(owner['completed_manifest_summary'],checked(owner['completion_log']).read_text()[-8192:],owner['expected_completion_bindings']);c=json.loads(checked(owner['completion_verification']).read_text());ok=ok and c['normal_final_marker'] is True and c['manifest']==plan['inputs']['completed_materializer_component']
  else:
   rp=R/owner['completion_receipt_path'];lp=R/owner['completion_log_path'];ok=rp.exists() and lp.exists() and normal_source_proof(json.loads(rp.read_text()),lp.read_text()[-16384:],owner['expected_completion_bindings'])
  completed[key]=bool(ok);return completed[key]
 def guard(force=False):
  nonlocal last_owner_check,last_saved,other,owner_state,guard_total,last_snapshot
  now=time.monotonic();viol=[]
  if force or now-last_owner_check>=1:
   owner_state=[];other=0
   for owner in plan['RNA_resource_owners']:
    try:
     actual=snapshot(owner);rss,state=owned_phase_RSS(actual,owner['expected_process'],owner_completion(owner));other+=rss;owner_state.append(dict(phase=owner['phase'],state=state,process=actual,RSS_bytes=rss))
    except PhaseCoexistenceError as error:viol.append(str(error));owner_state.append(dict(phase=owner['phase'],state='unverified_identity_or_completion'))
   last_owner_check=now
  for path in plan['uncoordinated_future_RNA_phase_receipts']:
   if (R/path).exists():viol.append('uncoordinated_successor_RNA_phase:'+path)
  limit=plan['resources'];peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;st=os.statvfs(R);s=dict(stage=stage,elapsed_seconds=now-start,deadline_utc=datetime.datetime.fromtimestamp(started+limit['max_seconds'],datetime.timezone.utc).isoformat(),own_peak_RSS_bytes=peak,own_current_RSS_bytes=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),verified_other_owned_RNA_RSS_bytes=other,owned_RNA_states=owner_state,aggregate_with_RNA_and_wrapper_reserve_bytes=peak+other+limit['wrapper_reserve_bytes'],free_hardware_bytes=st.f_bavail*st.f_frsize)
  for key,threshold in [('elapsed_seconds',limit['max_seconds']),('own_peak_RSS_bytes',limit['max_own_peak_RSS_bytes']),('aggregate_with_RNA_and_wrapper_reserve_bytes',limit['aggregate_RSS_with_known_RNA_and_wrapper_limit_bytes'])]:
   if s[key]>threshold:viol.append(key+'>limit:'+str(threshold))
  if s['free_hardware_bytes']<limit['hardware_free_floor_bytes']:viol.append('free_hardware_bytes<floor:'+str(limit['hardware_free_floor_bytes']))
  s['threshold_failures']=viol;last_snapshot=s;guard_total+=1
  if force or viol or now-last_saved>=5:
   guard_samples.append(s);guard_samples[:]=guard_samples[-64:]
   with (O/'guard-samples.jsonl').open('a') as log:log.write(json.dumps(s)+'\n')
   last_saved=now
  if viol:
   (O/'guard-stop-diagnostics.json').open('x').write(json.dumps(s,indent=2)+'\n');raise RuntimeError('bounded audit guard: '+','.join(viol))
 def progress(shards):
  p=O/'live-progress.json';temporary=O/'live-progress.new.json';temporary.write_text(json.dumps(dict(stage=stage,shards_done=shards,counts=dict(counts),seconds=time.monotonic()-start,last_guard=last_snapshot),indent=2)+'\n');os.replace(temporary,p)
 try:
  guard(True);master=json.loads(checked(plan['inputs']['master']).read_text());protected=set(json.loads(checked(plan['inputs']['native_zero_shot_ID_union']).read_text())['GO_IDs'])
  if len(protected)!=529:raise ValueError('exact native protected-ID set changed')
  role_members=json.loads(checked(plan['inputs']['native_role_ID_sets']).read_text());assert role_members['native_official_heldouts_not_redefined'] is True
  db=sqlite3.connect(O/'native-heldout-GO-exposure-and-proposals.sqlite');db.executescript('PRAGMA cache_size=-32768;PRAGMA temp_store=FILE;CREATE TABLE role_exposure(source_family_json TEXT,native_schema TEXT,source_split TEXT,native_role TEXT,GO_ID TEXT,task_views INTEGER,PRIMARY KEY(source_family_json,native_schema,source_split,native_role,GO_ID));CREATE TABLE protected_task_views(task_id TEXT PRIMARY KEY,source_family_json TEXT,native_schema TEXT,source_split TEXT,literal_record_id TEXT,subject_scope TEXT,source_shard TEXT,source_shard_SHA TEXT,source_shard_row INTEGER,native_payload_SHA TEXT,protected_target INTEGER,protected_condition INTEGER,proposal_json TEXT);CREATE TABLE protected_GOA_annotation_contexts(task_id TEXT,annotation_id TEXT,literal_subject TEXT,subject_scope TEXT,root_GO_ID TEXT,root_GO_protected INTEGER,context_json TEXT,PRIMARY KEY(task_id,annotation_id));CREATE TABLE scanned_shards(path TEXT PRIMARY KEY,sha256 TEXT,bytes INTEGER,rows INTEGER,opening_and_closing_SHA_verified INTEGER);')
  overlay=(O/'NONAPPLYING-final-mixture-proposed-exclusions-and-target-units.jsonl').open('x')
  expected_schema=collections.Counter(master['native_schema_counts']);actual_schema=collections.Counter()
  for n,shard in enumerate(master['shards']):
   stage='master_native_shard:'+str(n);guard();p=checked(shard);pf=pq.ParquetFile(p)
   if pf.metadata.num_rows!=shard['rows']:raise ValueError('master shard footer count changed')
   local=0
   columns=['task_id','source_family','record_id','task_name','split','native_schema','native_payload','native_payload_sha256','admitted_for_task']
   for batch in pf.iter_batches(columns=columns,batch_size=16):
    updates=collections.Counter()
    for row in batch.to_pylist():
     if row['admitted_for_task'] is not True:raise ValueError('unadmitted row in pinned master')
     raw=row['native_payload']
     if hashlib.sha256(raw).hexdigest()!=row['native_payload_sha256']:raise ValueError('native canonical bytes hash changed')
     packet=json.loads(raw);family=row['source_family'];schema=row['native_schema'];split=row['split'];exposure=row_exposure(packet,schema,family,protected);key=json.dumps([family,schema,split]);roll=rollups[key];roll['task_views']+=1;actual_schema[schema]+=1;counts['all_admitted_master_task_views']+=1
     for role,IDs in exposure['GO_IDs_by_native_role'].items():
      if IDs:roll['views_with_any_GO_ID_'+role]+=1
      for go in IDs:updates[(json.dumps(family),schema,split,role,go)]+=1
     for field in ['protected_target_exposure','protected_condition_exposure','proposed_coarse_entire_view_exclusion']:
      roll[field+'_views']+=int(exposure[field]);counts[field+'_views']+=int(exposure[field])
     if family=='UniProt-GOA-assertion-packets':
      atoms=exposure['GOA_atomic_source_annotations'];roll['GOA_source_annotations_nonadditive_raw_typed']+=atoms;counts['GOA_source_annotations_nonadditive_raw_typed']+=atoms;counts['all_admitted_GOA_packet_views']+=1;subject_counts=GOA_subject_rollups[subject_scope(row['record_id'])];subject_counts['packet_views']+=1;subject_counts['atomic_source_annotations']+=atoms;subject_counts['protected_target_packet_views']+=int(exposure['protected_target_exposure']);subject_counts['protected_condition_packet_views']+=int(exposure['protected_condition_exposure'])
      selective=exposure['GOA_selective_proposal'];subject_counts['atomic_annotations_proposed_for_projection']+=selective['target_annotations_projected_out'];subject_counts['candidate_nonempty_views_after_projection']+=int(selective['retained_nonempty_view_candidate_only']);roll['GOA_atomic_annotations_proposed_for_projection']+=selective['target_annotations_projected_out'];counts['GOA_atomic_annotations_proposed_for_projection']+=selective['target_annotations_projected_out'];roll['GOA_nonempty_candidate_views_after_selective_projection']+=int(selective['retained_nonempty_view_candidate_only'])
     if exposure['proposed_coarse_entire_view_exclusion']:
      db.execute('INSERT INTO protected_task_views VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(row['task_id'],json.dumps(family),schema,split,row['record_id'],subject_scope(row['record_id']),shard['path'],shard['sha256'],local,row['native_payload_sha256'],int(exposure['protected_target_exposure']),int(exposure['protected_condition_exposure']),json.dumps(exposure)))
      proposal=dict(schema='mora3.NONAPPLYINGHeldoutGOOverlayProposal/v1',task_id=row['task_id'],native_source_shard=shard,shard_row=local,native_payload_SHA256=row['native_payload_sha256'],source_family=family,source_split=split,literal_record_id=row['record_id'],native_official_role_sets=plan['inputs']['native_role_ID_sets'],coarse_entire_view_exclusion_proposed=True,selective_atomic_GOA_source_annotation_IDs=exposure.get('GOA_selective_proposal',{}).get('removed_source_annotation_IDs',[]),protected_condition_requires_view_exclusion_or_separate_admission=exposure['protected_condition_exposure'],non_GOA_fine_projection_unreviewed=family!='UniProt-GOA-assertion-packets',proposal_only=True,policy_activation=False,task_admission=False,model_use=False)
      overlay.write(json.dumps(proposal)+'\n')
     for context in exposure.get('protected_GOA_atomic_annotations',[]):
      db.execute('INSERT INTO protected_GOA_annotation_contexts VALUES(?,?,?,?,?,?,?)',(row['task_id'],context['source_annotation_ID'],context['literal_subject'],context['subject_scope'],context['root_GO_ID'],int(context['root_GO_is_protected']),json.dumps(context)))
     local+=1
    db.executemany('INSERT INTO role_exposure VALUES(?,?,?,?,?,?) ON CONFLICT(source_family_json,native_schema,source_split,native_role,GO_ID) DO UPDATE SET task_views=task_views+excluded.task_views',[(list(k)+[v]) for k,v in updates.items()]);guard()
   if local!=shard['rows']:raise ValueError('full shard row conservation failed')
   checked(shard);db.execute('INSERT INTO scanned_shards VALUES(?,?,?,?,1)',(shard['path'],shard['sha256'],shard['bytes'],local));db.commit();counts['whole_shards_all_native_payloads_verified']+=1
   if n%100==0:progress(n+1);print('MASTER_HELDOUT_GO_AUDIT',n+1,counts['all_admitted_master_task_views'],flush=True)
  overlay.close()
  if counts['all_admitted_master_task_views']!=master['serialized_tasks'] or dict(actual_schema)!=dict(expected_schema):raise ValueError('full master/schema conservation failed')
  if counts['all_admitted_GOA_packet_views']!=plan['expected_GOA']['packet_views'] or counts['GOA_source_annotations_nonadditive_raw_typed']!=plan['expected_GOA']['source_annotations']:raise ValueError('complete GOA packet/annotation conservation failed')
  assert db.execute('SELECT sum(rows) FROM scanned_shards').fetchone()[0]==master['serialized_tasks']
  assert db.execute('SELECT count(*) FROM protected_task_views').fetchone()[0]==counts['proposed_coarse_entire_view_exclusion_views']
  summaries=[dict(source_family=json.loads(k)[0],native_schema=json.loads(k)[1],source_split=json.loads(k)[2],counts=dict(v)) for k,v in sorted(rollups.items())]
  (O/'family-role-and-proposal-summary.json').open('x').write(json.dumps(dict(per_family_schema_original_split=summaries,GOA_subject_syntax_scopes={k:dict(v) for k,v in GOA_subject_rollups.items()},all_counts=dict(counts),official_evaluation_ID_sets_not_redefined=True,proposal_activation=False,exact_pair_exposure_separate_prior=plan['inputs']['completed_exact_pair_audit'],actual_optimizer_exposure=None),indent=2)+'\n');db.close()
  for v in list(plan['inputs'].values())+plan['dependencies']:checked(v)
  stage='complete';guard(True);progress(len(master['shards']))
  receipt=dict(schema='mora3.MasterWideNativeHeldoutGOExposureProposalAudit/v1',status='complete_exact_native_ID_namespace_exposure_and_nonapplying_policy_proposal',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=ref(P),approval=ref(ap),inputs=plan['inputs'],counts=dict(counts),all_native_schema_counts=dict(actual_schema),per_family_schema_original_split=summaries,GOA_subject_syntax_scopes={k:dict(v) for k,v in GOA_subject_rollups.items()},exposure_index=ref(O/'native-heldout-GO-exposure-and-proposals.sqlite'),NONAPPLYING_overlay_proposal=ref(O/'NONAPPLYING-final-mixture-proposed-exclusions-and-target-units.jsonl'),family_summary=ref(O/'family-role-and-proposal-summary.json'),guard_log=ref(O/'guard-samples.jsonl'),guard_checks=guard_total,last_64_guard_samples=guard_samples,scope_limits=plan['scope_limits'],task_admission=False,master_publication=False,mixture_policy_activation=False,model_use=False,actual_model_optimizer_exposure=None,full_dataset_ready=False,model_training=False,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  (O/'manifest.json').open('x').write(json.dumps(receipt,indent=2)+'\n');print('COMPLETE_MASTER_NATIVE_HELDOUT_GO_AUDIT',json.dumps(dict(counts=dict(counts),seconds=receipt['seconds'])),flush=True)
 except Exception as error:
  (O/'failure.json').open('x').write(json.dumps(dict(stage=stage,error_type=type(error).__name__,error=str(error),counts=dict(counts),last_guard=last_snapshot,seconds=time.monotonic()-start,no_retry_or_relaxation=True),indent=2)+'\n');raise
if __name__=='__main__':main()
