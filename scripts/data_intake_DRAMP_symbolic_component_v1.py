"""Finite candidate-only DRAMP110 writer. Execution requires a separately sealed supplement."""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,os,resource,sys,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_DRAMP_component_logic_v1 import project_row,group_proposal,verify_candidate
SCOPE='DRAMP110-source-symbolic-candidate-component-only'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while c:=f.read(2**20):h.update(c)
 return h.hexdigest()
def ref(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p))
def checked(v):
 p=Path(v['path']);p=p if p.is_absolute() else R/p
 if p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:raise ValueError('pinned immutable input/code changed: '+str(p))
 return p
def validate_approval(a,supplement,script,parent):
 if a.get('approved') is not True or a.get('scope')!=SCOPE:raise ValueError('separate exact candidate-component approval required')
 for k,v in [('supplement_sha256',supplement),('script_sha256',script),('parent_plan_sha256',parent)]:
  if a.get(k)!=v:raise ValueError('candidate component exact binding mismatch: '+k)
 for k in ['physical_ADMET','task_admission','master_publication','model_use','model_training','full_dataset_ready']:
  if a.get(k) is not False:raise ValueError('candidate component cannot open scientific/model/master gates')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--supplement',required=True);ap.add_argument('--approval',required=True);args=ap.parse_args();sp=R/args.supplement;approval=R/args.approval;s=json.loads(sp.read_text());validate_approval(json.loads(approval.read_text()),sha(sp),sha(Path(__file__).resolve()),s['parent_plan']['sha256']);started=time.time();start=time.monotonic();os.sched_setaffinity(0,{27});nice=os.getpriority(os.PRIO_PROCESS,0)
 if nice<10:os.nice(10-nice)
 pa.set_cpu_count(1);pa.set_io_thread_count(1)
 for v in [s['parent_plan'],*s['inputs'].values(),*s['code_refs']]:checked(v)
 O=R/s['output_namespace']
 if O.exists():raise ValueError('fresh unexecuted output namespace required')
 # This intentionally requires sequential, root-attested quiescence. It does
 # not guess a live C1/GO/C2 successor or silently acquire shared ownership.
 for pid in s['quiescence']['old_owned_PIDs']:
  if (Path('/proc')/str(pid)).exists():raise ValueError('old/reused owned PID requires fresh root recovery review')
 q=json.loads(checked(s['quiescence']['root_attestation']).read_text())
 if q.get('verified_no_live_constructor_phase') is not True:raise ValueError('new exact quiescence attestation required')
 O.mkdir();stage='initial';counts=collections.Counter();samples=[]
 stat=Path('/proc/self/stat').read_text();tail=stat[stat.rfind(')')+2:].split();identity=dict(pid=os.getpid(),start_ticks=int(tail[19]),SID=os.getsid(0),PGID=os.getpgid(0),argv=sys.argv,CPU_affinity=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0),script=ref(Path(__file__).resolve()))
 (O/'worker-identity.json').open('x').write(json.dumps(identity,indent=2)+'\n')
 def guard():
  st=os.statvfs(R);mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k=='MemAvailable'};peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;fail=[]
  for pid in s['quiescence']['old_owned_PIDs']:
   if (Path('/proc')/str(pid)).exists():fail.append('owned_PID_reappeared_or_reused:'+str(pid))
  for v in s['quiescence']['closed_receipt_refs']:checked(v)
  progress=json.loads((R/'planning/data_intake_progress.json').read_text())
  for job in progress.get('running_jobs',[]):
   pid=job.get('python_worker_pid')
   if pid is not None and pid!=os.getpid() and job.get('live') is True and (Path('/proc')/str(pid)).exists():fail.append('uncoordinated_live_constructor_phase:'+str(pid))
  for path in s['quiescence']['must_remain_absent_paths']:
   if (R/path).exists():fail.append('uncoordinated_future_phase:'+path)
  if time.monotonic()-start>120:fail.append('elapsed_seconds>120')
  if peak>256*2**20:fail.append('own_peak_RSS>256MiB')
  if st.f_bavail*st.f_frsize<64*2**30:fail.append('hardware_free<64GiB')
  if mem['MemAvailable']<8*2**30:fail.append('host_MemAvailable<8GiB')
  d=dict(stage=stage,elapsed_seconds=time.monotonic()-start,deadline_UTC=datetime.datetime.fromtimestamp(started+120,datetime.timezone.utc).isoformat(),own_peak_RSS_bytes=peak,free_hardware_bytes=st.f_bavail*st.f_frsize,host_MemAvailable_bytes=mem['MemAvailable'],other_constructor_RSS=0,other_constructor_scope='only valid under pinned root verified quiescence;newphasesstopthiswriter',threshold_failures=fail);samples.append(d)
  with (O/'guard-samples.jsonl').open('a') as log:log.write(json.dumps(d)+'\n')
  if fail:
   (O/'guard-stop.json').open('x').write(json.dumps(d,indent=2)+'\n');raise RuntimeError('candidate component resource/owner guard: '+','.join(fail))
 try:
  guard();prep=json.loads(checked(s['inputs']['native_preparation']).read_text());payload=checked(s['inputs']['prepared_payload']);role=json.loads(checked(s['inputs']['actual_role_review']).read_text());ctx=json.loads(checked(role['context_review']).read_text());assert prep['source_native_stability_rows']==role['source_rows']==110 and not role['candidate_builder_imported_or_executed'];assert role['source_payload']==s['inputs']['prepared_payload'];rows=pq.read_table(payload).to_pylist();assert len(rows)==110 and len({r['native_source_row'] for r in rows})==110;source_rows={r['native_source_row']:r for r in rows};dispositions=[];candidates=[];groups=[]
  for row in rows:
   stage='native_source_row:'+str(row['native_source_row']);guard();d=json.loads(row['full_source_named_cells_json'])
   for key in ['Type','Assay']:
    entries=ctx['source_field_roles'][key][d[key]['decoded_value']]['rows'];matches=[x for x in entries if x['native_source_row']==row['native_source_row'] and x['DRAMP_ID']==row['source_native_DRAMP_ID']]
    if len(matches)!=1:raise ValueError('actual110source context role/source row review mismatch')
   disposition,views=project_row(row,dict(native_preparation=s['inputs']['native_preparation'],prepared_payload=s['inputs']['prepared_payload']));dispositions.append(disposition);candidates.extend(views);groups.append(group_proposal(row));counts['source_rows']+=1;counts['source_report_occurrences']+=disposition['source_report_occurrences'];counts['candidate_views']+=len(views)
   for c in views:counts['candidate_views:'+c['task_name']]+=1
  assert len(dispositions)==len(groups)==110 and len(candidates)<=220
  source_schema=pa.schema([('source_native_ID',pa.string()),('source_row',pa.int64()),('disposition_json',pa.string())]);source_records=[dict(source_native_ID=d['source_native_ID'],source_row=d['source_row'],disposition_json=json.dumps(d,ensure_ascii=False,sort_keys=True)) for d in dispositions];p=O/'source-row-dispositions.parquet';pq.write_table(pa.Table.from_pylist(source_records,schema=source_schema),p,compression='zstd');assert pq.read_table(p).to_pylist()==source_records
  p=O/'NONAPPLYING-native-group-proposal.json';p.open('x').write(json.dumps(dict(groups=groups,source_member_rows=110,split=None,applied_to_master=False,current_global_exposure='not_evaluated',homology_parent_assay_material_equivalence_inferred=False),indent=2,ensure_ascii=False)+'\n')
  schema=pa.schema([('candidate_id',pa.string()),('source_native_ID',pa.string()),('native_source_row',pa.int64()),('task_name',pa.string()),('native_schema',pa.string()),('native_payload',pa.binary()),('native_payload_sha256',pa.string()),('source_preparation_sha256',pa.string()),('split',pa.string()),('global_task_admitted',pa.bool_()),('physical_ADMET_objective_admitted',pa.bool_()),('model_adapter_supported',pa.bool_())]);(O/'candidates').mkdir();shards=[];seen=set();readback=collections.Counter()
  for index,first in enumerate(range(0,len(candidates),128)):
   stage='candidate_shard:'+str(index);guard();out=[]
   for c in candidates[first:first+128]:
    raw=json.dumps(c,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode();n=c['audit']['source_ref']['native_source_row'];out.append(dict(candidate_id=c['candidate_id'],source_native_ID=source_rows[n]['source_native_DRAMP_ID'],native_source_row=n,task_name=c['task_name'],native_schema=c['schema'],native_payload=raw,native_payload_sha256=hashlib.sha256(raw).hexdigest(),source_preparation_sha256=s['inputs']['native_preparation']['sha256'],split=None,global_task_admitted=False,physical_ADMET_objective_admitted=False,model_adapter_supported=False))
   p=O/'candidates'/f'{index:05d}.parquet';pq.write_table(pa.Table.from_pylist(out,schema=schema),p,compression='zstd');back=pq.read_table(p).to_pylist()
   if back!=out:raise ValueError('whole candidate Parquet/canonical-byte roundtrip differs')
   for record in back:
    if record['candidate_id'] in seen:raise ValueError('duplicate candidate ID')
    seen.add(record['candidate_id']);c=json.loads(record['native_payload']);verify_candidate(c,source_rows[record['native_source_row']]);readback[(record['native_source_row'],record['task_name'])]+=1
   shards.append(dict(**ref(p),rows=len(out)))
  for d in dispositions:
   n=d['source_row'];actual=sum(v for (row,_),v in readback.items() if row==n)
   if actual!=d['candidate_views'] or readback[(n,'source_DRAMP_stability_report_prediction')]>1 or readback[(n,'source_DRAMP_single_half_life_literal_prediction')]>1:raise ValueError('all-source candidate cardinality conservation failed')
  for v in [s['parent_plan'],*s['inputs'].values(),*s['code_refs']]:checked(v)
  stage='complete';guard();receipt=dict(schema='mora3.DRAMP110SourceSymbolicCandidateComponent/v1',status='source_candidate_component_only_no_task_or_physical_admission',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supplement=ref(sp),approval=ref(approval),inputs=s['inputs'],worker_identity=ref(O/'worker-identity.json'),counts=dict(counts),source_rows_preserved=110,shards=shards,dispositions=ref(O/'source-row-dispositions.parquet'),NONAPPLYING_group_proposal=ref(O/'NONAPPLYING-native-group-proposal.json'),all_candidate_rows_full_source_target_condition_mask_ID_and_roundtrip_checked=len(seen),scalar_and_report_views_not_independent_experiments=True,source_evidence='database_curated_report;originalprimaryassaynotindependentlyverified',physical_ADMET=False,global_task_admission=False,model_use=False,master_publication=False,full_dataset_ready=False,model_training=False,actual_model_tokens=None,resource_samples=samples,seconds=time.monotonic()-start);(O/'manifest.json').open('x').write(json.dumps(receipt,indent=2)+'\n');print('COMPLETE_DRAMP110_SOURCE_SYMBOLIC_CANDIDATE_COMPONENT',json.dumps(dict(counts=dict(counts),seconds=receipt['seconds'])),flush=True)
 except Exception as error:
  (O/'failure.json').open('x').write(json.dumps(dict(stage=stage,error_type=type(error).__name__,error=str(error),counts=dict(counts),last_guard=samples[-1] if samples else None,partials_retained_no_retry=True),indent=2)+'\n');raise
if __name__=='__main__':main()
