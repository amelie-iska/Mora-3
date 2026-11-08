"""One separately reviewed downstream RNA phase; no automatic retries/chaining."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,time,sys
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'));B=R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1'
SPECS={
 'materialize':dict(script='data_intake_full_rnacentral_materialize.py',prereq='census/manifest.json',previous_log='census.log',previous_marker='COMPLETE_FULL_RNACENTRAL_CENSUS',output='materialized/manifest.json'),
 'verify-source-tasks':dict(script='data_intake_full_rnacentral_verify_source_tasks.py',prereq='materialized/manifest.json',previous_log='materialize.log',previous_marker='COMPLETE_FULL_RNACENTRAL_MATERIALIZATION',output='independent-source-task-proof-v1/receipt.json'),
 'union-master':dict(script='data_intake_full_rnacentral_union_master.py',prereq='independent-source-task-proof-v1/receipt.json',previous_log='verify-source-tasks.log',previous_marker='COMPLETE_FULL_RNA_SOURCE_TASK_PROOF',output='independent-master-union-v23/master-corpus-manifest-v23-whole-RNAcentral-native-v1.json')}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p))
def remaining_whole_seconds(clock,now):
 created=datetime.datetime.fromisoformat(clock['created_utc']).timestamp()
 if clock['whole_max_seconds']!=43200 or abs(clock['deadline_epoch']-created-43200)>5:raise ValueError('one immutable reviewed12h clock required')
 seconds=int(clock['deadline_epoch']-now-30)
 if seconds<=60:raise ValueError('insufficient whole-build time; preserve results and seek a new reviewed continuation')
 return seconds

def last_completion_marker(path,expected):
 with path.open('rb') as f:
  f.seek(max(0,path.stat().st_size-65536));lines=f.read().decode().splitlines()
 nonempty=[x for x in lines if x.strip()]
 return bool(nonempty and nonempty[-1].startswith(expected))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=SPECS,required=True);ap.add_argument('--approval',type=Path,required=True);args=ap.parse_args();spec=SPECS[args.phase];approval=args.approval if args.approval.is_absolute() else R/args.approval;a=json.loads(approval.read_text());assert a.get('launcher_sha256')==sha(Path(__file__));pp=B/'build-plan.json';plan=json.loads(pp.read_text());cp=B/'census/manifest.json';clockpath=B/'build-clock.json';clock=json.loads(clockpath.read_text());assert clock['plan_sha256']==sha(pp);seconds=remaining_whole_seconds(clock,time.time());assert (B/spec['prereq']).exists() and not (B/spec['output']).exists() and last_completion_marker(B/spec['previous_log'],spec['previous_marker']);names=['data_intake_full_rnacentral_tasks.py','data_intake_full_rnacentral_state.py','data_intake_full_rnacentral_census.py','data_intake_rnacentral_prior.py','data_intake_nucleotide_prior.py','data_intake_corpus.py']
 if args.phase=='materialize':
  from data_intake_full_rnacentral_materialize import validate_materialization_approval
  validate_materialization_approval(a,plan=sha(pp),census=sha(cp),clock=sha(clockpath),script=sha(R/'scripts'/spec['script']),resource=plan['resource_coordination']['sha256'],dependencies=[ref(R/'scripts'/n) for n in names])
 elif args.phase=='verify-source-tasks':
  from data_intake_full_rnacentral_verify_source_tasks import validate_source_task_proof_approval
  names=['data_intake_full_rnacentral_replay_index.py','data_intake_full_rnacentral_proof_helpers.py','data_intake_full_rnacentral_state.py','data_intake_rnacentral_prior.py','data_intake_nucleotide_prior.py','data_intake_corpus.py'];validate_source_task_proof_approval(a,dict(plan_sha256=sha(pp),census_manifest_sha256=sha(cp),component_manifest_sha256=sha(B/'materialized/manifest.json'),clock_sha256=sha(clockpath),script_sha256=sha(R/'scripts'/spec['script']),resource_coordination_sha256=plan['resource_coordination']['sha256'],dependency_refs=[ref(R/'scripts'/n) for n in names]))
 else:
  from data_intake_full_rnacentral_union_master import validate_union_approval
  names=['data_intake_full_rnacentral_state.py','data_intake_full_rnacentral_tasks.py','data_intake_full_rnacentral_proof_helpers.py','data_intake_rnacentral_prior.py','data_intake_nucleotide_prior.py','data_intake_corpus.py'];validate_union_approval(a,dict(plan_sha256=sha(pp),census_manifest_sha256=sha(cp),component_manifest_sha256=sha(B/'materialized/manifest.json'),source_task_proof_sha256=sha(B/'independent-source-task-proof-v1/receipt.json'),clock_sha256=sha(clockpath),script_sha256=sha(R/'scripts'/spec['script']),resource_coordination_sha256=plan['resource_coordination']['sha256'],dependency_refs=[ref(R/'scripts'/n) for n in names]))
 launchdir=B/'downstream-launches';launchdir.mkdir(exist_ok=True);rp=launchdir/(args.phase+'.json');assert not rp.exists()
 receipts=[B/'census-launch-receipt.json',*launchdir.glob('*.json')];owned_names=[b'data_intake_full_rnacentral_census.py',*[s['script'].encode() for s in SPECS.values()]]
 for receipt in receipts:
  old=json.loads(receipt.read_text())
  for pid in [old.get('pid'),old.get('python_worker_pid')]:
   if pid is None:continue
   p=Path('/proc')/str(pid)/'cmdline';raw=p.read_bytes() if p.exists() else b''
   if any(n in raw for n in owned_names):raise RuntimeError('an existing owned RNA phase is live; do not duplicate or interrupt')
 for record in plan['code_refs']:
  p=R/record['path'];assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256']
 st=os.statvfs(R);assert st.f_bavail*st.f_frsize-2**30>64*2**30;mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k=='MemAvailable'};assert mem['MemAvailable']>=8*2**30;script=R/'scripts'/spec['script'];command=['/usr/bin/timeout','--signal=TERM','--kill-after=30s',str(seconds)+'s','/home/iska/miniconda3/envs/tokengt/bin/python','-u',str(script.relative_to(R)),'--approval',str(approval.relative_to(R))];log=B/(args.phase+'.log');f=log.open('xb');proc=subprocess.Popen(command,cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,{26,27}));f.close();child=None
 for _ in range(200):
  p=Path('/proc')/str(proc.pid)/'task'/str(proc.pid)/'children'
  if p.exists():
   for pid in p.read_text().split():
    c=Path('/proc')/pid/'cmdline'
    if c.exists() and spec['script'].encode() in c.read_bytes():child=int(pid)
  if child is not None or proc.poll() is not None:break
  time.sleep(.025)
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();job=dict(schema='mora3.FullRNASeparatelyReviewedPhaseLaunch/v1',kind='finite_whole_RNAcentral_'+args.phase,pid=proc.pid,timeout_supervisor_pid=proc.pid,python_worker_pid=child,live=proc.poll() is None,started_utc=now,deadline_utc=datetime.datetime.fromtimestamp(clock['deadline_epoch'],datetime.timezone.utc).isoformat(),plan=ref(pp),clock=ref(clockpath),execution_approval=ref(approval),script=ref(script),launcher=ref(Path(__file__)),CPU_affinity=[26,27],max_CPU=2,nice_minimum=10,owned_aggregate_RSS_bound_bytes=8*2**30,GPU=0,network=False,hardware_free_buffer_bytes=64*2**30,artificial_storage_cap_bytes=None,log=str(log.relative_to(R)),expected_output=str((B/spec['output']).relative_to(R)),launch_returncode=proc.poll(),master_pointer_not_yet_changed=True)
 rp.open('x').write(json.dumps(job,indent=2)+'\n');p=R/'planning/data_intake_progress.json';x=json.loads(p.read_text());x.update(updated_utc=now,status='active_whole_RNAcentral_'+args.phase,running_jobs=[job],current_owned_task='Complete reviewed whole-source '+args.phase+' under original immutable clock',next_job='Verify normal completion then separately reviewed next stage; no partial master promotion');p.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(job,indent=2))
if __name__=='__main__':main()
