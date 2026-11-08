"""Explicit root-reviewed whole-source census launch; no automatic retry/next phase."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,time
R=Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p))
def write(p,v):p.open('x').write(json.dumps(v,indent=2)+'\n')
def validate_approval(approval,plan_hash,launcher_hash,resource_hash):
 if approval.get('approved') is not True or approval.get('phase')!='census' or approval.get('scope')!='whole-native-RNAcentral-census-only':raise ValueError('explicit exact census execution approval required')
 if approval.get('plan_sha256')!=plan_hash or approval.get('launcher_sha256')!=launcher_hash or approval.get('resource_coordination_sha256')!=resource_hash:raise ValueError('approval must bind exact plan/launcher/resource versions')
 if approval.get('whole_source_occurrences')!=61026468:raise ValueError('approval must cover complete native source, not replacement pilot')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--approval',type=Path,required=True);a=parser.parse_args();approval=a.approval if a.approval.is_absolute() else R/a.approval;planpath=B/'build-plan.json';plan=json.loads(planpath.read_text());preflight=B/'preflight/receipt.json';pre=json.loads(preflight.read_text());assert pre['status']=='verified_preflight_only_not_launched' and pre['plan']==ref(planpath);assert plan['CPU_affinity']==[26,27] and plan['max_peak_RSS_bytes']==8*2**30 and plan['whole_deadline_seconds']==43200 and plan['free_buffer_bytes']==64*2**30;validate_approval(json.loads(approval.read_text()),sha(planpath),sha(Path(__file__)),plan['resource_coordination']['sha256'])
 for record in plan['code_refs']:
  p=R/record['path'];assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256']
 resource=R/plan['resource_coordination']['path'];assert sha(resource)==plan['resource_coordination']['sha256'];assert not (B/'build-clock.json').exists() and not (B/'census').exists() and not (B/'census-launch-receipt.json').exists();p=R/'planning/data_intake_progress.json';x=json.loads(p.read_text());assert not x['running_jobs'];assert x['current_corpus_manifest']==plan['baseline_master']['path'];st=os.statvfs(R);assert st.f_bavail*st.f_frsize-2**30>=plan['free_buffer_bytes'];mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k=='MemAvailable'};assert mem['MemAvailable']>=8*2**30
 now=datetime.datetime.now(datetime.timezone.utc);clock=dict(schema='mora3.FullRNAcentralWholeBuildClock/v1',created_utc=now.isoformat(),plan_sha256=sha(planpath),deadline_epoch=time.time()+43200,whole_max_seconds=43200,root_execution_approval=ref(approval));write(B/'build-clock.json',clock);deadline=datetime.datetime.fromtimestamp(clock['deadline_epoch'],datetime.timezone.utc).isoformat();script=R/'scripts/data_intake_full_rnacentral_census.py';seconds=int(clock['deadline_epoch']-time.time()-30);command=['/usr/bin/timeout','--signal=TERM','--kill-after=30s',str(seconds)+'s','/home/iska/miniconda3/envs/tokengt/bin/python','-u',str(script.relative_to(R))];log=B/'census.log';f=log.open('xb');proc=subprocess.Popen(command,cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,{26,27}));f.close();child=None
 for _ in range(200):
  children=Path('/proc')/str(proc.pid)/'task'/str(proc.pid)/'children'
  if children.exists():
   for pid in children.read_text().split():
    c=Path('/proc')/pid/'cmdline'
    if c.exists() and c.read_bytes().split(b'\0')[:3]==[b'/home/iska/miniconda3/envs/tokengt/bin/python',b'-u',str(script.relative_to(R)).encode()]:child=int(pid)
  if child is not None or proc.poll() is not None:break
  time.sleep(.025)
 job=dict(schema='mora3.FullRNAcentralCensusExplicitLaunch/v1',kind='finite_whole_RNAcentral_native_census',pid=proc.pid,timeout_supervisor_pid=proc.pid,python_worker_pid=child,live=proc.poll() is None,started_utc=now.isoformat(),deadline_utc=deadline,plan=ref(planpath),preflight=ref(preflight),execution_approval=ref(approval),clock=ref(B/'build-clock.json'),script=ref(script),launcher=ref(Path(__file__)),CPU_affinity=[26,27],max_CPU=2,nice_minimum=10,owned_aggregate_RSS_bound_bytes=8*2**30,GPU=0,network=False,hardware_free_buffer_bytes=64*2**30,artificial_storage_cap_bytes=None,whole_source_occurrences=61026468,new_task_admissions=0,substantive_training=False,log=str(log.relative_to(R)),expected_output=str((B/'census/manifest.json').relative_to(R)),launch_returncode=proc.poll(),host_memory_before_launch=mem)
 write(B/'census-launch-receipt.json',job);x.update(updated_utc=now.isoformat(),status='active_whole_RNAcentral_native_census',running_jobs=[job],active_batch=str(B.relative_to(R)),current_owned_task='Complete native61M source/identity/URS/heldout census under frozen reviewed package',next_job='After verified census, separately reviewed full materialization and independent proof; no partial master admission',full_RNAcentral_execution_plan=ref(planpath));p.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(job,indent=2))
if __name__=='__main__':main()
