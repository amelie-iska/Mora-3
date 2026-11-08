"""One finite candidate-only launch after separate exact root approval; no chaining."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys,time
R=Path('/home/iska/Documents/amelie/bio/Mora-3');SCRIPT=R/'scripts/data_intake_DRAMP_symbolic_component_v1.py'
def ref(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def snapshot(pid):
 try:
  p=Path('/proc')/str(pid);stat=(p/'stat').read_text();parts=stat[stat.rfind(')')+2:].split()
  return dict(pid=pid,start_ticks=int(parts[19]),state=parts[0],SID=os.getsid(pid),PGID=os.getpgid(pid),argv=(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'),CPU_affinity=sorted(os.sched_getaffinity(pid)),nice=os.getpriority(os.PRIO_PROCESS,pid),RSS_bytes=int((p/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'))
 except (FileNotFoundError,ProcessLookupError):return None

def main():
 p=argparse.ArgumentParser();p.add_argument('--supplement',required=True);p.add_argument('--approval',required=True);args=p.parse_args();sp=R/args.supplement;ap=R/args.approval;s=json.loads(sp.read_text());a=json.loads(ap.read_text())
 if a.get('launcher_sha256')!=ref(Path(__file__).resolve())['sha256']:raise ValueError('exact launcher approval missing')
 if ref(SCRIPT)!=s['script_ref']:raise ValueError('writer changed')
 sys.path.insert(0,str(R/'scripts'));from data_intake_DRAMP_symbolic_component_v1 import validate_approval
 validate_approval(a,ref(sp)['sha256'],ref(SCRIPT)['sha256'],s['parent_plan']['sha256'])
 output=R/s['output_namespace'];log=R/s['exclusive_log'];receipt=R/s['launch_receipt']
 for path in [output,log,receipt]:
  if path.exists():raise ValueError('one fresh candidate output/log/receipt required')
 if not output.is_relative_to(R/'outputs/data-intake') or not receipt.is_relative_to(R/'planning/research/data_intake'):raise ValueError('new owned candidate namespace required')
 for pid in s['quiescence']['old_owned_PIDs']:
  if (Path('/proc')/str(pid)).exists():raise ValueError('old/reused constructorPID needs fresh root review')
 command=['/usr/bin/timeout','--signal=TERM','--kill-after=15s','150s','/home/iska/miniconda3/envs/tokengt/bin/python','-u',str(SCRIPT.relative_to(R)),'--supplement',str(sp.relative_to(R)),'--approval',str(ap.relative_to(R))]
 f=log.open('xb');job=subprocess.Popen(command,cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,{27}));f.close();stable=None;child=None
 for _ in range(200):
  children=Path('/proc')/str(job.pid)/'task'/str(job.pid)/'children'
  for pid in children.read_text().split() if children.exists() else []:
   actual=snapshot(int(pid))
   if actual and str(SCRIPT.relative_to(R)) in actual['argv']:
    if stable and actual['start_ticks']==stable['start_ticks'] and actual['argv']==stable['argv'] and actual['nice']>=10:child=actual
    stable=actual
  if child or job.poll() is not None:break
  time.sleep(.025)
 record=dict(schema='mora3.DRAMP110FiniteCandidateLaunch/v1',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),timeout_supervisor_pid=job.pid,timeout_supervisor=snapshot(job.pid),python_worker=child or stable,command_argv=command,approval=ref(ap),supplement=ref(sp),writer=ref(SCRIPT),launcher=ref(Path(__file__).resolve()),exclusive_log=str(log.relative_to(R)),output_namespace=s['output_namespace'],launch_returncode=job.poll(),max_seconds=120,outer_timeout_seconds=150,CPU_affinity=[27],nice_minimum=10,model_use=False,master_publication=False,task_admission=False,no_retry_or_chained_phase=True)
 receipt.open('x').write(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
