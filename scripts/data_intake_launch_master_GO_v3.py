from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
R=Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261101-ProCyon-native-corpus-v1';Q=R/'planning/research/data_intake/20261101-expanded-scope-v3';P=B/'master-wide-heldout-GO-audit-plan-v3.json';A=R/'outputs/monitoring/review-20261101T165330Z-Norman-index-GO-audit-safe-continuations-v1/GO-v3-audit-execution-approval.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p))
def snapshot(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return None
 stat=(p/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(tail[19]),state=tail[0],argv=(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'),CPU_affinity=sorted(os.sched_getaffinity(pid)),nice=os.getpriority(os.PRIO_PROCESS,pid),RSS_bytes=int((p/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'))
def main():
 assert sha(A)=='5c61a27ec5d84c9d65c1fc8441daa8eb613b8b32b40b84cff027c18cba09bed7';assert sha(P)=='96d3c8811425bea65eb2fee3a86120a295499af4db55e0a2af5ef1d8e124c8b1';plan=json.loads(P.read_text());script=R/plan['script']['path'];assert sha(script)=='0c9b6b01e2a5e5c115cca203eba12169d1d12f65a0161f8a8b027ad82ff77001'
 for v in list(plan['inputs'].values())+plan['dependencies']+[plan['script']]:
  p=Path(v['path']);p=p if p.is_absolute() else R/p;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256']
 sys.path.insert(0,str(R/'scripts'));from data_intake_audit_master_heldout_GO_v3 import approve;approve(json.loads(A.read_text()),ref(P),ref(script))
 O=R/plan['output_namespace'];L=R/plan['launch']['exclusive_log'];receipt=Q/'master-wide-heldout-GO-v3-launch.json';assert not O.exists() and not L.exists() and not receipt.exists()
 duplicates=[]
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   try:raw=(p/'cmdline').read_bytes()
   except (FileNotFoundError,PermissionError,ProcessLookupError):continue
   if (script.name.encode()+b'\0') in raw:duplicates.append(int(p.name))
 assert not duplicates
 owner=next(x for x in plan['RNA_resource_owners'] if x['phase']=='verify-source-tasks');expected=owner['expected_process'];rna=snapshot(expected['pid']);assert rna is not None
 for key in ['pid','start_ticks','argv']:assert rna[key]==expected[key]
 assert rna['state']!='Z' and rna['CPU_affinity']==[26,27] and rna['nice']==10 and sha(R/owner['script_ref']['path'])==expected['script_sha256']
 assert not any((R/x).exists() for x in plan['uncoordinated_future_RNA_phase_receipts']);stat=os.statvfs(R);free=stat.f_bavail*stat.f_frsize;assert free>=64*2**30
 mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k=='MemAvailable'};assert mem['MemAvailable']>=512*2**20 and rna['RSS_bytes']+512*2**20+128*2**20<=8*2**30
 command=[x if x!='REPLACE_WITH_EXACT_ROOT_APPROVAL_PATH' else str(A.relative_to(R)) for x in plan['launch']['command_argv']];assert command[:4]==['/usr/bin/timeout','--signal=TERM','--kill-after=30s','1830s']
 f=L.open('xb');proc=subprocess.Popen(command,cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,{26,27}));f.close();child=None;stable=None
 for i in range(200):
  children=Path('/proc')/str(proc.pid)/'task'/str(proc.pid)/'children'
  for pid in children.read_text().split() if children.exists() else []:
   s=snapshot(int(pid))
   if s and plan['script']['path'] in s['argv']:
    if stable and s['start_ticks']==stable['start_ticks'] and s['argv']==stable['argv'] and s['nice']==10:child=s
    stable=s
  if child is not None or proc.poll() is not None:break
  time.sleep(.025)
 assert child is not None and proc.poll() is None
 guard=None
 for i in range(200):
  p=O/'guard-samples.jsonl'
  if p.exists() and p.stat().st_size:
   lines=p.read_text().splitlines()
   try:guard=json.loads(lines[0])
   except ValueError:pass
  if guard is not None or proc.poll() is not None:break
  time.sleep(.025)
 job=dict(schema='mora3.MasterWideHeldoutGOv3FiniteAuditLaunch/v1',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),timeout_supervisor=snapshot(proc.pid),python_worker=snapshot(child['pid']),command_argv=command,cwd=str(R),approval=ref(A),plan=ref(P),script=ref(script),prelaunch_RNA_identity=rna,prelaunch_free_bytes=free,prelaunch_MemAvailable_bytes=mem['MemAvailable'],output_namespace=plan['output_namespace'],exclusive_log=ref(L),first_guard=guard,max_seconds=1800,timeout_seconds=1830,network=False,GPU=0,task_admission=False,master_publication=False,mixture_policy_activation=False,model_use=False,model_training=False,no_retry_or_deadline_extension=True)
 receipt.open('x').write(json.dumps(job,indent=2)+'\n');assert guard is not None and not guard['threshold_failures'];print(json.dumps(dict(receipt=ref(receipt),job=job),indent=2))
if __name__=='__main__':main()
