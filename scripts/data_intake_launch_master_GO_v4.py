from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
R=Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261101-ProCyon-native-corpus-v1';Q=R/'planning/research/data_intake/20261101-expanded-scope-v3';P=B/'master-wide-heldout-GO-audit-plan-v4.json';A=R/'outputs/monitoring/execution-20261101T183828Z-GOv4-and-RGD1-preparation-v2/GO-v4-execution-approval.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p))
def snapshot(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return None
 stat=(p/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
 return dict(pid=pid,SID=os.getsid(pid),PGID=os.getpgid(pid),start_ticks=int(tail[19]),state=tail[0],argv=(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'),CPU_affinity=sorted(os.sched_getaffinity(pid)),nice=os.getpriority(os.PRIO_PROCESS,pid),RSS_bytes=int((p/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'))
def main():
 assert sha(A)=='9949ae0c92922a00ed91ef418fcbe59ba01d4703928b0bd8a2f4149e3967d335';assert sha(P)=='c40f5d735dd704fc35bbd4d7609703606fd53a8572b8869c8f926e6037c96ea9';plan=json.loads(P.read_text());script=R/plan['script']['path'];assert sha(script)=='aca38ee7deac667f84812044484eb05bf0482c6860b475ecadd757b59d9ea92e'
 for v in list(plan['inputs'].values())+plan['dependencies']+[plan['script']]:
  p=Path(v['path']);p=p if p.is_absolute() else R/p;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256']
 sys.path.insert(0,str(R/'scripts'));from data_intake_audit_master_heldout_GO_v4 import approve;approve(json.loads(A.read_text()),ref(P),ref(script))
 O=R/plan['output_namespace'];L=R/plan['launch']['exclusive_log'];receipt=Q/'master-wide-heldout-GO-v4-launch.json';assert not O.exists() and not L.exists() and not receipt.exists()
 duplicates=[]
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   try:raw=(p/'cmdline').read_bytes()
   except (FileNotFoundError,PermissionError,ProcessLookupError):continue
   if (script.name.encode()+b'\0') in raw:duplicates.append(int(p.name))
 assert not duplicates
 from data_intake_RNA_phase_coexistence import normal_source_proof,normal_materialization
 for owner in plan['RNA_resource_owners']:
  assert snapshot(owner['expected_process']['pid']) is None
  if owner['phase']=='materialize':
   assert normal_materialization(owner['completed_manifest_summary'],(R/owner['completion_log']['path']).read_text()[-8192:],owner['expected_completion_bindings'])
  else:
   assert normal_source_proof(json.loads((R/owner['completion_receipt_path']).read_text()),(R/owner['completion_log_path']).read_text()[-16384:],owner['expected_completion_bindings'])
 assert snapshot(3756182) is None and snapshot(3896454) is None and snapshot(3896455) is None
 rna=dict(state='verified_normal_completion',RSS_bytes=0,normal_receipt=plan['inputs']['source_task_proof_normal_receipt'])
 old=R/plan['inputs']['stopped_v3_database']['path'];f=json.loads((R/plan['inputs']['stopped_v3_forensic']['path']).read_text());assert {p.name for p in old.parent.iterdir()}==set(f['original_artifacts_before'])
 assert not any(Path(str(old)+s).exists() for s in ['-journal','-wal','-shm'])
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
 job=dict(schema='mora3.MasterWideHeldoutGOv4FiniteContinuationLaunch/v1',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),timeout_supervisor=snapshot(proc.pid),python_worker=snapshot(child['pid']),command_argv=command,cwd=str(R),approval=ref(A),plan=ref(P),script=ref(script),prelaunch_RNA_identity=rna,prelaunch_free_bytes=free,prelaunch_MemAvailable_bytes=mem['MemAvailable'],output_namespace=plan['output_namespace'],exclusive_log=ref(L),first_guard=guard,max_seconds=1800,timeout_seconds=1830,network=False,GPU=0,task_admission=False,master_publication=False,mixture_policy_activation=False,model_use=False,model_training=False,no_retry_or_deadline_extension=True)
 receipt.open('x').write(json.dumps(job,indent=2)+'\n');assert guard is not None and not guard['threshold_failures'];print(json.dumps(dict(receipt=ref(receipt),job=job),indent=2))
if __name__=='__main__':main()
