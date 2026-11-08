from pathlib import Path
import json,hashlib,os,sys,subprocess,time,datetime,shutil
R=Path('/home/iska/Documents/amelie/bio/Mora-3');os.chdir(R);sys.path.insert(0,str(R/'scripts'))
from data_intake_publish_full_rnacentral_guard_v1 import reference,validate_pin,validate_approval,verify_index_stats,require_no_other_RNA_owner
B=R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1';F=R/'planning/research/data_intake/20261101-RNAcentral-full-construction-v1/publisher-review-preparation-v1/frozen-execution-package-v1'
planfile=F/'execution-plan.json';plan=json.loads(planfile.read_text());a=R/'outputs/monitoring/execution-20261101T194238Z-RNA-v23-internal-pointer-handoff-v1/execution-approval.json'
assert reference(a)['sha256']=='0f388e97f2c47f4ac2050b2130b82f3b5c807a5d116eef3467fff497cb3d85b7'
validate_approval(json.loads(a.read_text()),reference(planfile)['sha256'],reference(R/plan['guard']['path'])['sha256'])
for p in plan['input_pins']:validate_pin(p)
closure=json.loads((R/plan['closure']['path']).read_text());verify_index_stats(closure['giant_index_open_close_hash_evidence']);require_no_other_RNA_owner()
assert not (R/plan['run_directory']).exists() and not (R/plan['handoff_directory']).exists()
assert json.loads((R/'planning/data_intake_progress.json').read_text())['current_master_corpus_manifest']==plan['baseline_master']['path']
assert shutil.disk_usage(R).free>=65*1024**3
log=B/'pointer-publication-v1.log';launch=B/'pointer-publication-launch-v1.json';assert not log.exists() and not launch.exists()
argv=json.loads((F/'future-exact-argv.json').read_text())['argv'];assert argv[-1]=='ROOT_EXACT_APPROVAL_PATH_REQUIRED';argv[-1]=str(a)
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
with log.open('xb') as f:p=subprocess.Popen(argv,cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
def proc(pid):
 d=Path('/proc')/str(pid)
 try:
  s=(d/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,start_ticks=int(s[19]),parent_pid=int(s[1]),SID=int(s[3]),argv=[os.fsdecode(x) for x in (d/'cmdline').read_bytes().split(b'\0') if x])
 except (FileNotFoundError,ProcessLookupError):return None
timeout_info=proc(p.pid);worker=None
end=time.monotonic()+2
while time.monotonic()<end:
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  z=proc(int(d.name))
  if z and z['SID']==p.pid and any(Path(x).name=='data_intake_publish_full_rnacentral_guard_v1.py' for x in z['argv']) and z['pid']!=p.pid:worker=z;break
 if worker or p.poll() is not None:break
 time.sleep(0.02)
receipt=dict(schema='mora3.RNAInternalPointerActualLaunch/v1',started_utc=started,approval=reference(a),plan=reference(planfile),argv=argv,timeout=timeout_info,worker=worker,SID=p.pid,log=str(log.relative_to(R)),new_session=True,stdin='DEVNULL',log_exclusive_created=True,old_pins_rechecked=True,full_dataset_ready=False,model_use=False,training=False)
launch.open('x').write(json.dumps(receipt,indent=2)+'\n');print('ACTUAL_LAUNCH',json.dumps(reference(launch)),flush=True);print('OWNER',json.dumps(dict(timeout=timeout_info,worker=worker)),flush=True)
try:rc=p.wait(timeout=20)
except subprocess.TimeoutExpired:print('FINITE_APPROVED_JOB_STILL_RUNNING',p.pid,flush=True);raise SystemExit(0)
print('EXIT',rc,flush=True)
if rc!=0: print(log.read_text()[-5000:]);raise SystemExit(rc)
run=R/plan['run_directory'];final=json.loads((run/'receipt.json').read_text());assert final['status']=='normal_complete'
if worker is None:receipt['worker_from_final_initial_guard']=json.loads((run/'prepublication-backup-receipt.json').read_text())['initial_guard']
checks=[]
for _ in range(2):
 members=[]
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  z=proc(int(d.name))
  if z and z['SID']==p.pid:members.append(z)
 checks.append(members);time.sleep(0.1)
assert checks==[[],[]]
assert log.read_text().splitlines()[-1].startswith('COMPLETE_FULL_RNACENTRAL_POINTER_PUBLICATION ')
normal=dict(schema='mora3.RNAInternalPointerNormalClosure/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),launch=reference(launch),approved_wrapper_receipt=reference(run/'receipt.json'),exit0=True,timeout_worker_absent=True,SID_members_twice=checks,final_marker=True,current_master=json.loads((R/'planning/data_intake_progress.json').read_text())['current_master_corpus_manifest'],current_handoff=json.loads((R/'planning/data_intake_progress.json').read_text())['executable_model_handoff'],actual_worker=worker or receipt.get('worker_from_final_initial_guard'),log=reference(log),full_dataset_ready=False,model_use=False,mixture_activation=False,training=False)
np=B/'pointer-publication-normal-closure-v1.json';np.open('x').write(json.dumps(normal,indent=2)+'\n');print('NORMAL_CLOSURE',json.dumps(reference(np)));print('FINAL',json.dumps(final,indent=2))
