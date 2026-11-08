"""Launch one explicit finite downstream phase; never poll/resume in a loop."""
from pathlib import Path
import os,sys,json,hashlib,time,datetime,subprocess
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_full_pubchem_guards import immutable_json
from data_intake_full_pubchem_guards_v2 import BuildGuard
O=R/'outputs/data-intake/20261101-PubChem-native-corpus-v1'
SPECS={'census-proof':('data_intake_full_pubchem_census_proof.py','census/manifest.json','materialized/census-independent-proof-v1/receipt.json'),'materialize':('data_intake_full_pubchem_materialize.py','materialized/census-independent-proof-v1/receipt.json','materialized/manifest.json'),'finalize':('data_intake_full_pubchem_finalize.py','materialized/manifest.json','materialized/independent-final-v1/master-corpus-manifest-v21-PubChem-whole-native-train-v1.json')}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=SPECS,required=True);args=parser.parse_args();phase=args.phase;plan=json.loads((O/'build-plan.json').read_text());guard=BuildGuard(O,census_cap=2*2**30,total_cap=12*2**30,deadline_epoch=plan['deadline_epoch']);guard.check('materialized',reserve=64*2**20);script,prereq,output=SPECS[phase];assert (O/prereq).is_file() and not (O/output).exists();parent=json.loads((O/prereq).read_text());assert parent['status'].startswith(('complete_census','verified','serialized_pending'));launchdir=O/'phase-launches';launchdir.mkdir(exist_ok=True);receipt=launchdir/(phase+'.json');assert not receipt.exists(),'phasealreadylaunched;noautomaticresetorretry';receipts=[O/'launch-receipt.json',*launchdir.glob('*.json')]
 for p in receipts:
  if not p.exists():continue
  old=json.loads(p.read_text())
  for pid in [old.get('pid'),old.get('python_worker_pid')]:
   if pid is None:continue
   cmdline=Path('/proc')/str(pid)/'cmdline'
   if cmdline.exists():
    raw=cmdline.read_bytes()
    if b'data_intake_full_pubchem_' in raw and b'phase_launch.py' not in raw:raise RuntimeError('existingfullbuildphaseisstillalive;donotduplicateorinterrupt')
 cores=plan.get('CPU_affinity') or json.loads((O/'launch-receipt.json').read_text())['CPU_affinity'];assert len(cores)<=2;remaining=int(plan['deadline_epoch']-time.time()-30);assert remaining>30,'insufficientwholedeadline;preservepartialsandreport';command=['/usr/bin/timeout','--signal=TERM','--kill-after=30s',str(remaining)+'s','/home/iska/miniconda3/envs/tokengt/bin/python','-u','scripts/'+script];log=O/(phase+'.log');f=log.open('xb');process=subprocess.Popen(command,cwd=R,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,set(cores)));f.close();python_pid=None
 for _ in range(120):
  children=Path('/proc')/str(process.pid)/'task'/str(process.pid)/'children'
  if children.exists():
   for candidate in children.read_text().split():
    p=Path('/proc')/candidate/'cmdline'
    if p.exists() and script.encode() in p.read_bytes():python_pid=int(candidate)
  if python_pid is not None or process.poll() is not None:break
  time.sleep(.025)
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();job=dict(pid=process.pid,timeout_supervisor_pid=process.pid,python_worker_pid=python_pid,live=process.poll() is None,kind='finite_full_PubChem_'+phase,started_utc=now,deadline_utc=plan['deadline_utc'],CPU_affinity=cores,max_CPU=2,GPU=0,network=False,log=str(log.relative_to(R)),expected_output=str((O/output).relative_to(R)),plan_sha256=sha(O/'build-plan.json'),script_sha256=sha(R/'scripts'/script),launch_returncode=process.poll(),source_master_not_yet_changed=True);immutable_json(receipt,job);p=R/'planning/data_intake_progress.json';x=json.loads(p.read_text());x.update(updated_utc=now,status='active_authorized_whole_native_PubChem_'+phase,running_jobs=[job],current_owned_task='finitefullPubChem_'+phase,active_batch=str(O.relative_to(R)),next_job='complete_currentfinitephase_thenexplicit_nextsafephase_under_same_deadline_caps');p.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(job,indent=2))
if __name__=='__main__':main()
