"""One explicit finite LINCS union launch; no retries or model/source-job changes."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time
R=Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261101-LINCS-corpus-v1';S=R/'scripts/data_intake_union_lincs_v22.py';O=B/'independent-master-union-v22';receipt=B/'union-v22-launch.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p))
def main():
 assert not O.exists() and not receipt.exists()
 p=R/'planning/data_intake_progress.json';x=json.loads(p.read_text());assert not x['running_jobs']
 master=R/'outputs/data-intake/20261101-PubChem-native-corpus-v1/materialized/independent-final-v1/master-corpus-manifest-v21-PubChem-whole-native-train-v1.json';pub=json.loads((master.parent/'publication-receipt.json').read_text());assert pub['master']==ref(master);assert x['current_master_corpus_manifest']==str(master.relative_to(R)) or x['current_corpus_manifest']==str(master.relative_to(R))
 now=datetime.datetime.now(datetime.timezone.utc);command=['/usr/bin/timeout','--signal=TERM','--kill-after=30s','1830s','/home/iska/miniconda3/envs/tokengt/bin/python','-u',str(S.relative_to(R))];log=B/'union-v22.log';f=log.open('xb');process=subprocess.Popen(command,cwd=R,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,{30}));f.close();child=None
 for _ in range(120):
  c=Path('/proc')/str(process.pid)/'task'/str(process.pid)/'children'
  if c.exists():
   for value in c.read_text().split():
    cmd=Path('/proc')/value/'cmdline'
    if cmd.exists() and S.name.encode() in cmd.read_bytes():child=int(value)
  if child is not None or process.poll() is not None:break
  time.sleep(.025)
 job=dict(kind='finite_qualified_LINCS_v22_master_union',pid=process.pid,timeout_supervisor_pid=process.pid,python_worker_pid=child,live=process.poll() is None,started_utc=now.isoformat(),deadline_utc=(now+datetime.timedelta(seconds=1830)).isoformat(),CPU_affinity=[30],max_CPU=1,GPU=0,network=False,log=str(log.relative_to(R)),script=ref(S),parent_master=ref(master),expected_output=str((O/'master-corpus-manifest-v22-LINCS-qualified-source-coordinate-v1.json').relative_to(R)),source_component=ref(B/'source-dose-coordinate-qualified-v2/manifest.json'),launch_returncode=process.poll(),no_training=True)
 receipt.open('x').write(json.dumps(job,indent=2)+'\n');x.update(updated_utc=now.isoformat(),status='active_qualified_LINCS_v22_independent_union',running_jobs=[job],active_batch=str(B.relative_to(R)),current_owned_task='full_inherited_master_ID_group_hash_union_and_LINCS_qualified_component_admission',next_job='Publish v22 handoff after verified normal completion; prepare full RNAcentral hardware-policy construction and expanded scopes');p.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(job,indent=2))
if __name__=='__main__':main()
