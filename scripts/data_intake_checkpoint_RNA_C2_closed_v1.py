from pathlib import Path
import json,hashlib,datetime,os
R=Path('/home/iska/Documents/amelie/bio/Mora-3');os.chdir(R)
D=R/'planning/research/data_intake/20261101-RNAcentral-full-construction-v1/publisher-review-preparation-v1'
D.mkdir(exist_ok=False)
def ref(p):
 p=Path(p); b=p.read_bytes(); return dict(path=str(p.relative_to(R)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(s):return json.loads((R/s).read_text())
closure=ref(R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1/C2-normal-completion-closure-v1/receipt.json')
c=read(closure['path']); assert c['owners_absent'] and c['SID_members_twice']==[[],[]] and not c['publisher_invoked']
now=datetime.datetime.now(datetime.timezone.utc).isoformat();changes=[]
for f in ['planning/data_intake_progress.json','configs/dataset_acquisition_queue.json']:
 p=R/f; old=p.read_bytes(); backup=D/(p.stem+'-before-terminal-checkpoint.json');backup.open('xb').write(old); d=json.loads(old)
 assert 'v22' in d.get('current_master_corpus_manifest',d['current_corpus_manifest'])
 d.update(updated_utc=now,whole_RNAcentral_staged_v23_normal_closure=closure,whole_RNAcentral_C2_live=False,whole_RNAcentral_C2_status='normal_complete_staged_only_pointer_publication_not_executed')
 ad=d['scope_extensions']['ADMET-CORPUS-01']; ad['DRAMP_source_symbolic_writer_preparation']=ref(R/'planning/research/data_intake/20261101-DRAMP-offline-preparation-v1/writer-preparation-v1/preparation-manifest.json');ad['DRAMP_source_symbolic_static_root_review']=ref(R/'outputs/monitoring/review-20261101T192703Z-DRAMP-candidate-writer-static-only-v1/review.json');ad['DRAMP_source_symbolic_component_status']='complete110native_preparation_and_actualcontextrole_review_verified;writer_frozen16syntheticchecks_and_root_static_review_only;exact_execution_supplement_and_approval_pending;actual110_projection_unrun'
 if f.startswith('planning/'):
  jobs=d.get('running_jobs',[])
  for job in jobs: assert job.get('pid')==4086786 and job.get('python_worker_pid')==4086787 and not Path('/proc/4086786').exists() and not Path('/proc/4086787').exists()
  d.update(status='verified_whole_RNAcentral_v23_staged_pending_pointer_review',running_jobs=[],current_owned_task='Freeze separately reviewed RNAcentral v23 pointer publisher with exact backups; no activation yet',next_job='Await exact publisher review; keep staged v23/current v22 unchanged until approved',last_completed_job=dict(kind='whole_RNAcentral_C2_union_master',normal_completion=True,closure=closure,former_launches=jobs))
  for k in ['whole_RNAcentral_C2_actual_runtime']:
   if isinstance(d.get(k),dict): d[k]['historical_runtime_only_currently_live']=False
 new=json.dumps(d,indent=2)+'\n'; tmp=p.with_name(p.name+'.C2-closed-new');tmp.open('x').write(new);os.replace(tmp,p);changes.append(dict(path=f,before=ref(backup),after=ref(p)))
receipt=dict(schema='mora3.RNAC2TerminalCheckpoint/v1',created_utc=now,closure=closure,changes=changes,current_pointer_v22_unchanged=True,publication_executed=False,full_dataset_ready=False,model_use=False,training=False)
p=D/'terminal-checkpoint-receipt.json';p.open('x').write(json.dumps(receipt,indent=2)+'\n');print(json.dumps(ref(p)))
