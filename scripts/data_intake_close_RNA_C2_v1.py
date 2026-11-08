"""Closed staged-master proof; no giant-index replay or pointer publication."""
from pathlib import Path
import datetime,hashlib,json,os,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1';U=B/'independent-master-union-v23';O=B/'C2-normal-completion-closure-v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while c:=f.read(2**20):h.update(c)
 return h.hexdigest()
def ref(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p))
def checked(v):
 p=R/v['path'];assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'];return p
def sessions():
 members=[]
 for p in Path('/proc').iterdir():
  if p.name.isdigit():
   try:s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split()
   except (FileNotFoundError,ProcessLookupError,PermissionError):continue
   if int(f[3])==4086786:members.append(dict(pid=int(p.name),state=f[0],start_ticks=int(f[19])))
 return members
start=time.monotonic();os.sched_setaffinity(0,{27});os.nice(10);pa.set_cpu_count(1);assert not O.exists();proofpath=U/'receipt.json';p=json.loads(proofpath.read_text());pubpath=U/'publication-receipt.json';pub=json.loads(pubpath.read_text());assert pub['proof']==ref(proofpath);assert p['status']=='verified' and p['actual_new_old_task_ID_intersections']==p['known_newtrain_oldheldout_group_intersections']==0;assert p['all_old_task_and_group_files_unchanged'] and p['all_inherited_data_group_hashes_sizes_footer_rows_checked'];assert p['all_inherited_task_IDs_scanned']==26322315 and p['new_task_views']==54732469 and p['preserved_Rfam_and_old_pilot_exact_tasks_reused']==6273790;assert p['preserved_task_group_sidecars_new_versions']==3201
for v in [p['prior_master'],p['component'],p['census'],p['all_source_and_new_task_proof'],p['clock'],p['execution_approval']]:checked(v)
component=json.loads(checked(p['component']).read_text());census=json.loads(checked(p['census']).read_text());index_refs=[census['native_identity_URS_index'],component['task_identity_serialization_index']];del component,census
index_stats=[];created=datetime.datetime.fromisoformat(p['created_utc']).timestamp()
for v in index_refs:
 path=R/v['path'];st=path.stat();assert st.st_size==v['bytes'] and st.st_mtime<=created
 index_stats.append(dict(expected_full_SHA_reference=v,current_stat=dict(bytes=st.st_size,inode=st.st_ino,device=st.st_dev,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns),closure_evidence='ReviewedC2script performs completeSHAopening+closing before verifiedreceipt/finalmarker;no freshgiantindexhashrerun inthis closure',fresh_full_SHA_in_this_closure=False))
mp=checked(pub['master']);assert ref(mp)['sha256']=='f8b09a2259393fd4147fb549520634aa3552d628b94059c0af72f2c669c69c1b';m=json.loads(mp.read_text());assert m['serialized_tasks']==81054784 and len(m['shards'])==82141 and m['total_shard_bytes']==37333973228;assert m['split_counts']==dict(train=81034656,test=10052,validation=10076);assert m['prior_manifest_sha256']==p['prior_master']['sha256'];assert m['additional_whole_RNAcentral_source_and_task_proof']==p['all_source_and_new_task_proof'];assert m['additional_whole_RNAcentral_native_prior_manifest']==p['component'];assert m['final_full_training_dataset_complete'] is False and m['substantive_training_authorized'] is False and m['full_dataset_readiness_signature'] is None and m['global_leakage_safe'] is False
log=B/'union-master.log';last=[v for v in log.read_text().splitlines() if v.strip()][-1];marker='COMPLETE_FULL_RNACENTRAL_MASTER ';assert last.startswith(marker);actual=json.loads(last[len(marker):]);assert actual['master']==pub['master'] and actual['views']==pub['views']==81054784 and actual['shards']==pub['shards']==82141 and actual['task_bytes']==pub['task_bytes']==37333973228;assert not any((Path('/proc')/str(pid)).exists() for pid in [4086786,4086787]);s1=sessions();s2=sessions();assert s1==s2==[]
sidecars=[];prefix=str((U/'preserved-prior-extra-groups').relative_to(R))+'/'
for source_shard,v in m['effective_group_lookup'].items():
 if v['path'].startswith(prefix):
  path=checked(v);assert pq.ParquetFile(path).metadata.num_rows==v['rows'];sidecars.append(dict(source_shard=source_shard,**v))
assert len(sidecars)==len({v['path'] for v in sidecars})==3201;O.mkdir();ledger=O/'new-versioned-sidecar-hash-ledger.json';ledger.open('x').write(json.dumps(dict(sidecars=sidecars,files=3201,total_file_bytes=sum(v['bytes'] for v in sidecars),full_physical_sidecar_rows=sum(v['rows'] for v in sidecars),actual_modified_rows_from_verifiedphase=6273790,note='whole sidecar files preserve unmodified rows/columns too; physical row count is not changed-row count'),indent=2)+'\n');script=R/'scripts/data_intake_full_rnacentral_union_master.py';assert ref(script)['sha256']=='07a8b18428d7ecfd8a597a3cc3f8d61d5abfa4625b19e999a71d5c1e840c244f';main=json.loads((R/'planning/data_intake_progress.json').read_text());assert main['current_master_corpus_manifest']==p['prior_master']['path'];now=datetime.datetime.now(datetime.timezone.utc);closure=dict(schema='mora3.WholeRNAC2NormalStagedMasterClosure/v1',created_utc=now.isoformat(),staged_master=ref(mp),union_proof=ref(proofpath),publication_staging_receipt=ref(pubpath),source_task_proof=p['all_source_and_new_task_proof'],baseline_master=p['prior_master'],component=p['component'],census=p['census'],clock=p['clock'],execution_approval=p['execution_approval'],code=ref(script),launch=ref(B/'downstream-launches/union-master.json'),runtime=ref(B/'union-master-initial-runtime-verification-v1.json'),normal_log=ref(log),final_marker_exact_counts_and_master=True,old_timeout_worker_identity=dict(timeout_pid=4086786,timeout_start_ticks=21219084,worker_pid=4086787,worker_start_ticks=21219084,SID=4086786),owners_absent=True,SID_members_twice=[s1,s2],giant_index_open_close_hash_evidence=index_stats,new_sidecar_ledger=ref(ledger),sidecars_all3201_fullSHA_size_footer_rechecked=True,counts=dict(views=81054784,shards=82141,task_bytes=37333973228,splits=m['split_counts'],new_RNA_views=54732469,reused_old_Rfam_and_pilot=6273790),native_RNA_source_evidence_units=dict(source_occurrences=61026468,new_native_symbols=46211589632,old_RNA_pilot_symbols=350334701,RNA_family_target_symbols=46561924333,actual_model_tokens=None,new_independent_experiments=0),current_master_pointer_still_v22=True,publisher_invoked=False,model_use=False,mixture_activated=False,full_dataset_ready=False,substantive_training=False,seconds=time.monotonic()-start);fp=O/'receipt.json';fp.open('x').write(json.dumps(closure,indent=2)+'\n');print(json.dumps(dict(closure=ref(fp),master=ref(mp),sidecar_ledger=ref(ledger),sidecar_file_bytes=sum(v['bytes'] for v in sidecars),sidecar_physical_rows=sum(v['rows'] for v in sidecars),seconds=closure['seconds']),indent=2))
