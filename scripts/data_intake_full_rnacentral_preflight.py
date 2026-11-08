"""Prepare the exact whole-RNA census package; this NEVER launches the census."""
from pathlib import Path
import collections,datetime,hashlib,json,os,resource,sys,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1'
H=R/'outputs/data-sources/20261031-legacy-mint-v1/handoff-RNAcentral28-full-native-verified-v1.json'
M=R/'outputs/data-intake/20261101-LINCS-corpus-v1/independent-master-union-v22/master-corpus-manifest-v22-LINCS-qualified-source-coordinate-v1.json'
RESOURCE=R/'outputs/monitoring/resources-20261101T105608Z-whole-RNAcentral-hardware-v1.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while x:=f.read(4*2**20):h.update(x)
 return h.hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p
 if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('immutable source hash/size mismatch: '+str(p))
 return p
def write(p,v):p.open('x').write(json.dumps(v,indent=2)+'\n')
def main():
 start=time.monotonic();os.sched_setaffinity(0,{26,27});os.nice(10);pa.set_cpu_count(1);pa.set_io_thread_count(1);assert not B.exists();assert sha(RESOURCE)=='0665076e6a2e41f651d955b95fbfbf8bfd3f56b41e5dc98546688190c0c2663f';assert sha(H)=='e8694fab174e9df1bb99246cd95e5f23ca41f10fbfe20222b1168be7e12a7567';assert sha(M)=='82ef478d53ccbd2bb8a87607b1ce5dd35a6a0f58f6afa7d912aa214129c1e6f2'
 def guard():
  st=os.statvfs(R)
  if st.f_bavail*st.f_frsize<65*2**30 or time.monotonic()-start>1800 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30-128*2**20:raise RuntimeError('preflight hardware/time/RSS guard; preserve outputs')
 guard();B.mkdir();O=B/'preflight';O.mkdir();(O/'code').mkdir();h=json.loads(H.read_text());m=json.loads(M.read_text());assert json.loads((M.parent/'publication-receipt.json').read_text())['master']==ref(M);source_files=[];raw=[];source_manifests=[];footers=0
 required=['source_record_id','source_record_index','source_payload','source_identifier','native_header','native_sequence','native_length','native_sequence_sha256','uppercase_sequence_sha256','sequence_alphabet_pass','native_T_count','native_U_count']
 for component,c in h['components'].items():
  mp=Path(c['native_manifest']);assert sha(mp)==c['native_manifest_sha256'];native=json.loads(mp.read_text());source_manifests.append(ref(mp));p=Path(c['source_path']);assert sha(p)==c['source_sha256'];raw.append(ref(p,component=component,raw_gzip_native_QC_inherited=True));observed=0
  for f in native['native_payloads']:
   p=Path(f['path']);assert p.stat().st_size==f['bytes'];pf=pq.ParquetFile(p);assert pf.metadata.num_rows==f['rows'] and set(required)<=set(pf.schema_arrow.names);source_files.append(dict(path=str(p.relative_to(R)),bytes=f['bytes'],sha256=f['sha256'],rows=f['rows'],component=component,original_FASTA=c['source_path']));observed+=f['rows'];footers+=1
  assert observed==c['occurrences'];guard()
 assert sum(f['rows'] for f in source_files)==61026468
 # Verify only actual source-defined heldout IDs/groups, with no artificial RNA split.
 held=set();count=collections.Counter();selected_group_refs=[]
 for sh in m['shards']:
  split=sh.get('split')
  if split=='train':continue
  p=R/sh['path']
  if split in ('validation','test'):
   heldids=None;count[split]+=sh['rows']
  else:
   ids=pq.read_table(p,columns=['task_id','split']).to_pylist();heldids={r['task_id'] for r in ids if r['split'] in ('validation','test')}
   for r in ids:
    if r['task_id'] in heldids:count[r['split']]+=1
   if not heldids:continue
  gr=m['effective_group_lookup'][sh['path']];gp=check(gr);rows=pq.read_table(gp).to_pylist();assert len(rows)==sh['rows'];selected_group_refs.append(gr)
  for r in rows:
   if heldids is None or r['task_id'] in heldids:held.update(json.loads(r['effective_groups_json']))
  guard()
 assert dict(count)=={k:v for k,v in m['split_counts'].items() if k in ('validation','test')}
 heldpath=O/'actual-baseline-heldout-groups.json';write(heldpath,dict(schema='mora3.FullRNAActualBaselineHeldoutSnapshot/v1',baseline_master=ref(M),source_defined_task_split_counts=dict(count),group_ids=sorted(held),selected_effective_group_refs=selected_group_refs,scope='known exact/declared groups only; not exhaustive homology, no newly invented RNA holdout'))
 pilot_manifest=R/'outputs/data-intake/20261101-RNAcentral-corpus-pilot-v1/manifest.json';pilot=json.loads(pilot_manifest.read_text());oldpilot=check(pilot['new_identity_and_group_index']);oldmanifest=R/'outputs/data-intake/20261101-nucleotide-corpus-v1/full-v1/manifest.json';old=json.loads(oldmanifest.read_text());oldrfam=check(old['generated_identity_index']);conservation=R/'outputs/data-intake/20261101-RNAcentral-full-state-conservation-v1/receipt.json';assert json.loads(conservation.read_text())['status']=='verified'
 names=['data_intake_full_rnacentral_preflight.py','data_intake_full_rnacentral_census.py','data_intake_full_rnacentral_state.py','data_intake_rnacentral_prior.py','data_intake_nucleotide_prior.py','data_intake_corpus.py'];code=[]
 for name in names:
  p=R/'scripts'/name;(O/'code'/name).open('xb').write(p.read_bytes());code.append(ref(p))
 q=json.loads((R/'configs/dataset_acquisition_queue.json').read_text());extensions={k:v['contract'] for k,v in q.get('scope_extensions',{}).items()};plan=dict(schema='mora3.FullRNAcentralConstructionPlan/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='prepared_exact_census_execution_review_required;not_launched',source_handoff=ref(H),baseline_master=ref(M),source_native_manifests=source_manifests,raw_source_refs=raw,source_files=source_files,source_occurrences=61026468,source_alphabet_pass=61006261,source_nonstandard_quarantines=20207,source_native_letters=47369240265,published_release_count_discrepancy=h['published_release_count_discrepancy'],old_Rfam_index=ref(oldrfam),old_pilot_index=ref(oldpilot),preserved_pilot_manifest=ref(pilot_manifest),all_same_pilot_conservation=ref(conservation),heldout_group_snapshot=ref(heldpath),code_refs=code,resource_coordination=ref(RESOURCE),CPU_affinity=[26,27],max_CPU=2,nice_minimum=10,max_peak_RSS_bytes=8*2**30,wrapper_RSS_reserve_bytes=128*2**20,whole_deadline_seconds=43200,clock_created_once_at_explicit_launch=True,free_buffer_bytes=64*2**30,artificial_storage_cap_bytes=None,GPU=0,network=False,source_reader_batch_rows=2048,source_Parquet_footer_rows_and_column_schema_checked=footers,full_typed_source_SHA_and_all_row_validation_phase='census; footer checks here are not whole-row admission',all_source_raw_gzip_SHA_rechecked=True,source_raw_gzip_native_parser_QC_inherited_from_exact_handoff=True,source_license='CC0-1.0',seed=0,native_identity='literal native sequence bytes+prior domain; case/T-U/RC guards never collapse identity',existing_prior_reuse='exact Rfam recipe first, then old RNAcentral pilot; no duplicated source task views',missing_modification_material_function_taxon_context='unknown, source described headers audit-only; no chemistry/ADMET/response/negative labels inferred',source_status_not_activity=True,split_recipe='all genuinely new eligible source priors train-only; full known baseline heldout guard; no artificial unseen split',phase_schedule=[dict(phase='census',script='scripts/data_intake_full_rnacentral_census.py',status='prepared_for_root_exact_execution_review',source_scope='all61026468sourceoccurrences',serialization=0),dict(phase='materialize',status='implementation_and_focused_checks_required_before_separate_launch',source_scope='all distinct eligible new native sequences, preserve full axes and reused-task group overlays'),dict(phase='independent_proof_and_master_union',status='implementation_and_full_exact_readback_required_before_any_new_master_admission',source_scope='all new source/task/group/ID/native-axis bytes plus exact old allocations and pilot reuse')],required_scope_extensions=extensions,final_full_dataset_ready=False,substantive_training_authorized=False,neural_tokens_or_optimizer_use=None)
 write(B/'build-plan.json',plan);guard();mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in ('MemAvailable','MemTotal')};st=os.statvfs(R);receipt=dict(schema='mora3.FullRNAcentralExactCensusPreflight/v1',status='verified_preflight_only_not_launched',plan=ref(B/'build-plan.json'),source_rows=sum(f['rows'] for f in source_files),source_files=footers,source_payload_rows_are_footer_evidence_only=True,raw_compressed_source_hashes_verified=raw,preserved_old_indices=[ref(oldrfam),ref(oldpilot)],same_pilot_conservation=ref(conservation),baseline_master=ref(M),actual_baseline_heldout_counts=dict(count),existing_heldout_group_count=len(held),host_memory=mem,free_bytes=st.f_bavail*st.f_frsize,execution_requires_root_review=True,seconds=time.monotonic()-start);write(O/'receipt.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
