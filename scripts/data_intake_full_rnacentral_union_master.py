"""Full RNA master union only AFTER complete independent source/task proof.

No original task or group file is overwritten. Old effective groups receive
new versioned sidecars only for exact proven RNA reference identity aliases.
"""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,os,resource,sqlite3,sys,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_full_rnacentral_state import NativeIdentityIndex
from data_intake_full_rnacentral_tasks import native_effective_groups
from data_intake_full_rnacentral_proof_helpers import overlay_group_table_preserving_columns
B=R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while x:=f.read(4*2**20):h.update(x)
 return h.hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p
 if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('immutable source hash/size mismatch')
 return p
def write(p,v):p.open('x').write(json.dumps(v,indent=2)+'\n')
def task_digest(text):
 if not isinstance(text,str) or len(text)!=64:return None
 try:
  value=bytes.fromhex(text)
  return value if len(value)==32 and value.hex()==text else None
 except ValueError:return None

def validate_union_approval(a,expected):
 if a.get('approved') is not True or a.get('phase')!='union-master' or a.get('scope')!='whole-native-RNAcentral-master-union-only':raise ValueError('separate exact master-union review required')
 for k,v in expected.items():
  if a.get(k)!=v:raise ValueError('master-union input/dependency review mismatch: '+k)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--approval',type=Path,required=True);args=parser.parse_args();approval=args.approval if args.approval.is_absolute() else R/args.approval;pp=B/'build-plan.json';plan=json.loads(pp.read_text());cp=B/'census/manifest.json';c=json.loads(cp.read_text());mp=B/'materialized/manifest.json';component=json.loads(mp.read_text());proofpath=B/'independent-source-task-proof-v1/receipt.json';sourceproof=json.loads(proofpath.read_text());clockpath=B/'build-clock.json';clock=json.loads(clockpath.read_text());names=[Path(__file__).name,'data_intake_full_rnacentral_state.py','data_intake_full_rnacentral_tasks.py','data_intake_full_rnacentral_proof_helpers.py','data_intake_rnacentral_prior.py','data_intake_nucleotide_prior.py','data_intake_corpus.py'];expected=dict(plan_sha256=sha(pp),census_manifest_sha256=sha(cp),component_manifest_sha256=sha(mp),source_task_proof_sha256=sha(proofpath),clock_sha256=sha(clockpath),script_sha256=sha(Path(__file__)),resource_coordination_sha256=plan['resource_coordination']['sha256'],dependency_refs=[ref(R/'scripts'/n) for n in names[1:]]);validate_union_approval(json.loads(approval.read_text()),expected);assert sourceproof['status']=='verified_source_and_new_tasks_pending_inherited_master_group_union' and sourceproof['component']==ref(mp) and sourceproof['census']==ref(cp) and sourceproof['baseline_master']==plan['baseline_master'];M=check(plan['baseline_master']);master=json.loads(M.read_text());old_tasks=master['serialized_tasks'];oldbytes=master['total_shard_bytes'];O=B/'independent-master-union-v23';assert not O.exists();os.sched_setaffinity(0,{26,27});nice=os.getpriority(os.PRIO_PROCESS,0)
 if nice<10:os.nice(10-nice)
 pa.set_cpu_count(1);pa.set_io_thread_count(1);O.mkdir();(O/'code').mkdir();(O/'preserved-prior-extra-groups').mkdir()
 for name in names:(O/'code'/name).open('xb').write((R/'scripts'/name).read_bytes())
 start=time.monotonic()
 def guard(reserve=2**30):
  st=os.statvfs(R)
  if st.f_bavail*st.f_frsize-reserve<64*2**30 or time.time()>clock['deadline_epoch'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30-128*2**20:raise RuntimeError('whole-build deadline/hardware/RSS guard; retain partial union')
 guard();statepath=check(c['native_identity_URS_index']);state=NativeIdentityIndex.open_readonly(statepath);state.db.execute('PRAGMA cache_size=-131072');taskpath=check(component['task_identity_serialization_index']);tasks=sqlite3.connect('file:'+str(taskpath)+'?mode=ro&immutable=1',uri=True);tasks.row_factory=sqlite3.Row;witness=sqlite3.connect(O/'old-reused-task-witness.sqlite');witness.execute('CREATE TABLE matched_reused(task_id BLOB PRIMARY KEY) WITHOUT ROWID');held=set(json.loads(check(plan['heldout_group_snapshot']).read_text())['group_ids']);actual_old_count=old_data_bytes=old_group_bytes=overlay_rows=overlay_files=collisions=0;old_RNA_letters=0;newpaths={x['path'] for x in component['shards']};assert len(newpaths)==len(component['shards']) and not newpaths&{x['path'] for x in master['shards']};old_RNA_source_views=master['serialized_distinct_record_counts_by_source']['RNAcentral-reference-priors']
 for number,sh in enumerate(master['shards']):
  guard();p=check(sh);pf=pq.ParquetFile(p);assert pf.metadata.num_rows==sh['rows'];gp=check(master['effective_group_lookup'][sh['path']]);assert pq.ParquetFile(gp).metadata.num_rows==sh['rows'];old_data_bytes+=p.stat().st_size;old_group_bytes+=gp.stat().st_size;cols=['task_id']
  for name in ['source_family','split','record_id']:
   if name in pf.schema_arrow.names:cols.append(name)
  if sh.get('source_family')=='RNAcentral-reference-priors':cols.append('full_axis_values')
  updates={}
  for batch in pf.iter_batches(columns=cols,batch_size=4096):
   rows=batch.to_pylist();actual_old_count+=len(rows);encoded=[task_digest(r['task_id']) for r in rows];found={}
   for first in range(0,len(rows),500):
    chunk=[x for x in encoded[first:first+500] if x is not None]
    if chunk:
     sql='SELECT * FROM tasks WHERE task_id IN ('+','.join('?' for _ in chunk)+')'
     found.update({bytes(r['task_id']):dict(r) for r in tasks.execute(sql,chunk)})
   for row,digest in zip(rows,encoded):
    family=row.get('source_family',sh.get('source_family'));split=row.get('split',sh.get('split'))
    if family=='RNAcentral-reference-priors':
     assert not row['record_id'].startswith('RNAcentral28-native-sequence:')
     if 'full_axis_values' in row:old_RNA_letters+=row['full_axis_values']
    entry=found.get(digest)
    if entry is None:continue
    if entry['reuse_kind']=='new':collisions+=1;raise ValueError('new RNA task ID already exists in inherited master')
    assert split=='train' and isinstance(family,str) and (family.startswith('Rfam') if entry['reuse_kind']=='Rfam_existing' else family=='RNAcentral-reference-priors');witness.execute('INSERT INTO matched_reused VALUES (?)',(digest,));s=state.lookup(bytes(entry['native_sha']));assert s['task_id']==row['task_id'] and s['reuse_kind']==entry['reuse_kind'];added=native_effective_groups(s,state.identifiers(bytes(entry['native_sha'])));assert not set(added)&held;updates[row['task_id']]=added
  if updates:
   original=pq.read_table(gp);assert set(updates)<=set(original.column('task_id').to_pylist());table,changed=overlay_group_table_preserving_columns(original,updates);assert changed==len(updates);target=O/'preserved-prior-extra-groups'/f'{overlay_files:05d}.parquet';pq.write_table(table,target,compression='zstd');back=pq.read_table(target);assert back.schema==table.schema and back.to_pylist()==table.to_pylist();assert back.drop(['effective_groups_json']).to_pylist()==original.drop(['effective_groups_json']).to_pylist();master['effective_group_lookup'][sh['path']]=ref(target,rows=sh['rows']);overlay_rows+=changed;overlay_files+=1
  witness.commit()
  if number%1000==0:print('FULL_RNA_INHERITED_MASTER_PROOF',number+1,'IDs',actual_old_count,'overlaid',overlay_rows,'seconds',round(time.monotonic()-start,1),flush=True)
 assert actual_old_count==old_tasks and old_data_bytes==oldbytes and collisions==0;expected_reused=tasks.execute('SELECT count(*) FROM tasks WHERE reuse_kind!=?',('new',)).fetchone()[0];matched=witness.execute('SELECT count(*) FROM matched_reused').fetchone()[0];assert matched==expected_reused==overlay_rows;witness.commit();witness.close();newbytes=newgroupbytes=newrows=0
 for f in component['shards']:
  guard();p=check(f);g=check(component['effective_group_lookup'][f['path']]);assert pq.ParquetFile(p).metadata.num_rows==pq.ParquetFile(g).metadata.num_rows==f['rows'];newbytes+=p.stat().st_size;newgroupbytes+=g.stat().st_size;newrows+=f['rows']
 assert newrows==component['counts']['new_task_views_serialized']==sourceproof['all_new_task_payload_source_target_ID_location_axis_group_readback'] and newbytes==component['task_shard_bytes'] and newgroupbytes==component['effective_group_bytes'];blocked_occurrences=state.db.execute('SELECT coalesce(sum(s.occurrences),0) FROM (SELECT DISTINCT native_sha FROM holdout_blocked) b JOIN sequences s USING(native_sha)').fetchone()[0];tasks.close();state.close();assert ref(statepath)==c['native_identity_URS_index'] and ref(taskpath)==component['task_identity_serialization_index'] and ref(M)==plan['baseline_master'];receipt=dict(schema='mora3.FullRNAcentralIndependentMasterUnion/v23',status='verified',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_master=ref(M),component=ref(mp),census=ref(cp),all_source_and_new_task_proof=ref(proofpath),all_inherited_data_group_hashes_sizes_footer_rows_checked=True,all_inherited_task_IDs_scanned=actual_old_count,actual_new_old_task_ID_intersections=collisions,preserved_Rfam_and_old_pilot_exact_tasks_reused=matched,preserved_task_group_sidecars_new_versions=overlay_files,new_sidecar_rows_with_exact_source_alias_additions=overlay_rows,all_old_task_and_group_files_unchanged=True,new_task_views=newrows,new_data_bytes=newbytes,new_group_bytes=newgroupbytes,known_newtrain_oldheldout_group_intersections=0,all_known_groups_not_exhaustive_homology_or_new_study_evaluation=True,old_RNA_target_native_symbols=old_RNA_letters,new_RNA_target_native_symbols=sourceproof['new_distinct_target_native_symbols'],all_RNA_family_target_native_symbols=old_RNA_letters+sourceproof['new_distinct_target_native_symbols'],old_RNA_source_views=old_RNA_source_views,neural_tokens=None,actual_optimizer_use=None,clock=ref(clockpath),execution_approval=ref(approval),seconds=time.monotonic()-start)
 write(O/'receipt.json',receipt);master['shards']+=component['shards'];master['effective_group_lookup'].update(component['effective_group_lookup']);master['serialized_tasks']+=newrows;master['total_shard_bytes']+=newbytes;master['split_counts']['train']+=newrows;master['native_schema_counts']['mora3.CorpusTask/v1']+=newrows;master['adapter_support_counts']['mora3.CorpusTask/v1;model_adapter_supported=False']+=newrows;master['serialized_distinct_record_counts_by_source']['RNAcentral-reference-priors']+=newrows;master.update(build_id='20261101-whole-RNAcentral-native-corpus-v23',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_manifest_sha256=sha(M),supersedes_for_consumption=str(M.relative_to(R)),source_components_base_manifest=str(M.relative_to(R)),additional_whole_RNAcentral_native_prior_manifest=ref(mp),additional_whole_RNAcentral_source_and_task_proof=ref(proofpath),verification=str((O/'receipt.json').relative_to(R)),final_full_training_dataset_complete=False,full_dataset_readiness_signature=None,substantive_training_authorized=False,global_leakage_safe=False,actual_whole_model_token_totals=None,expanded_scope_contracts_pending_full_readiness=plan['required_scope_extensions']);master['verification_layers'].append(str((O/'receipt.json').relative_to(R)));master['source_curation_manifests'].extend([ref(cp),ref(mp),ref(proofpath)])
 for entry in master['source_ledger']:
  if entry['source_family']=='RNAcentral-reference-priors':
   history={k:v for k,v in entry.items() if 'pilot' in k or k in ['note','admitted_records','serialized_tasks','reused_existing_Rfam_tasks']};entry.update(prior_pilot_history=history,retained_quality_pass_source_occurrences=61006261,source_occurrences_admitted_for_reference_prior=61006261-blocked_occurrences,admitted_distinct_source_native_reference_identities_across_prior_domains=c['identity_summary']['native_sequences']-c['distinct_native_sequences_with_known_heldout_group'],source_original_nonstandard_quarantines=20207,distinct_full_native_source_sequences=c['identity_summary']['native_sequences'],known_heldout_group_blocked_native_sequences=c['distinct_native_sequences_with_known_heldout_group'],admitted_records=old_RNA_source_views+newrows,serialized_tasks=old_RNA_source_views+newrows,reused_existing_Rfam_tasks=component['counts'].get('reused_Rfam_existing',0),reused_existing_RNAcentral_pilot_tasks=component['counts'].get('reused_RNAcentral_pilot_existing',0),note='Whole original active+inactive native source scanned; all eligible distinct native reference priors prepared, old pilot/Rfam reused exactly once. Nonstandard source values and the13 release-note discrepancy retained. Native case/T-U, unknown modifications/material/taxon/function and source-status-not-activity roles preserved.',count_unit='source FASTA occurrences vs literal native sequence reference tasks; not independent experiments; Rfam reuse nonadditive')
 master['contexts']['RNAcentral-reference-priors'].update(rows=old_RNA_source_views+newrows,native_axis_values='complete source-native targets with case/T-U/IUPAC retained, no8192crop',source_full_build=ref(O/'receipt.json'),target_native_symbols=old_RNA_letters+sourceproof['new_distinct_target_native_symbols'],actual_model_tokens=None,source_status='audit-only, not activity or negative label');master['open_gates'].append('Whole RNA reference source is prepared; homology/taxon/study heldouts, physical material/modification identity, actual native neural/full-context throughput/token ledger and intended mixture remain unresolved. Expanded ADMET/ProCyon/X-Pert/retrosynthesis and other required source scope must be complete before substantive training.');master['mixture_proposal']['RNAcentral_reference_prior_allocation']=dict(status='provisional unactivated; full native reference-source construction verified',weight=None,source_occurrences=61026468,new_tasks=newrows,preserved_pilot_tasks=old_RNA_source_views,old_Rfam_prior_reuse=component['counts'].get('reused_Rfam_existing',0),scope='source-native reference symbols, no measured biological function/ADMET/response claim; full native axes retained; no task-count-derived weight')
 target=O/'master-corpus-manifest-v23-whole-RNAcentral-native-v1.json';guard();write(target,master);write(O/'publication-receipt.json',dict(master=ref(target),proof=ref(O/'receipt.json'),source_task_proof=ref(proofpath),views=master['serialized_tasks'],shards=len(master['shards']),task_bytes=master['total_shard_bytes'],split_counts=master['split_counts'],full_dataset_ready=False,substantive_training_authorized=False,queue_progress_handoff_not_yet_updated=True));guard();print('COMPLETE_FULL_RNACENTRAL_MASTER',json.dumps(dict(master=ref(target),views=master['serialized_tasks'],shards=len(master['shards']),task_bytes=master['total_shard_bytes'])),flush=True)
if __name__=='__main__':main()
