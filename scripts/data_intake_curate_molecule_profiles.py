from pathlib import Path
import json,sys,hashlib,collections,datetime,time,copy,shutil
import pyarrow as pa
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import chemical_identity,CorpusTask,canonical,read_corpus_tasks
from data_intake_molecule_profile import molecule_profile_task
B=R/'outputs/data-intake/20261031-rosetta-molecule-profile-v1';out=B/'curated-corpus-v1';out.mkdir();(out/'shards/train').mkdir(parents=True);(out/'effective-groups').mkdir();start=time.perf_counter();budget=4*2**30;assert shutil.disk_usage(R).free>200*2**30+budget
prior_path=R/'outputs/data-intake/20261031-chembl-logd-context-v1/master-corpus-manifest-v7.json';prior=json.loads(prior_path.read_text());base=R/'outputs/data-intake/20261031-subagent/curated/cpg0003-rosetta-profile-links-v1';audit_path=B/'identity-audit-v2/source-identity-audit.parquet';identities={r['record_id']:r for r in pq.read_table(audit_path).to_pylist()};metadata={}
for modality in ['CellPainting','L1000']:
 p=base/f'CDRPBIO-BBBC036-Bray--{modality}--metadata.parquet'
 metadata.update({r['record_id']:r for r in pq.read_table(p).to_pylist()})
heldout=set()
for split in ['validation','test']:
 for row in read_corpus_tasks(prior_path,split=split,supported_only=False):heldout.update(json.loads(row['group_ids_json']))
def receipt(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),**kw)
shards=[];lookup={};buffer=[];newcount=0;newbytes=0;axis_counts=collections.Counter();sourcecounts=collections.Counter();identitygroups=set();decisions=[];admitted_records=collections.defaultdict(set);mutationchecks=0;profiles_seen=0;newtaskids=set()
def flush():
 global newbytes
 if not buffer:return
 i=len(shards);p=out/'shards/train'/f'{i:05d}.parquet';pq.write_table(pa.Table.from_pylist(buffer),p,compression='zstd');back=pq.read_table(p).to_pylist();assert back==buffer
 for r in back:
  assert hashlib.sha256(r['native_payload']).hexdigest()==r['native_payload_sha256'];assert CorpusTask.from_bytes(r['native_payload']).canonical_bytes()==r['native_payload']
  payload=json.loads(r['native_payload']);target_or_input=payload['targets'] if r['task_name']=='molecule_to_profile' else payload['conditions'];assert hashlib.sha256(canonical(target_or_input['source_profile_array'])).hexdigest()==r['original_array_sha256']
 shards.append(receipt(p,rows=len(buffer),split='train',source_family=buffer[0]['source_family'] if len({r['source_family'] for r in buffer})==1 else None));newbytes+=p.stat().st_size;assert newbytes<budget
 gp=out/'effective-groups'/f'{i:05d}.parquet';gr=[dict(task_id=r['task_id'],effective_groups_json=r['group_ids_json']) for r in buffer];pq.write_table(pa.Table.from_pylist(gr),gp,compression='zstd');assert pq.read_table(gp).to_pylist()==gr;lookup[str(p.relative_to(R))]=receipt(gp,rows=len(gr));buffer.clear()
for family,modality in [('Rosetta-CellPainting','CellPainting'),('Rosetta-L1000','L1000')]:
 for row in read_corpus_tasks(prior_path,source=family,supported_only=True):
  if not row['record_id'].startswith('CDRPBIO-BBBC036-Bray:'):continue
  profiles_seen+=1;ident=identities[row['record_id']];mdrow=metadata[row['record_id']];md=json.loads(mdrow['source_metadata_json']);smiles=ident['canonical_isomeric_smiles'];groups=set(json.loads(row['group_ids_json']));status='admitted';reason=None
  if not smiles:status='not_molecule_conditioned';reason='control_or_unresolved_exact_source_chemical_identity'
  if smiles:
   chem=chemical_identity(smiles);groups.update([chem['molecule_group'],chem['scaffold_group'],'source_chemical_sample:'+ident['source_sample_id']]);conflict=sorted(groups&heldout)
   if conflict:status='excluded_heldout_chemistry';reason=json.dumps(conflict)
  decisions.append(dict(record_id=row['record_id'],modality=modality,status=status,reason=reason,source_identity_join=ident['molecular_identity_source'],source_sample_id=ident['source_sample_id'],source_profile_native_payload_sha256=row['native_payload_sha256']))
  if status!='admitted':continue
  native=json.loads(row['native_payload']);assert len(native['arrays'])==1 and native['roles']==[['profile','target']];array=native['arrays'][0];assert len(array['values'])==(1783 if modality=='CellPainting' else 977)
  context=json.loads(mdrow['source_context_json']);cell_line=context.get('cell_line');source_audit=dict(source_profile_record_id=row['record_id'],source_profile_native_payload_sha256=row['native_payload_sha256'],source_metadata=md,source_context=context,source_chemical_sample_id=ident['source_sample_id'],molecular_identity_join=ident['molecular_identity_source'],chemical_source_evidence='exact source CPD_SMILES;identityfactonly,notCPandL1000matchedcellsorassays',image_context='cooperative-segmentation-v1/curated-v3 sameplatewell field context exists;crop/fullfieldregistration not asserted' if row['record_id']=='CDRPBIO-BBBC036-Bray:CellPainting:5' else None,source_scale=None,mask_available_in_profile=False)
  original_array_sha=hashlib.sha256(canonical(array)).hexdigest();admitted_records[modality].add(row['record_id']);newfamily='Rosetta-CDRP-molecule-profile-'+modality
  for direction in ['molecule_to_profile','profile_to_molecule']:
   task=molecule_profile_task(array,smiles,modality,direction,cell_line=cell_line,audit=source_audit);raw=task.canonical_bytes()
   if direction=='molecule_to_profile':
    mutated=copy.deepcopy(array);mutated['values']=[v+.125 for v in mutated['values']];other=molecule_profile_task(mutated,smiles,modality,direction,cell_line=cell_line,audit=dict(QC='mutated',source_sample='mutated',MOA='mutated'))
   else:other=molecule_profile_task(array,'CCN' if smiles!='CCN' else 'CCC',modality,direction,cell_line=cell_line,audit=dict(source_sample='mutated'))
   assert task.input_bytes()==other.input_bytes() and task.canonical_bytes()!=other.canonical_bytes();mutationchecks+=1
   taskid=hashlib.sha256(canonical([row['record_id'],direction,hashlib.sha256(raw).hexdigest()])).hexdigest();assert taskid not in newtaskids;newtaskids.add(taskid);newcount+=1;sourcecounts[newfamily]+=1;axis_counts[str(len(array['values']))]+=1
   buffer.append(dict(task_id=taskid,record_id=row['record_id'],source_family=newfamily,source_revision=row['source_revision'],source_sha256=row['source_sha256'],task_name=direction,native_schema='mora3.CorpusTask/v1',native_payload=raw,native_payload_sha256=hashlib.sha256(raw).hexdigest(),split='train',split_policy='inherits_source_profile_train_allocation;chemicalheldoutchecked;provisionalmixture',group_ids_json=json.dumps(sorted(groups)),evidence_class='source_computed_normalized_CellPainting_profile' if modality=='CellPainting' else 'source_normalized_L1000_profile;perfeaturemeasured_vs_inferred_unresolved;notUMI',admitted_for_task=True,model_adapter_supported=False,admission_receipt_json=json.dumps(dict(source_rights='CPG0003CC0',source_profile_native_sha256=row['native_payload_sha256'],identity_audit=str(audit_path.relative_to(R)),source_identity_scope=ident['molecular_identity_source'],conditional_scope='observationalsourceprofileormoleculecompletion;notphysicaldoseconditionedcausality',no_same_cell_join=True,full_feature_axis=True)),missing_context_json=json.dumps(json.loads(raw)['missing']),full_axis_values=len(array['values']),condition_context_json=json.dumps(dict(explicit_cell_line=cell_line,physical_dose=None,exposure_time=None,image_available_for_task=False,mask_available_for_task=False,source_generated_data=False,profile_target_evidence=source_audit['chemical_source_evidence'])),original_array_sha256=original_array_sha,original_source_native_payload_sha256=row['native_payload_sha256']))
   if len(buffer)==64:flush()
  if len(admitted_records[modality])%2000==0:print('CURATING',modality,len(admitted_records[modality]),newcount,round(time.perf_counter()-start,2),flush=True)
 flush()
# Every source metadata row has a disposition, including profiles already excluded in v7.
seen={r['record_id'] for r in decisions}
for record,identity in identities.items():
 if record not in seen:decisions.append(dict(record_id=record,modality=identity['modality'],status='not_in_prior_admitted_native_pool',reason='prior_source_profile_admission_or_heldout_exclusion',source_identity_join=identity['molecular_identity_source'],source_sample_id=identity['source_sample_id'],source_profile_native_payload_sha256=None))
pq.write_table(pa.Table.from_pylist(decisions),out/'profile-admission-decisions.parquet',compression='zstd')
rec=dict(schema='mora3.Rosetta_molecule_profile_observational_corpus.v1',source_total_CDRP_profiles=28051,source_noncontrol_chemical_identity_profiles=18463,prior_admitted_profile_rows_scanned=profiles_seen,admitted_distinct_source_profiles={k:len(v) for k,v in admitted_records.items()},serialized_tasks=newcount,source_task_counts=dict(sourcecounts),task_axis_counts=dict(axis_counts),direction_count_per_profile=2,profile_decision_counts=dict(collections.Counter(r['status'] for r in decisions)),shards=shards,shard_bytes=newbytes,effective_group_lookup=lookup,all_new_native_payloads_roundtrip=True,all_new_full_source_arrays_canonical_hashes_exact=True,target_or_target_molecule_and_audit_mutation_input_invariance_checks=mutationchecks,source_chemical_identity_conflicts=0,source_profile_counts_shared_with_existing_unconditional_tasks=True,independent_new_experiments=0,no_raw_downloads=True,rights='CC0-1.0 CPG0003 source profiles+chemical metadata',same_cell_or_matched_CP_L1000_pair=False,physical_dose_or_time_conditioning=False,explicit_cell_line_only=True,profiles_do_not_provide_images_or_masks=True,generated_examples=False,protein_sequence_admissions=0,model_adapter_supported=False,optimizer_steps=0,global_leakage_safe=False,seconds=time.perf_counter()-start)
with (out/'receipt.json').open('x') as f:json.dump(rec,f,indent=2)
print('NEW_TASKS_COMPLETE',json.dumps({k:rec[k] for k in ['admitted_distinct_source_profiles','serialized_tasks','source_task_counts','task_axis_counts','profile_decision_counts','shard_bytes','seconds']}),flush=True)
# Complete union integrity and effective group audit, preserving all prior native bytes.
m=copy.deepcopy(prior);m['shards']+=shards;m['effective_group_lookup'].update(lookup);seen=set();splits=collections.Counter();schemas=collections.Counter();sources=collections.defaultdict(set);support=collections.Counter();train=set();held=set();checkedbytes=0;groupbytes=0
for i,s in enumerate(m['shards']):
 p=R/s['path'];data=p.read_bytes();assert len(data)==s['bytes'] and hashlib.sha256(data).hexdigest()==s['sha256'];checkedbytes+=len(data);pf=pq.ParquetFile(p);fields=[x for x in ['task_id','record_id','split','source_family','native_schema','model_adapter_supported'] if x in pf.schema_arrow.names];rows=pf.read(columns=fields).to_pylist();assert len(rows)==s['rows']
 g=m['effective_group_lookup'][s['path']];gp=R/g['path'];data=gp.read_bytes();assert len(data)==g['bytes'] and hashlib.sha256(data).hexdigest()==g['sha256'];groupbytes+=len(data);gr=pq.read_table(gp).to_pylist();groups={r['task_id']:r['effective_groups_json'] for r in gr};assert len(gr)==g['rows']==len(rows)==len(groups);assert set(groups)=={r['task_id'] for r in rows}
 for r in rows:
  assert r['task_id'] not in seen;seen.add(r['task_id']);splits[r['split']]+=1;schemas[r['native_schema']]+=1;sources[r['source_family']].add(r['record_id']);sup=r.get('model_adapter_supported',r['native_schema']=='mora3.ObservationalSciPlexTask/v1');support[r['native_schema']+';model_adapter_supported='+str(sup)]+=1;(train if r['split']=='train' else held).update(json.loads(groups[r['task_id']]))
 if (i+1)%400==0:print('UNION',i+1,len(seen),round(time.perf_counter()-start,2),flush=True)
assert not train&held
verify=dict(schema='mora3.multisource_union_verification.v8',complete_data_shard_hashes_verified=True,complete_group_sidecar_hashes_verified=True,all_taskIDs_unique=True,serialized_tasks=len(seen),shards=len(m['shards']),split_counts=dict(splits),data_bytes=checkedbytes,effective_group_bytes=groupbytes,defined_train_heldout_group_conflicts=0,global_leakage_safe=False,prior_native_payloads_unchanged=True,new_full_array_roundtrips_and_all_task_targetmutation_checked=True,final_full_training_dataset_complete=False,optimizer_steps=0,seconds=time.perf_counter()-start)
with (B/'union-verification-v8.json').open('x') as f:json.dump(verify,f,indent=2)
m.update(build_id='20261031-rosetta-molecule-profile-corpus-v8',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='versioned_multisource_candidate_with_source_specific_task_admission',serialized_tasks=len(seen),split_counts=dict(splits),total_shard_bytes=checkedbytes,native_schema_counts=dict(schemas),adapter_support_counts=dict(support),serialized_distinct_record_counts_by_source={k:len(v) for k,v in sources.items()},prior_manifest_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),supersedes_for_consumption=str(prior_path.relative_to(R)),additional_molecule_profile_admission_receipt=str((out/'receipt.json').relative_to(R)))
for modality,records in admitted_records.items():m['source_ledger'].append(dict(source_family='Rosetta-CDRP-molecule-profile-'+modality,raw_available_source_profiles=21122 if modality=='CellPainting' else 6929,exact_source_molecule_identity_candidates=15012 if modality=='CellPainting' else 3451,admitted_records=len(records),serialized_tasks=2*len(records),count_unit='two observational directions per existing source profile;not independent new experiments',release='CPG0003 hash pinned snapshot20261031',license='CC0-1.0',lineage='same source profiles already serialized unconditional;source exact chemical sample ID identity only;noCP_L1000assaypair',feature_axis=1783 if modality=='CellPainting' else 977,physical_dose_time_conditions=False,model_adapter_supported=False))
m['verification_layers']+=[str((out/'receipt.json').relative_to(R)),str((B/'union-verification-v8.json').relative_to(R))];m['mixture_proposal']['new_molecule_profile_allocation']=dict(status='provisional_unactivated;sourceprofilesoverlapexistingunconditionalobjectives',weight=None,requires='modelconditionalreaders,generativeandpredictivetargetexclusion,measuredfullaxisquality/throughputandmixtureevaluation');m['optimizer_steps_from_this_build']=0
with (B/'master-corpus-manifest-v8.json').open('x') as f:json.dump(m,f,indent=2)
print('COMPLETE_V8',json.dumps(verify),flush=True)
