"""Independent full native-bit/context/target readback of the LINCS component."""
from pathlib import Path
import collections,datetime,hashlib,json,os,shutil,struct,sys,time
import pyarrow as pa
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import CorpusTask,canonical
B=R/'outputs/data-intake/20261101-LINCS-corpus-v1';T=B/'curated-profile-tasks-v1/manifest.json';A=B/'metadata-admission-audit-v1/manifest.json';H=R/'outputs/data-sources/20261101-LINCS-Pilot1-source-v1/handoff-two-originals-full-native-verified-v1.json';O=B/'independent-native-task-proof-v1';CAP=2*2**30;FUTURE_SOURCE_NOTES_RESERVE=4*2**20

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p
 assert sha(p)==r['sha256'] and ('bytes' not in r or p.stat().st_size==r['bytes']);return p

def main():
 start=time.perf_counter();deadline=start+1800;os.sched_setaffinity(0,{30});pa.set_cpu_count(1);pa.set_io_thread_count(1);t=json.loads(T.read_text());assert t['status']=='serialized_pending_independent_full_source_task_proof';a=json.loads(check(t['source_metadata_admission_audit']).read_text());h=json.loads(check(t['source_handoff']).read_text());href=ref(H);aref=ref(A);assert t['source_handoff']==href and t['source_metadata_admission_audit']==aref;O.mkdir();(O/'code').mkdir();(O/'code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 def guard():
  usage=sum(p.stat().st_size for p in B.rglob('*') if p.is_file())+h['derived_bytes_observed']+FUTURE_SOURCE_NOTES_RESERVE
  assert usage<CAP and shutil.disk_usage(R).free>200*2**30
  if time.perf_counter()>deadline:raise RuntimeError('finite proof deadline reached; preserve partials')
  return usage
 pilot=json.loads((B/'pilot64-native-v1/manifest.json').read_text());pilot_ids={}
 for sh in pilot['shards']:
  for row in pq.read_table(check(sh),columns=['task_id','native_payload_sha256']).to_pylist():assert row['task_id'] not in pilot_ids;pilot_ids[row['task_id']]=row['native_payload_sha256']
 assert len(pilot_ids)==128
 metadata=pq.read_table(check(a['candidate_rows'])).to_pylist();wanted=collections.defaultdict(dict)
 for row in metadata:
  if row['chemical_dose_time_cell_task_candidate']:wanted[row['source_asset_id']][row['source_native_row']]=row
 counts=collections.Counter();units=collections.Counter();records=collections.Counter();ids=set();groups_seen=set();mutations=0;taskbytes=groupbytes=0;pilot_matches=0;allsource_rows=0;graphs=set()
 def stream():
  nonlocal taskbytes,groupbytes
  for sh in t['shards']:
   p=check(sh);gp=check(t['effective_group_lookup'][sh['path']]);rows=pq.read_table(p).to_pylist();gr=pq.read_table(gp).to_pylist();assert len(rows)==len(gr)==sh['rows'];taskbytes+=p.stat().st_size;groupbytes+=gp.stat().st_size
   for i,(row,group) in enumerate(zip(rows,gr)):
    assert row['task_id']==group['task_id'];yield row,group
   guard()
 iterator=iter(stream())
 for comp in h['components']:
  component=json.loads(check(comp).read_text());asset=component['asset_id'];mod='CellPainting' if asset.endswith('-CP') else 'L1000';family='Rosetta-LINCS-Pilot1-molecule-profile-'+mod;source=check(component['original_payload']);sref=ref(source);axispath=Path(component['full_axis_file']);assert sha(axispath)==component['full_axis_sha256'];axis=json.loads(axispath.read_text());names=axis['ordered_feature_names'];schema={f['name']:f['type'] for f in axis['native_schema']};types=[schema[n] for n in names];pf=pq.ParquetFile(source);assert all(str(pf.schema_arrow.field(n).type)==tp for n,tp in zip(names,types));ordinal=0
  for batch in pf.iter_batches(batch_size=128):
   positions=[i for i in range(batch.num_rows) if ordinal+i in wanted[asset]]
   if positions:
    for local,raw in zip(positions,batch.take(pa.array(positions,type=pa.int64())).to_pylist()):
     index=ordinal+local;meta=wanted[asset][index];md=json.loads(meta['source_native_metadata_json']);assert all(raw[k]==md[k] for k in axis['metadata_columns']);values=[raw[n] for n in names];mask=[v is not None for v in values];assert sum(not k for k in mask)==meta['source_feature_nulls'];array=dict(axis_ids=names,native_types=types,values=values,observed_mask=mask,feature_units=axis['feature_units']);records[mod]+=1;graphs.add(meta['unique_exact_source_reference_graph_sha']);units[mod+':unique_source_profile_feature_slots']+=len(values);units[mod+':unique_observed_source_values']+=sum(mask);units[mod+':source_missing_feature_slots']+=len(values)-sum(mask);units[mod+':native_int64_slots']+=sum(k=='int64' for k in types)
     expected_groups=set(json.loads(meta['chemical_groups_json']));expected_groups.update(['source_study:LINCS-Pilot1','source_parent:'+asset+':'+component['original_payload']['sha256'],'source_perturbation:LINCS-Pilot1:'+hashlib.sha256(canonical(md['Metadata_pert_id'])).hexdigest(),'source_plate:'+hashlib.sha256(canonical([asset,md['Metadata_Plate']])).hexdigest(),'source_well:'+hashlib.sha256(canonical([asset,md['Metadata_Plate'],md['Metadata_Well']])).hexdigest()]);assert not json.loads(meta['existing_heldout_chemical_groups_json']) and not json.loads(meta['frozen_pilot_validation_groups_json']);groups_seen.update(expected_groups)
     for direction in ['molecule_to_profile','profile_to_molecule']:
      row,group=next(iterator);rid=meta['source_occurrence_key'];tid=hashlib.sha256(canonical([family,rid,direction])).hexdigest();assert row['task_id']==tid and tid not in ids;ids.add(tid);assert row['record_id']==rid and row['source_family']==family and row['task_name']==direction and row['split']=='train' and row['admitted_for_task'] and row['model_adapter_supported'] is False and row['source_sha256']==href['sha256'] and row['native_schema']=='mora3.CorpusTask/v1';assert set(json.loads(row['group_ids_json']))==set(json.loads(group['effective_groups_json']))==expected_groups;payload=row['native_payload'];assert sha_payload(payload)==row['native_payload_sha256'];task=CorpusTask.from_bytes(payload);assert task.canonical_bytes()==payload;d=json.loads(payload);assert d['task']=='source_observational_LINCS_'+direction;sa=d['audit']['source_audit'];assert sa['source_native_record']==dict(**sref,row_index=index) and sa['source_original_metadata']==md and sa['source_handoff']==href and sa['admission_audit']==aref;assert d['audit']['same_cell_crossassay_pair'] is False and d['audit']['causal_or_independent_biological_validation'] is False and d['audit']['generated_data'] is False
      common=dict(assay_modality=mod,cell_line='A549',source_nominal_dose=dict(value=md['Metadata_pert_dose_micromolar'],unit='micromolar',evidence='author curated source field; no new conversion or concentration calibration'),source_exposure_time=dict(value=md['Metadata_pert_timepoint'],unit='hours',evidence='author curation assignment retained in every native row'));assert common['source_nominal_dose']['value']>0 and common['source_exposure_time']['value']==(48 if mod=='CellPainting' else 24)
      if direction=='molecule_to_profile':expected_inputs=dict(common,canonical_isomeric_smiles=meta['canonical_isomeric_source_reference_graph'],known_profile_axis=names,known_native_feature_types=types);expected_targets=dict(source_profile_native_array=array);restored=d['targets']['source_profile_native_array']
      else:expected_inputs=dict(common,source_profile_native_array=array);expected_targets=dict(canonical_isomeric_smiles=meta['canonical_isomeric_source_reference_graph']);restored=d['conditions']['source_profile_native_array']
      assert d['conditions']==expected_inputs and d['targets']==expected_targets and row['full_axis_values']==len(values);assert restored['axis_ids']==names and restored['native_types']==types and restored['observed_mask']==mask
      for original,actual,tp in zip(values,restored['values'],types):
       if original is None:assert actual is None
       else:assert type(actual) is (int if tp=='int64' else float) and struct.pack('<q' if tp=='int64' else '<d',original)==struct.pack('<q' if tp=='int64' else '<d',actual)
      assert task.input_bytes()==CorpusTask(d['task'],expected_inputs,{'hidden_target_mutated':'X'},{'names_metadata_IDs_QC_SMILES':'changed'},['changed']).input_bytes();mutations+=1
      if tid in pilot_ids:assert pilot_ids[tid]==row['native_payload_sha256'];pilot_matches+=1
      counts[family+':'+direction]+=1;units[mod+':'+direction+':profile_feature_slots']+=len(values);units[mod+':'+direction+':observed_profile_values']+=sum(mask)
   ordinal+=batch.num_rows;guard()
  assert ordinal==component['original_payload']['rows'];allsource_rows+=ordinal;assert ref(source)['sha256']==component['original_payload']['sha256'];print('LINCS_INDEPENDENT_PROOF',mod,records[mod],len(ids),'seconds',round(time.perf_counter()-start,2),flush=True)
 try:next(iterator);raise AssertionError('extra unbound task row')
 except StopIteration:pass
 assert len(ids)==t['serialized_tasks']==88286 and sum(records.values())==44143 and dict(records)==t['distinct_source_profiles'] and dict(counts)==t['task_counts'] and dict(units)==t['source_native_units_not_model_tokens'] and taskbytes==t['task_shard_bytes'] and pilot_matches==128 and allsource_rows==80060
 receipt=dict(schema='mora3.LINCSFullNativeTaskIndependentProof/v1',status='verified',component=ref(T),source_handoff=href,metadata_admission_audit=aref,all_native_source_rows_iterated=allsource_rows,original_source_profiles_serialized=dict(records),distinct_strict_source_reference_graphs=len(graphs),serialized_task_views=len(ids),task_counts=dict(counts),all_new_task_IDs_unique=True,all_native_float64_int64_bits_and_nulls_exact=True,all_full_1783_978_axes_and_feature_types_preserved=True,all_source_conditions_targets_context_and_masks_independently_replayed=True,source_parent_study_plate_well_perturbation_chemical_groups_replayed=True,new_train_existing_heldout_or_frozen_group_intersections=0,target_and_audit_mutation_input_invariance_checks=mutations,preserved_unadmitted_pilot_payloads_equal_full_component=pilot_matches,task_shard_bytes=taskbytes,group_sidecar_bytes=groupbytes,source_native_units_not_model_tokens=dict(units),actual_model_input_target_tokens=None,actual_optimizer_exposure=None,no_new_independent_experiments=True,no_new_heldout=True,no_same_cell_assay_or_matched_replicate_pair=True,no_registered_image_mask_scale_or_spatial_entities=True,model_adapter_supported=False,shared_derived_usage_including_4MiB_source_notes_reserve=guard(),shared_cap_bytes=CAP,seconds=time.perf_counter()-start);write(O/'receipt.json',receipt);guard();print('COMPLETE_LINCS_INDEPENDENT_COMPONENT_PROOF',json.dumps({k:receipt[k] for k in ['original_source_profiles_serialized','serialized_task_views','distinct_strict_source_reference_graphs','task_shard_bytes','group_sidecar_bytes','seconds']}),flush=True)

def sha_payload(payload):return hashlib.sha256(payload).hexdigest()
def write(p,v):p.open('x').write(json.dumps(v,indent=2)+'\n')
if __name__=='__main__':main()
