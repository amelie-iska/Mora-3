"""Finite native LINCS task serialization; explicit pilot/full namespace and masks."""
from pathlib import Path
import argparse,collections,copy,datetime,hashlib,json,os,shutil,struct,sys,time
import pyarrow as pa
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import CorpusTask,canonical
from data_intake_lincs_profile_tasks import source_native_profile_task
from data_intake_full_pubchem_guards import task_identifier
B=R/'outputs/data-intake/20261101-LINCS-corpus-v1';A=B/'metadata-admission-audit-v1/manifest.json';S=R/'outputs/data-sources/20261101-LINCS-Pilot1-source-v1';H=S/'handoff-two-originals-full-native-verified-v1.json';CAP=2*2**30

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p
 assert sha(p)==r['sha256'] and ('bytes' not in r or p.stat().st_size==r['bytes']);return p

def write(p,v):p.open('x').write(json.dumps(v,indent=2)+'\n')
def bit_equal(a,b,types):
 assert len(a)==len(b)==len(types)
 for x,y,t in zip(a,b,types):
  if x is None or y is None:assert x is None and y is None
  else:assert struct.pack('<q' if t=='int64' else '<d',x)==struct.pack('<q' if t=='int64' else '<d',y)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['pilot','full'],required=True);args=parser.parse_args();pilot=args.mode=='pilot';start=time.perf_counter();deadline=start+(300 if pilot else 1800);os.sched_setaffinity(0,{30});pa.set_cpu_count(1);pa.set_io_thread_count(1)
 h=json.loads(H.read_text());a=json.loads(A.read_text());href=ref(H);aref=ref(A);assert a['source_handoff']==href;O=B/('pilot64-native-v1' if pilot else 'curated-profile-tasks-v1');O.mkdir();(O/'shards').mkdir();(O/'groups').mkdir();(O/'code').mkdir();(O/'code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes());(O/'code/data_intake_lincs_profile_tasks.py').write_bytes((R/'scripts/data_intake_lincs_profile_tasks.py').read_bytes())
 def guard(reserve=0):
  if time.perf_counter()>deadline:raise RuntimeError('finite phase deadline reached; preserve partials')
  usage=sum(p.stat().st_size for p in B.rglob('*') if p.is_file())+h['derived_bytes_observed']
  if usage+reserve>CAP:raise RuntimeError('shared LINCS 2 GiB derived cap reached; no overflow or cleanup')
  if shutil.disk_usage(R).free<200*2**30:raise RuntimeError('200 GiB reserve violated')
  return usage
 guard(16*2**20);allrows=pq.read_table(check(a['candidate_rows'])).to_pylist();candidates={}
 for mod in ['CellPainting','L1000']:
  rows=[r for r in allrows if r['modality']==mod and r['chemical_dose_time_cell_task_candidate']]
  if pilot:
   complete=[r for r in rows if r['source_finite_complete_profile']][:16];partial=[r for r in rows if not r['source_finite_complete_profile']][:16];selected={r['source_native_row']:r for r in complete+partial}
   for r in rows:
    if len(selected)>=32:break
    selected[r['source_native_row']]=r
   rows=list(selected.values())
  candidates[mod]={r['source_native_row']:r for r in rows}
 shards=[];lookup={};buffer=[];groups_buffer=[];seen=set();taskids=set();records=collections.Counter();counts=collections.Counter();units=collections.Counter();mutations=0;source_refs=[];types_by_mod={};axes_by_mod={};component_refs=[]
 def flush():
  if not buffer:return
  guard(reserve=2*sum(len(r['native_payload']) for r in buffer)+2**20);i=len(shards);p=O/'shards'/f'{i:05d}.parquet';pq.write_table(pa.Table.from_pylist(buffer),p,compression='zstd');back=pq.read_table(p).to_pylist();assert back==buffer
  for row in back:assert hashlib.sha256(row['native_payload']).hexdigest()==row['native_payload_sha256'] and CorpusTask.from_bytes(row['native_payload']).canonical_bytes()==row['native_payload']
  gp=O/'groups'/p.name;pq.write_table(pa.Table.from_pylist(groups_buffer),gp,compression='zstd');assert pq.read_table(gp).to_pylist()==groups_buffer;shards.append(ref(p,rows=len(buffer),source_family=buffer[0]['source_family'],split='train'));lookup[str(p.relative_to(R))]=ref(gp,rows=len(groups_buffer));buffer.clear();groups_buffer.clear();guard()
 for comp in h['components']:
  cp=check(comp);c=json.loads(cp.read_text());component_refs.append(ref(cp));mod='CellPainting' if c['asset_id'].endswith('-CP') else 'L1000';source=check(c['original_payload']);source_ref=ref(source);source_refs.append(c['original_payload']);axispath=Path(c['full_axis_file']);assert sha(axispath)==c['full_axis_sha256'];axis=json.loads(axispath.read_text());names=axis['ordered_feature_names'];native_schema={r['name']:r['type'] for r in axis['native_schema']};types=[native_schema[n] for n in names];assert len(names)==c['full_feature_axis'];types_by_mod[mod]=types;axes_by_mod[mod]=dict(axis_file=ref(axispath),features=len(names),native_type_counts=dict(collections.Counter(types)));wanted=candidates[mod];last=max(wanted) if wanted else -1;ordinal=0
  for batch in pq.ParquetFile(source).iter_batches(batch_size=128):
   positions=[i for i in range(batch.num_rows) if ordinal+i in wanted]
   if positions:
    selected=batch.take(pa.array(positions,type=pa.int64())).to_pylist()
    for pos,native in zip(positions,selected):
     index=ordinal+pos;meta=wanted[index];assert meta['source_asset_id']==c['asset_id'];original_metadata=json.loads(meta['source_native_metadata_json']);assert all(native[k]==original_metadata[k] for k in axis['metadata_columns']);values=[native[k] for k in names];mask=[v is not None for v in values];assert sum(not x for x in mask)==meta['source_feature_nulls'];array=dict(axis_ids=names,native_types=types,values=values,observed_mask=mask,feature_units=axis['feature_units']);rid=meta['source_occurrence_key'];assert rid not in seen;seen.add(rid);records[mod]+=1;units[mod+':unique_source_profile_feature_slots']+=len(values);units[mod+':unique_observed_source_values']+=sum(mask);units[mod+':source_missing_feature_slots']+=len(values)-sum(mask);units[mod+':native_int64_slots']+=sum(t=='int64' for t in types);groups=set(json.loads(meta['chemical_groups_json']));groups.update(['source_study:LINCS-Pilot1','source_parent:'+c['asset_id']+':'+c['original_payload']['sha256'],'source_perturbation:LINCS-Pilot1:'+hashlib.sha256(canonical(original_metadata['Metadata_pert_id'])).hexdigest(),'source_plate:'+hashlib.sha256(canonical([c['asset_id'],original_metadata['Metadata_Plate']])).hexdigest(),'source_well:'+hashlib.sha256(canonical([c['asset_id'],original_metadata['Metadata_Plate'],original_metadata['Metadata_Well']])).hexdigest()]);audit=dict(source_handoff=href,source_native_record=dict(**source_ref,row_index=index),source_original_metadata=original_metadata,exact_reference_binding_scope=meta['chemical_binding_scope'],source_graph_binding_GE_rows=json.loads(meta['source_graph_binding_GE_rows_json']),admission_audit=aref,target_loss_mask_source='native source feature null bitmap; not predictor condition in forward direction',source_raw_images_masks_unavailable_for_task=True);family='Rosetta-LINCS-Pilot1-molecule-profile-'+mod
     for direction in ['molecule_to_profile','profile_to_molecule']:
      task=source_native_profile_task(array,meta['canonical_isomeric_source_reference_graph'],mod,direction,nominal_dose_micromolar=meta['source_nominal_dose_micromolar'],time_hours=meta['source_time_hours'],cell_line='A549',audit=audit);raw=task.canonical_bytes();decoded=json.loads(raw);restored=decoded['targets']['source_profile_native_array'] if direction=='molecule_to_profile' else decoded['conditions']['source_profile_native_array'];bit_equal(values,restored['values'],types);assert restored['observed_mask']==mask and restored['axis_ids']==names and restored['native_types']==types
      if direction=='molecule_to_profile':
       mutated=copy.deepcopy(array);j=next(i for i,v in enumerate(mask) if v);mutated['values'][j]+=1 if types[j]=='int64' else .125;other=source_native_profile_task(mutated,meta['canonical_isomeric_source_reference_graph'],mod,direction,nominal_dose_micromolar=meta['source_nominal_dose_micromolar'],time_hours=meta['source_time_hours'],cell_line='A549',audit={'names_source_SMILES_QC_ids':'mutated'})
      else:other=source_native_profile_task(array,'CCN' if meta['canonical_isomeric_source_reference_graph']!='CCN' else 'CCO',mod,direction,nominal_dose_micromolar=meta['source_nominal_dose_micromolar'],time_hours=meta['source_time_hours'],cell_line='A549',audit={'names_source_SMILES_QC_ids':'mutated'})
      assert task.input_bytes()==other.input_bytes();mutations+=1;tid=task_identifier(family,rid,direction);assert tid not in taskids;taskids.add(tid);counts[family+':'+direction]+=1;unitprefix=mod+':'+direction;units[unitprefix+':profile_feature_slots']+=len(values);units[unitprefix+':observed_profile_values']+=sum(mask)
      buffer.append(dict(task_id=tid,record_id=rid,source_family=family,source_revision='S3 Jan26_2025 entity '+c['source_release_entity']['etag']+';author v1.0.0',source_sha256=href['sha256'],task_name=direction,native_schema='mora3.CorpusTask/v1',native_payload=raw,native_payload_sha256=hashlib.sha256(raw).hexdigest(),split='train',split_policy='source_train_only;known_old_chemical_and_frozen_pilot_guards_excluded;no_new_heldout',group_ids_json=json.dumps(sorted(groups)),evidence_class='source_image_computed_well_profile' if mod=='CellPainting' else 'source_preprocessed_probe_profile;measured_vs_inferred_perprobe_unresolved',admitted_for_task=not pilot,model_adapter_supported=False,admission_receipt_json=json.dumps(dict(source=href,source_license='CC0-1.0',source_metadata_audit=aref,source_native_null_mask_preserved=True,source_physical_types_preserved=True,source_parent_and_perturbation_groups_present=True,no_same_cell_crossassay_join=True,full_profile_axis=len(values),pilot_not_admitted=pilot)),missing_context_json=json.dumps(decoded['missing']),full_axis_values=len(values),condition_context_json=json.dumps(dict(cell_line='A549',source_nominal_dose_micromolar=meta['source_nominal_dose_micromolar'],source_exposure_time_hours=meta['source_time_hours'],source_nominal_dose_calibrated=False,raw_image_available=False,source_mask_available=False,segmentation_available=False,source_generated=False))))
      groups_buffer.append(dict(task_id=tid,effective_groups_json=json.dumps(sorted(groups))))
      if len(buffer)>=64:flush()
   ordinal+=batch.num_rows;guard()
   if pilot and ordinal>last:break
  flush();assert records[mod]==len(wanted)
  assert ref(source)['sha256']==c['original_payload']['sha256'];print('LINCS_TASKS',mod,records[mod],len(taskids),'seconds',round(time.perf_counter()-start,2),flush=True)
 assert len(seen)==sum(len(v) for v in candidates.values()) and len(taskids)==2*len(seen);usage=guard();manifest=dict(schema='mora3.LINCSSourceNativeProfileTaskComponent/v1',status='pilot_verified_not_admitted' if pilot else 'serialized_pending_independent_full_source_task_proof',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_handoff=ref(H),source_metadata_admission_audit=aref,source_components=component_refs,original_sources=source_refs,distinct_source_profiles=dict(records),task_counts=dict(counts),serialized_tasks=len(taskids),task_shard_bytes=sum(s['bytes'] for s in shards),shards=shards,effective_group_lookup=lookup,source_full_axes=axes_by_mod,source_native_units_not_model_tokens=dict(units),actual_model_input_target_tokens=None,source_native_int64_double_bits_and_nulls_roundtrip=True,target_and_source_audit_mutation_input_invariance=mutations,all_source_metadata_and_feature_labels_preserved=True,new_independent_experiments=0,source_observations_not_additive_to_unconditional_views=True,no_new_heldouts=True,source_assays_not_same_cell_or_matched_replicates=True,profile_features_not_images_masks_or_spatial_cells=True,model_adapter_supported=False,shared_2GiB_derived_usage_before_manifest=usage,cap_bytes=CAP,seconds=time.perf_counter()-start);write(O/'manifest.json',manifest);guard();print('COMPLETE_LINCS_TASK_COMPONENT',json.dumps({k:manifest[k] for k in ['status','distinct_source_profiles','serialized_tasks','task_shard_bytes','seconds']}),flush=True)

if __name__=='__main__':main()
