"""Complete lossless native ADMET observation projection; no task admission/split repair."""
from pathlib import Path
import collections,csv,datetime,hashlib,json,os,sys,time
import pyarrow as pa
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_admet_observations import native_cell,observation
from data_intake_corpus import canonical
H=R/'outputs/data-sources/20261101-ADMET-cross-modality-v1/handoff-OpenADMET-full-native-v1.json'
O=R/'outputs/data-intake/20261101-ADMET-corpus-v1/native-observations-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def checked(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p
 if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('source hash/size mismatch')
 return p
def write(p,v):p.open('x').write(json.dumps(v,indent=2)+'\n')
def main():
 start=time.monotonic();os.sched_setaffinity(0,{29});pa.set_cpu_count(1);pa.set_io_thread_count(1)
 assert not O.exists();O.mkdir(parents=True);(O/'code').mkdir();(O/'observations').mkdir()
 for s in [Path(__file__).name,'data_intake_admet_observations.py']:(O/'code'/s).open('xb').write((R/'scripts'/s).read_bytes())
 assert sha(H)=='b54afa8be4c25eaec66fb249ae96cfb2704cc64b3e4849217c46b0e7fc0a0eb5'
 h=json.loads(H.read_text());m=json.loads(checked(h['manifest']).read_text());files=m['source_files'];published={};manifestrefs=[]
 for f in files:
  if f['source']['path'].endswith(('expansion_data_train.csv','expansion_data_test.csv')):
   rows=list(csv.DictReader(checked(f['source']).open(newline='')))
   for row in rows:
    identity=row['Molecule Name'];assert identity not in published
    published[identity]=('native_train' if 'train.csv' in f['source']['path'] else 'native_test',row['SMILES'],ref(checked(f['source'])))
 assert len(published)==7608
 counts=collections.Counter();axes=collections.defaultdict(collections.Counter);shards=[];buffer=[];native_entities=collections.Counter();sourcefiles=[];numeric_missing_differences=collections.Counter()
 def guard():
  if time.monotonic()-start>600:raise RuntimeError('finite projection deadline; retain outputs')
  st=os.statvfs(R)
  if st.f_bavail*st.f_frsize<64*2**30:raise RuntimeError('hardware free buffer; retain outputs')
 def flush():
  if not buffer:return
  p=O/'observations'/f'{len(shards):04d}.parquet';pq.write_table(pa.Table.from_pylist(buffer),p,compression='zstd')
  back=pq.read_table(p).to_pylist();assert back==buffer
  for row in back:
   d=json.loads(row['native_observation']);assert canonical(d)==row['native_observation']
   assert d['cell']['native_literal']==d['native_source_record'][d['endpoint']['source_endpoint']]
   assert d['cell']['observation_present']==row['observation_present']
   assert d['admission']['split_group_review'] is False and row['task_admitted'] is False
  shards.append(ref(p,rows=len(back)));buffer.clear();guard()
 for f in files:
  if not f['source']['path'].endswith(('expansion_data_raw.csv','ASAP/ADMET.csv')):continue
  source=checked(f['source']);typed=checked(f['payload']);literal_rows=list(csv.DictReader(source.open(newline='')));typed_rows=pq.read_table(typed).to_pylist();assert len(literal_rows)==len(typed_rows)==f['rows']
  family='ExpansionRx' if source.name=='expansion_data_raw.csv' else 'ASAP';sourcefiles.append(ref(source));native_entities[family]+=len(literal_rows)
  for i,(literal,native) in enumerate(zip(literal_rows,typed_rows)):
   assert literal==json.loads(native['native_row_json']) and native['native_row_index']==i
   native_id=native['native_entity_id'];representation=literal['SMILES' if family=='ExpansionRx' else 'CXSMILES']
   split=dict(native_export=native['native_published_split'],assignment_basis='source record; not final corpus allocation')
   if family=='ExpansionRx':
    p=published.get(native_id)
    if p is None:split.update(parent_protocol='raw_only_unallocated',original_ML_export=None)
    else:
     assert p[1]==representation;split.update(parent_protocol=p[0],original_ML_export=p[2])
   else:split.update(parent_protocol=literal['Set'])
   original_measurements=json.loads(native['measurements_json'])
   for endpoint in f['measurement_axes']:
    cell=native_cell(literal[endpoint]);expected=original_measurements[endpoint]
    assert cell['native_literal']==expected['raw'];assert {'missing':'missing','exact':'finite_numeric','censored_bound':'censored_bound'}[cell['status']]==expected['kind']
    unit=m['source_reported_units'][family][endpoint] if family=='ExpansionRx' else None
    ep=dict(endpoint_key=family+':'+endpoint,source_endpoint=endpoint,unit=unit,unit_basis='official_source_schema' if unit is not None else 'unresolved',measurement_type='native source-curated assay readout; not clinical PK or safety validation',assay_protocol_context={'source_schema':h['manifest'],'original_rights':h[family]['rights_evidence']})
    gates=dict(source_rights=True,exact_identity=False,endpoint_definition=False,task_units_or_declared_native_coordinate=False,split_group_review=False)
    obs=observation(entity=dict(molecule_class='small_molecule',representation_type='native_SMILES' if family=='ExpansionRx' else 'native_CXSMILES',representation=representation,native_modifications=None,source_identity_missingness=['full material batch/formulation/replicate context']),endpoint=ep,cell=cell,known_context={},source_ref=dict(source=ref(source),native_payload=f['payload'],native_row_index=i,native_entity_namespace=family,native_entity_id=native_id),native_source_record=literal,evidence_role='source_curated_measurement',source_split=split,admission=gates)
    identity=hashlib.sha256(canonical([sha(source),i,endpoint])).hexdigest();payload=canonical(obs)
    buffer.append(dict(observation_id=identity,source_family=family,source_native_row_index=i,source_endpoint=endpoint,source_native_entity_id=native_id,source_parent_protocol=split['parent_protocol'],native_observation=payload,native_observation_sha256=hashlib.sha256(payload).hexdigest(),observation_present=cell['observation_present'],native_numeric_parse_pass=cell['numeric_loss_eligible'],native_censored=cell['status']=='censored_bound',physical_unit_source_defined=unit is not None,task_admitted=False,model_adapter_supported=False))
    counts['native_axis_slots']+=1;counts[cell['status']]+=1;axes[family+':'+endpoint][cell['status']]+=1
    if len(buffer)>=4096:flush()
 flush();assert counts['native_axis_slots']==78980 and counts['censored_bound']==697;assert sum(s['rows'] for s in shards)==counts['native_axis_slots'];assert native_entities=={'ExpansionRx':7618,'ASAP':560}
 receipt=dict(schema='mora3.ADMETCompleteNativeObservationProjection/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='complete_full_native_source_observation_projection_pending_task_specific_admission',source_handoff=ref(H),source_manifest=h['manifest'],original_sources=sourcefiles,native_source_entity_occurrences=dict(native_entities),native_axis_slots_and_status=dict(counts),per_endpoint_status={k:dict(v) for k,v in axes.items()},counts_are_not_independent_assays_or_additive_ML_views=True,raw_ML_views_not_duplicated=True,original_train_test_parent_assignments_preserved=True,raw_only_unallocated_entities=10,all_source_cells_literals_and_unknown_record_fields_roundtrip=True,all_native_payload_bytes_SHA_and_full_columns_read=True,all_original_CXSMILES_enhanced_stereo_preserved=True,physical_units_not_converted=True,known_context_empty_is_explicit_missing_not_negative=True,task_admitted_observations=0,serialized_training_task_views=0,actual_optimizer_use=None,model_tokens=None,shards=shards,observation_shard_bytes=sum(x['bytes'] for x in shards),open_gates=['exact native chemistry/enhanced stereo source projection and material scope','full source protocol/endpoint/unit interpretation, especially ASAP','whole-corpus existing chemical/scaffold and original test exposure','source-replicate/study/generalization metadata and applicable evaluation','task-specific target masks and native model reader/objective'],hardware_free_buffer_bytes=64*2**30,model_or_live_job_code_modified=False,seconds=time.monotonic()-start)
 write(O/'manifest.json',receipt);guard();print(json.dumps({k:receipt[k] for k in ['status','native_source_entity_occurrences','native_axis_slots_and_status','observation_shard_bytes','task_admitted_observations','seconds']},indent=2))
if __name__=='__main__':main()
