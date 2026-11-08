"""Finite reviewed ExpansionRx native readout component; no master publication."""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,os,resource,sqlite3,sys,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import canonical
from data_intake_openadmet_admission import admitted_observation,source_task_views
B=R/'outputs/data-intake/20261101-ADMET-corpus-v1'
P=B/'ExpansionRx-component-execution-plan-v1.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while chunk:=f.read(2**20):h.update(chunk)
 return h.hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def checked(v):
 p=Path(v['path']);p=p if p.is_absolute() else R/p
 if p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:raise ValueError('pinned input changed: '+str(p))
 return p
def approved(approval,plan_ref,script_ref):
 if approval.get('approved') is not True or approval.get('scope')!='ExpansionRx-complete-eligible-native-observation-component-only':raise ValueError('exact component scope approval required')
 if approval.get('plan_sha256')!=plan_ref['sha256'] or approval.get('script_sha256')!=script_ref['sha256']:raise ValueError('exact plan and script binding required')
 if approval.get('master_publication') is not False or approval.get('model_training') is not False:raise ValueError('master publication and training remain separate')

def independent_envelope_check(row,record,direction,groups):
 payload=row['native_payload'];raw=json.loads(payload)
 if canonical(raw)!=payload or hashlib.sha256(payload).hexdigest()!=row['native_payload_sha256']:raise ValueError('native canonical-byte mismatch')
 if raw['audit']['source_observation']!=record:raise ValueError('complete source observation/proof fields were lost')
 native=record['cell'];literal={k:native[k] for k in ['native_literal','status','decimal_literal','relation']}
 if direction=='entity_to_measurement':
  if raw['targets']!={'native_measurement':literal} or raw['conditions']['native_entity']!=record['entity']:raise ValueError('forward roles differ from exact native source')
  forbidden=['desired_native_measurement']
 else:
  if raw['targets']!={'native_entity':record['entity']} or raw['conditions']['desired_native_measurement']!=literal:raise ValueError('inverse roles differ from exact native source')
  forbidden=['native_entity']
 if any(k in raw['conditions'] for k in forbidden):raise ValueError('hidden target entered predictor')
 if 'assay_protocol_context' in raw['conditions']['endpoint']:raise ValueError('source proof refs are audit only')
 if raw['conditions']['endpoint']!=record['endpoint'] or raw['conditions']['assay_context']!=record['known_context']:raise ValueError('schema/assay native context changed')
 if json.loads(row['group_ids_json'])!=sorted(groups) or row['split']!=record['admission']['allocation']:raise ValueError('reviewed allocation/groups changed')
 if record['source_native_axis_cell_multiplicity']!=1 or record['independent_experimental_measurement_count'] is not None:raise ValueError('axis cells are not extra independent experiments')
 return True

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--approval',required=True);args=parser.parse_args();start=time.monotonic();os.sched_setaffinity(0,{26,27});os.nice(10);pa.set_cpu_count(1);pa.set_io_thread_count(1)
 plan=json.loads(P.read_text());ap=R/args.approval;approval=json.loads(ap.read_text());approved(approval,ref(P),ref(Path(__file__).resolve()))
 for v in plan['inputs'].values():checked(v)
 for v in plan['dependencies']:checked(v)
 O=R/plan['output_namespace'];assert not O.exists();O.mkdir();(O/'code').mkdir()
 for path in [Path(__file__).resolve()]+[R/x['path'] for x in plan['dependencies']]:(O/'code'/path.name).open('xb').write(path.read_bytes())
 def guard():
  st=os.statvfs(R);rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;other=Path('/proc/3501176/status');other_rss=0
  if other.exists():
   for line in other.read_text().splitlines():
    if line.startswith('VmRSS:'):other_rss=int(line.split()[1])*1024
  if time.monotonic()-start>900 or rss>512*2**20 or rss+other_rss+128*2**20>8*2**30 or st.f_bavail*st.f_frsize<64*2**30:raise RuntimeError('finite constructor shared hardware guard; preserve partial outputs')
 guard();projection=json.loads(checked(plan['inputs']['native_observation_projection']).read_text());gm=json.loads(checked(plan['inputs']['chemical_group_review']).read_text());jm=json.loads(checked(plan['inputs']['joint_endpoint_review']).read_text());registry=json.loads(checked(jm['endpoint_registry']).read_text());reviews={(r['source_family'],r['source_endpoint']):r for r in registry['endpoint_reviews']};graphs=pq.read_table(checked(gm['source_identity_group_review'])).to_pylist();jointrows=pq.read_table(checked(jm['joint_known_source_guard_rows'])).to_pylist();bygraph={(r['source_family'],r['native_entity_id']):r for r in graphs};byjoint={(r['source_family'],r['native_entity_id']):r for r in jointrows};assert len(bygraph)==len(byjoint)==8178
 db=sqlite3.connect(O/'task-and-source-cell-index.sqlite');db.executescript('CREATE TABLE tasks(task_id TEXT PRIMARY KEY,observation_id TEXT NOT NULL,direction TEXT NOT NULL,split TEXT NOT NULL,shard_path TEXT NOT NULL,shard_row INTEGER NOT NULL); CREATE TABLE source_cells(observation_id TEXT PRIMARY KEY,source_family TEXT NOT NULL,source_entity TEXT NOT NULL,endpoint TEXT NOT NULL,source_parent TEXT NOT NULL,status TEXT NOT NULL,disposition TEXT NOT NULL,serialized_views INTEGER NOT NULL);')
 counts=collections.Counter();dispositions=collections.Counter();endpoint_counts=collections.defaultdict(collections.Counter);entity_counts=collections.defaultdict(set);shards=[];lookup={};buffer=[];source_schema='mora3.CorpusTask/v1';direction_counts=collections.Counter()
 def flush():
  if not buffer:return
  guard();split=buffer[0]['split'];assert all(x['split']==split for x in buffer);number=len(shards);sp=O/f'shards/{split}/{number:05d}.parquet';gp=O/f'effective-groups/{number:05d}.parquet';sp.parent.mkdir(parents=True,exist_ok=True);gp.parent.mkdir(exist_ok=True)
  pq.write_table(pa.Table.from_pylist(buffer),sp,compression='zstd');assert pq.read_table(sp).to_pylist()==buffer;groups=[dict(task_id=r['task_id'],effective_groups_json=r['group_ids_json']) for r in buffer];pq.write_table(pa.Table.from_pylist(groups),gp,compression='zstd');assert pq.read_table(gp).to_pylist()==groups
  for i,row in enumerate(buffer):db.execute('INSERT INTO tasks VALUES(?,?,?,?,?,?)',(row['task_id'],row['record_id'],row['direction'],split,str(sp.relative_to(R)),i))
  r=ref(sp,rows=len(buffer),split=split,source_family='ExpansionRx-source-native-ADMET-readouts');shards.append(r);lookup[r['path']]=ref(gp,rows=len(buffer));buffer.clear();db.commit()
 for shard in projection['shards']:
  guard();path=checked(shard)
  for source_row in pq.read_table(path).to_pylist():
   record=json.loads(source_row['native_observation']);raw=source_row['native_observation'];assert hashlib.sha256(raw).hexdigest()==source_row['native_observation_sha256'];family=source_row['source_family'];entity=source_row['source_native_entity_id'];key=(family,entity);counts['source_native_axis_slots_seen']+=1;status=record['cell']['status'];counts['source_status:'+status]+=1
   if family!='ExpansionRx':qualified=None;reason='ASAP_source_units_and_tabular_transform_unresolved'
   else:qualified,reason=admitted_observation(record,bygraph[key],byjoint[key],reviews[(family,source_row['source_endpoint'])])
   dispositions[family+':'+reason]+=1;views=0
   if qualified is not None:
    allocation=qualified['admission']['allocation'];entity_counts[allocation].add(entity);counts['qualified_source_axis_cells']+=1;counts['qualified_source_axis_cells:'+allocation]+=1;endpoint_counts[family+':'+source_row['source_endpoint']][allocation+':'+status]+=1;groups=set(json.loads(bygraph[key]['coarse_graph_groups_json']))|{'source_compound:ExpansionRx:'+entity,'source_native_axis_cell:'+source_row['observation_id']}
    for direction,task in source_task_views(qualified).items():
     if buffer and buffer[0]['split']!=allocation:flush()
     payload=task.canonical_bytes();tid=hashlib.sha256(canonical(['ExpansionRx-source-native-ADMET-v1',source_row['observation_id'],direction])).hexdigest();row=dict(task_id=tid,record_id=source_row['observation_id'],source_family='ExpansionRx-source-native-ADMET-readouts',source_revision='openadmet-expansionrx-challenge-data@6b898ccc43d10d25b230fb09e22a6e30c30022b5',source_sha256=record['source_ref']['source']['sha256'],task_name='source_native_ADMET_'+direction,native_schema=source_schema,native_payload=payload,native_payload_sha256=hashlib.sha256(payload).hexdigest(),split=allocation,split_policy=reason,group_ids_json=json.dumps(sorted(groups)),evidence_class='source_curated_assay_readout;notindependentexperimentalreplicateorclinicalvalidation',admitted_for_task=True,model_adapter_supported=False,direction=direction,full_axis_values=1,admission_receipt_json=json.dumps(dict(component_task_scoped=True,global_master_admission=False,source_rights=qualified['admission']['source_rights'],native_schema_units_or_coordinate_review=jm['endpoint_registry'],known_group_guards=plan['inputs']['chemical_group_review'],global_novelty_or_study_disjointness_proven=False,original_source_parent_protocol=record['source_split'],independent_experimental_measurement_count=None)),missing_context_json=json.dumps(json.loads(payload)['missing']),condition_context_json=json.dumps(dict(physical_unit_known=qualified['endpoint']['unit_basis']=='official_source_schema',native_coordinate_only=reviews[(family,source_row['source_endpoint'])]['native_coordinate_only'],no_unit_conversion=True,source_missing_is_unknown_not_negative=True,original_official_test_not_train=True,model_training=False)))
     independent_envelope_check(row,qualified,direction,groups);buffer.append(row);views+=1;counts['serialized_task_views']+=1;counts['serialized_task_views:'+allocation]+=1;direction_counts[direction]+=1
     if len(buffer)>=512:flush()
   db.execute('INSERT INTO source_cells VALUES(?,?,?,?,?,?,?,?)',(source_row['observation_id'],family,entity,source_row['source_endpoint'],record['source_split']['parent_protocol'],status,reason,views))
  db.commit();print('EXPANSIONRX_COMPONENT_SOURCE_SHARD',path.name,'source_cells',counts['source_native_axis_slots_seen'],'views',counts['serialized_task_views'],flush=True)
 flush();assert counts['source_native_axis_slots_seen']==78980;assert counts['serialized_task_views']==2*counts['qualified_source_axis_cells'];assert db.execute('SELECT count(*) FROM tasks').fetchone()[0]==counts['serialized_task_views'];assert db.execute('SELECT count(*) FROM source_cells').fetchone()[0]==78980
 # Exact new train/test group check, independent of source-native row order.
 train=set();test=set()
 for sh in shards:
  gs=pq.read_table(checked(lookup[sh['path']]))['effective_groups_json'].to_pylist()
  for g in gs:(train if sh['split']=='train' else test).update(json.loads(g))
 assert not train&test;db.close();guard()
 for v in plan['inputs'].values():checked(v)
 for v in plan['dependencies']:checked(v)
 receipt=dict(schema='mora3.ExpansionRxNativeADMETComponent/v1',status='complete_task_scoped_component_pending_independent_master_union_and_native_model_handoff',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=ref(P),approval=ref(ap),inputs=plan['inputs'],source_native_axis_cell_counts=dict(counts),exclusion_dispositions=dict(dispositions),per_endpoint_allocation_status={k:dict(v) for k,v in endpoint_counts.items()},distinct_source_entity_counts_with_admitted_numeric_cells={k:len(v) for k,v in entity_counts.items()},serialized_directions=dict(direction_counts),serialized_tasks=counts['serialized_task_views'],shards=shards,effective_group_lookup=lookup,total_task_shard_bytes=sum(s['bytes'] for s in shards),source_cell_and_task_index=ref(O/'task-and-source-cell-index.sqlite'),new_train_test_known_group_intersections=0,all_78980_source_axis_cells_retained_in_source_index=True,unknown_native_fields_preserved_in_complete_observation_audit=True,full_native_exact_or_censored_literals_preserved=True,inferred_unit_conversions=[],schema_defined_units_distinct_from_native_coordinate_and_unresolved_units=True,source_test_exposure_exclusions_and_train_blocked_groups_preserved=True,raw_ML_views_not_additive_source_observations=True,directions_are_not_independent_assays=True,independent_experimental_measurement_count=None,global_master_admitted_tasks=0,model_adapter_verified=False,actual_model_optimizer_use=None,model_tokens=None,final_mixed_weights=None,full_dataset_ready=False,substantive_training_authorized=False,source_originals_indexes_and_RNA_writer_unchanged=True,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 (O/'manifest.json').open('x').write(json.dumps(receipt,indent=2)+'\n');print('COMPLETE_EXPANSIONRX_NATIVE_COMPONENT',json.dumps(dict(counts=dict(counts),entity_counts=receipt['distinct_source_entity_counts_with_admitted_numeric_cells'],seconds=receipt['seconds'])),flush=True)
if __name__=='__main__':main()
