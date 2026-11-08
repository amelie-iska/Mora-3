"""Tiny closed ExpansionRx task/disposition fixture; zero extra corpus views."""
from pathlib import Path
import collections,datetime,functools,hashlib,json,os,resource,sqlite3,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261101-ADMET-corpus-v1';C=B/'ExpansionRx-native-readout-component-v1';O=B/'ExpansionRx-closed-native-reader-fixture-v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while x:=f.read(2**20):h.update(x)
 return h.hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(v):
 p=Path(v['path']);p=p if p.is_absolute() else R/p
 assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'];return p
def main():
 start=time.monotonic();os.sched_setaffinity(0,{26,27});os.nice(10);pa.set_cpu_count(1);assert not O.exists();O.mkdir();mp=C/'manifest.json';m=json.loads(mp.read_text());assert m['status']=='complete_task_scoped_component_pending_independent_master_union_and_native_model_handoff';assert not Path('/proc/3613076').exists();projection=json.loads(check(m['inputs']['native_observation_projection']).read_text());gmanifest=json.loads(check(m['inputs']['chemical_group_review']).read_text());graphs={(r['source_family'],r['native_entity_id']):r for r in pq.read_table(check(gmanifest['source_identity_group_review'])).to_pylist()};index=check(m['source_cell_and_task_index']);before=ref(index);db=sqlite3.connect('file:'+str(index)+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
 specs=[('actual_exact_train_both_directions',"source_family='ExpansionRx' AND source_parent='native_train' AND status='exact' AND serialized_views=2"),('actual_censored_train_both_directions',"source_family='ExpansionRx' AND source_parent='native_train' AND status='censored_bound' AND serialized_views=2"),('actual_original_source_test_both_directions',"source_family='ExpansionRx' AND source_parent='native_test' AND status='exact' AND serialized_views=2"),('LogD_native_coordinate_only',"source_family='ExpansionRx' AND source_parent='native_train' AND endpoint='LogD' AND status='exact' AND serialized_views=2"),('efflux_native_coordinate_only',"source_family='ExpansionRx' AND source_parent='native_train' AND endpoint='Caco-2 Permeability Efflux' AND status='exact' AND serialized_views=2"),('RLM_raw_extra_endpoint_on_allocated_parent',"source_family='ExpansionRx' AND source_parent='native_train' AND endpoint='RLM CLint' AND status='exact' AND serialized_views=2"),('missing_native_axis_not_negative',"source_family='ExpansionRx' AND source_parent='native_train' AND status='missing' AND serialized_views=0"),('source_train_excluded_official_heldout_parent',"source_family='ExpansionRx' AND source_parent='native_train' AND status='exact' AND disposition='known_official_source_heldout_parent_graph_group'"),('source_test_known_oldtrain_exposure_excluded',"source_family='ExpansionRx' AND source_parent='native_test' AND status='exact' AND disposition='source_test_known_prior_train_graph_group_exposure'"),('raw_only_UNALLOCATED_entity_not_RLM_endpoint_role',"source_family='ExpansionRx' AND source_parent='raw_only_unallocated' AND status='exact'"),('ASAP_native_readout_transform_gate_closed',"source_family='ASAP' AND status='exact'")]
 selected=[(label,db.execute('SELECT * FROM source_cells WHERE '+where+' LIMIT 1').fetchone()) for label,where in specs];assert all(row for _,row in selected)
 @functools.lru_cache(maxsize=3)
 def table(p):return pq.read_table(p)
 def row_at(path,i):return table(str(path)).slice(i,1).to_pylist()[0]
 def native_observation(cell):
  family=cell['source_family'];g=graphs[(family,cell['source_entity'])];axes=[k.split(':',1)[1] for k in projection['per_endpoint_status'] if k.startswith(family+':')];offset=0 if family=='ExpansionRx' else 76180;ordinal=offset+g['native_row_index']*len(axes)+axes.index(cell['endpoint'])
  for sh in projection['shards']:
   if ordinal<sh['rows']:
    p=check(sh);r=row_at(p,ordinal);assert r['observation_id']==cell['observation_id'];return r,sh,ordinal
   ordinal-=sh['rows']
  raise ValueError('source observation ordinal outside pinned projection')
 sources=[];cases=[];unique_tasks={};group_lookup=m['effective_group_lookup'];taskmap={x['path']:x for x in m['shards']}
 for label,cell0 in selected:
  assert time.monotonic()-start<180 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<512*2**20
  cell=dict(cell0);source,sourceref,local=native_observation(cell);native=json.loads(source['native_observation']);assert hashlib.sha256(source['native_observation']).hexdigest()==source['native_observation_sha256'];assert source['source_endpoint']==cell['endpoint'] and native['cell']['status']==cell['status'];sources.append(source);tasks=[]
  for task in db.execute('SELECT * FROM tasks WHERE observation_id=? ORDER BY direction',(cell['observation_id'],)):
   task=dict(task);sp=check(taskmap[task['shard_path']]);gp=check(group_lookup[task['shard_path']]);row=row_at(sp,task['shard_row']);group=row_at(gp,task['shard_row']);assert row['task_id']==task['task_id']==group['task_id'];assert row['record_id']==cell['observation_id'];assert hashlib.sha256(row['native_payload']).hexdigest()==row['native_payload_sha256'];payload=json.loads(row['native_payload']);assert payload['audit']['source_observation']['cell']==native['cell'];key=row['task_id']
   if key not in unique_tasks:
    p=O/f'task-{len(unique_tasks):02d}.parquet';pq.write_table(pa.Table.from_pylist([row]),p,compression='zstd');assert pq.read_table(p).to_pylist()==[row];unique_tasks[key]=ref(p,rows=1)
   tasks.append(dict(task_fixture=unique_tasks[key],direction=task['direction'],source_task_shard=taskmap[task['shard_path']],source_task_shard_row=task['shard_row'],source_effective_group=group_lookup[task['shard_path']],selected_group_row=group,component_task_scoped=True,global_master_admission=False,model_use=False))
  assert len(tasks)==cell['serialized_views'];cases.append(dict(role=label,source_disposition=cell,native_source_observation_fixture_row=len(sources)-1,native_source_projection_shard=sourceref,native_source_projection_row=local,tasks=tasks))
 db.close();assert ref(index)==before;p=O/'native-source-observation-records.parquet';pq.write_table(pa.Table.from_pylist(sources),p,compression='zstd');assert pq.read_table(p).to_pylist()==sources
 receipt=dict(schema='mora3.ClosedExpansionRxNativeTaskReaderFixture/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),component=ref(mp),plan=m['plan'],approval=m['approval'],source_cell_task_index=before,native_projection=m['inputs']['native_observation_projection'],source_known_group_review=m['inputs']['chemical_group_review'],endpoint_and_joint_source_review=m['inputs']['joint_endpoint_review'],native_source_observation_records=ref(p,rows=len(sources)),cases=cases,unique_copied_task_views=len(unique_tasks),fixture_task_copies_not_new_corpus_views=True,all_fixture_tasks_native_full_unknown_fields_and_censor_literals_preserved=True,all_source_unit_schema_refs_distinct_from_native_record_literals=True,current_master='v22',component_master_union_complete=False,all_model_supervision_and_optimizer_use_authorized=False,substantive_training_authorized=False,full_dataset_ready=False,originals_and_indexes_unchanged=True,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 p=O/'manifest.json';p.open('x').write(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(manifest=ref(p),roles=[c['role'] for c in cases],native_records=len(sources),task_copies=len(unique_tasks),seconds=receipt['seconds'])))
if __name__=='__main__':main()
