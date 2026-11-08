from pathlib import Path
import json,hashlib,datetime,os
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');os.chdir(R)
O=R/'planning/research/data_intake/20261101-expanded-scope-v4/ExpansionRx-post-RNA-exact-union-preparation-v1/closed-eight-row-fixture-v1';O.mkdir(exist_ok=False)
manifest=R/'outputs/data-intake/20261101-ADMET-corpus-v1/ExpansionRx-closed-native-reader-fixture-v1/manifest.json';d=json.loads(manifest.read_text())
def ref(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
tables=[];groups=[];provenance=[];selected=[]
for status in ['exact','censored_bound']:
 case=next(c for c in d['cases'] if c['source_disposition']['status']==status and c['source_disposition']['serialized_views']==2 and c['source_disposition']['source_parent']=='native_train')
 assert len(case['tasks'])==2;selected.append(dict(role=case['role'],source_disposition=case['source_disposition']))
 for task in case['tasks']:
  pin=task['task_fixture'];p=R/pin['path'];actual=ref(p);assert actual=={k:pin[k] for k in ['path','bytes','sha256']}
  table=pq.read_table(p);assert table.num_rows==1;row=table.to_pylist()[0];assert row['native_payload_sha256']==hashlib.sha256(row['native_payload']).hexdigest()
  group=task['selected_group_row'];assert group['task_id']==row['task_id'] and json.loads(group['effective_groups_json'])==json.loads(row['group_ids_json'])
  tables.append(table);groups.append(group);provenance.append(dict(closed_task_fixture=actual,original_source_task_shard=task['source_task_shard'],original_source_task_row=task['source_task_shard_row'],original_source_group=task['source_effective_group'],group_row_authority='exact selected_group_row from pinned closed fixture manifest; not a new full group scan'))
assert len(tables)==len(groups)==4;table=pa.concat_tables(tables);p=O/'four-native-task-copies.parquet';pq.write_table(table,p,compression='zstd');assert pq.read_table(p).to_pylist()==table.to_pylist()
g=O/'four-native-effective-group-row-copies.json';g.open('x').write(json.dumps(groups,indent=2)+'\n')
x=dict(schema='mora3.ExpansionRxExactUnionBoundedFixture/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_closed_fixture=ref(manifest),native_task_copies=ref(p),native_effective_group_copies=ref(g),actual_task_rows_decoded=4,exact_group_row_copies_from_pinned_metadata=4,total_existing_record_copies=8,source_axis_cells=2,directions_per_cell=2,selected_source_cases=selected,source_refs=provenance,native_payload_full_bytes_unknown_fields_censor_units_and_source_groups_unchanged=True,new_corpus_or_admitted_views=0,full_component_or_ID_scan=False,RNA_candidate_membership_queries=0,model_use=False,training=False)
f=O/'manifest.json';f.open('x').write(json.dumps(x,indent=2)+'\n');print(json.dumps(ref(f)))
