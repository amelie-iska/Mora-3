"""Independent original-CSV/native-cell/task proof; no master or model changes."""
from pathlib import Path
import argparse,collections,csv,datetime,hashlib,json,os,re,resource,sqlite3,sys,time
from decimal import Decimal
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import canonical
B=R/'outputs/data-intake/20261101-ADMET-corpus-v1';C=B/'ExpansionRx-native-readout-component-v1';P=B/'ExpansionRx-independent-source-task-proof-plan-v1.json'
EXCLUDED={'endpoint','known_context','admission','original_endpoint_projection_audit','endpoint_source_review_audit','graph_scope_review_audit','source_protocol_group_review_audit','joint_source_benchmark_guard_audit','inverse_design_scope_audit'}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while x:=f.read(2**20):h.update(x)
 return h.hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def checked(v):
 p=Path(v['path']);p=p if p.is_absolute() else R/p
 if p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:raise ValueError('immutable source/proof binding changed: '+str(p))
 return p
def approved(a,plan_ref,script_ref):
 if a.get('approved') is not True or a.get('scope')!='ExpansionRx-independent-original-source-native-task-proof-only':raise ValueError('exact independent source/task proof approval required')
 if a.get('plan_sha256')!=plan_ref['sha256'] or a.get('script_sha256')!=script_ref['sha256']:raise ValueError('exact proof plan/script binding required')
 if a.get('master_publication') is not False or a.get('model_training') is not False:raise ValueError('master publication/model training remain forbidden')
def original_cell(literal):
 if literal=='':return dict(native_literal=literal,status='missing',decimal_literal=None,relation=None,observation_present=False,numeric_loss_eligible=False)
 m=re.fullmatch(r'\s*(<=|>=|<|>|=|≤|≥)?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*',literal)
 if not m or not Decimal(m[2]).is_finite():return dict(native_literal=literal,status='unparsed',decimal_literal=None,relation=None,observation_present=True,numeric_loss_eligible=False)
 relation={'≤':'<=','≥':'>='}.get(m[1],m[1]) or '='
 return dict(native_literal=literal,status='exact' if relation=='=' else 'censored_bound',decimal_literal=m[2],relation=relation,observation_present=True,numeric_loss_eligible=True)
def allocation(family,parent,g,j):
 if family!='ExpansionRx':return None,'ASAP_source_units_and_tabular_transform_unresolved'
 scope=json.loads(g['source_graph_scope_json'])
 if not scope['native_predictor_identity_candidate'] or not scope['legacy_plain_SMILES_exact_comparison_eligible']:return None,'native_graph_representation_scope'
 if json.loads(g['known_current_heldout_group_matches_json']) or json.loads(g['frozen_pilot_reservation_matches_json']):return None,'existing_protected_group_or_frozen_reservation'
 if parent=='native_train':
  if json.loads(j['any_benchmark_heldout_parent_source_families_json']):return None,'known_official_source_heldout_parent_graph_group'
  if j['candidate_train_after_joint_known_source_guards'] is not True:return None,'joint_source_train_guard_not_qualified'
  return 'train','guarded_original_source_train'
 if parent=='native_test':
  if json.loads(g['oldtrain_group_matches_json']):return None,'source_test_known_prior_train_graph_group_exposure'
  return 'test','known_group_screened_original_source_test_subset_not_global_blind_evaluation'
 return None,'raw_only_original_source_parent_unallocated'
def validate_task(row,original,registry,g,j,source_core_sha,endpoint_sha):
 raw=json.loads(row['native_payload']);q=raw['audit']['source_observation'];direction=row['direction'];family=original['family'];entity=original['entity'];endpoint=original['endpoint'];cell=original_cell(original['literal']);review=registry[(family,endpoint)];split,_=allocation(family,original['parent'],g,j)
 if canonical(raw)!=row['native_payload'] or hashlib.sha256(row['native_payload']).hexdigest()!=row['native_payload_sha256']:raise ValueError('canonical native payload differs')
 if hashlib.sha256(canonical({k:v for k,v in q.items() if k not in EXCLUDED})).hexdigest()!=source_core_sha:raise ValueError('unknown original native field/cell/identity/provenance lost or changed')
 if hashlib.sha256(canonical(q['original_endpoint_projection_audit'])).hexdigest()!=endpoint_sha:raise ValueError('original endpoint projection was not retained')
 if q['endpoint_source_review_audit']!=review or q['graph_scope_review_audit']!=json.loads(g['source_graph_scope_json']) or q['source_protocol_group_review_audit']!=g or q['joint_source_benchmark_guard_audit']!=j:raise ValueError('source/protocol/unit proof pointer changed')
 ep=dict(endpoint_key=family+':'+endpoint,source_endpoint=endpoint,readout=review['source_readout'],unit=review['unit_from_official_schema'],unit_basis='official_source_schema' if review['schema_unit_defined'] else 'unresolved',measurement_type='source-curated assay readout; not clinical PK or independent validation')
 context=dict(review['source_context'])
 if review['native_coordinate_only'] and review['source_native_coordinate_objective_supported']:context['source_measurement_coordinate']=review['source_readout']
 if q['endpoint']!=ep or q['known_context']!=context or q['cell']!=cell:raise ValueError('source-native unit/context/value/censor changed')
 entity_rep=original['raw']['SMILES'];assert q['entity']['representation']==entity_rep
 native={k:cell[k] for k in ('native_literal','status','decimal_literal','relation')};common=dict(endpoint=ep,assay_context=context,evidence_role='source_curated_measurement')
 if direction=='entity_to_measurement':conditions=dict(native_entity=q['entity'],**common);targets={'native_measurement':native}
 elif direction=='measurement_to_entity':conditions=dict(desired_native_measurement=native,molecule_class='small_molecule',**common);targets={'native_entity':q['entity']}
 else:raise ValueError('unknown native direction')
 if raw['conditions']!=conditions or raw['targets']!=targets or raw['task']!='source_native_ADMET_'+direction:raise ValueError('hidden target/source audit entered predictors or source target altered')
 groups=sorted(set(json.loads(g['coarse_graph_groups_json']))|{'source_compound:ExpansionRx:'+entity,'source_native_axis_cell:'+row['record_id']})
 tid=hashlib.sha256(canonical(['ExpansionRx-source-native-ADMET-v1',row['record_id'],direction])).hexdigest()
 if row['task_id']!=tid or row['split']!=split or json.loads(row['group_ids_json'])!=groups:raise ValueError('native source task ID/allocation/group mismatch')
 if q['admission']['model_use_authorized'] is not False or q['independent_experimental_measurement_count'] is not None or q['source_native_axis_cell_multiplicity']!=1:raise ValueError('model use or independent assay count promoted')
 if ('physical_unit' in raw['missing'])!=(ep['unit_basis']=='unresolved'):raise ValueError('native coordinate physical-unit missingness changed')
 return groups

def main():
 args=argparse.ArgumentParser();args.add_argument('--approval',required=True);ap=args.parse_args();start=time.monotonic();os.sched_setaffinity(0,{26,27});os.nice(10);pa.set_cpu_count(1);pa.set_io_thread_count(1);plan=json.loads(P.read_text());approval=R/ap.approval;approved(json.loads(approval.read_text()),ref(P),ref(Path(__file__).resolve()))
 for v in list(plan['inputs'].values())+plan['dependencies']:checked(v)
 O=R/plan['output_namespace'];assert not O.exists();O.mkdir();m=json.loads(checked(plan['inputs']['component']).read_text());pm=json.loads(checked(m['inputs']['native_observation_projection']).read_text());gm=json.loads(checked(m['inputs']['chemical_group_review']).read_text());jm=json.loads(checked(m['inputs']['joint_endpoint_review']).read_text());reg=json.loads(checked(jm['endpoint_registry']).read_text());registry={(r['source_family'],r['source_endpoint']):r for r in reg['endpoint_reviews']};graphs={(r['source_family'],r['native_entity_id']):r for r in pq.read_table(checked(gm['source_identity_group_review'])).to_pylist()};joints={(r['source_family'],r['native_entity_id']):r for r in pq.read_table(checked(jm['joint_known_source_guard_rows'])).to_pylist()};source_h=json.loads(checked(m['inputs']['source_handoff']).read_text());sm=json.loads(checked(source_h['manifest']).read_text());original={};source_refs={};parents={}
 def guard():
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;other=Path('/proc/3501176/statm');other_rss=int(other.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') if other.exists() else 0;st=os.statvfs(R)
  if time.monotonic()-start>900 or rss>512*2**20 or rss+other_rss+128*2**20>8*2**30 or st.f_bavail*st.f_frsize<64*2**30:raise RuntimeError('finite source/task proof hardware guard; retain partials')
 for f in sm['source_files']:
  p=checked(f['source'])
  with p.open(newline='',encoding='utf-8-sig') as handle:
   reader=csv.DictReader(handle);assert len(set(reader.fieldnames))==len(reader.fieldnames);rows=list(reader)
  assert len(rows)==f['rows']
  if p.name in ['expansion_data_train.csv','expansion_data_test.csv']:
   for r in rows:
    entity=r['Molecule Name'];assert entity not in parents;parents[entity]='native_train' if p.name.endswith('train.csv') else 'native_test'
  else:
   family='ExpansionRx' if p.name=='expansion_data_raw.csv' else 'ASAP';original[family]=rows;source_refs[family]=f['source']
 assert len(original['ExpansionRx'])==7618 and len(original['ASAP'])==560
 for review in registry.values():
  card=checked(review['source_schema_ref'])
  if 'exact_source_schema_line' in review:assert review['exact_source_schema_line'] in card.read_text().splitlines()
 db=sqlite3.connect(O/'independent-native-source-task-proof.sqlite');db.executescript('CREATE TABLE source_cells(observation_id TEXT PRIMARY KEY,family TEXT,entity TEXT,endpoint TEXT,parent TEXT,status TEXT,expected_split TEXT,reason TEXT,core_sha TEXT,endpoint_sha TEXT,expected_views INTEGER);CREATE TABLE tasks(task_id TEXT PRIMARY KEY,observation_id TEXT,direction TEXT,split TEXT,UNIQUE(observation_id,direction));');old=sqlite3.connect('file:'+str(checked(m['source_cell_and_task_index']))+'?mode=ro&immutable=1',uri=True);old.row_factory=sqlite3.Row;counts=collections.Counter();disposition=collections.Counter()
 for sh in pm['shards']:
  guard()
  for r in pq.read_table(checked(sh)).to_pylist():
   q=json.loads(r['native_observation']);family=r['source_family'];entity=r['source_native_entity_id'];endpoint=r['source_endpoint'];native=original[family][r['source_native_row_index']];assert q['native_source_record']==native and q['entity']['representation']==native['SMILES' if family=='ExpansionRx' else 'CXSMILES'] and entity==native['Molecule Name'];assert q['source_ref']['source']['sha256']==source_refs[family]['sha256'];parent=parents.get(entity,'raw_only_unallocated') if family=='ExpansionRx' else native['Set'];assert q['source_split']['parent_protocol']==parent;cell=original_cell(native[endpoint]);assert q['cell']==cell and hashlib.sha256(r['native_observation']).hexdigest()==r['native_observation_sha256'];g=graphs[(family,entity)];j=joints[(family,entity)];split,reason=allocation(family,parent,g,j)
   if split is not None and not cell['numeric_loss_eligible']:split=None;reason='missing_or_unparsed_source_native_axis_cell'
   expected=2 if split is not None else 0;stored=old.execute('SELECT * FROM source_cells WHERE observation_id=?',(r['observation_id'],)).fetchone();assert stored and dict(stored)==dict(observation_id=r['observation_id'],source_family=family,source_entity=entity,endpoint=endpoint,source_parent=parent,status=cell['status'],disposition=reason,serialized_views=expected);core=hashlib.sha256(canonical({k:v for k,v in q.items() if k not in EXCLUDED})).hexdigest();endpoint_sha=hashlib.sha256(canonical(q['endpoint'])).hexdigest();db.execute('INSERT INTO source_cells VALUES(?,?,?,?,?,?,?,?,?,?,?)',(r['observation_id'],family,entity,endpoint,parent,cell['status'],split,reason,core,endpoint_sha,expected));counts['original_source_axis_slots']+=1;counts['source_status:'+cell['status']]+=1;counts['qualified_source_axis_cells']+=bool(expected);disposition[family+':'+reason]+=1
  db.commit()
 guard();assert counts['original_source_axis_slots']==78980 and counts['qualified_source_axis_cells']==30713;seen_train=set();seen_test=set();task_bytes=0
 for n,sh in enumerate(m['shards']):
  guard();p=checked(sh);gp=checked(m['effective_group_lookup'][sh['path']]);rows=pq.read_table(p).to_pylist();gr=pq.read_table(gp).to_pylist();assert len(rows)==len(gr)==sh['rows'];task_bytes+=p.stat().st_size
  for i,row in enumerate(rows):
   ob=db.execute('SELECT family,entity,endpoint,parent,status,expected_split,core_sha,endpoint_sha FROM source_cells WHERE observation_id=?',(row['record_id'],)).fetchone();assert ob and ob[5];family,entity,endpoint,parent,status,split,core,ep=ob;g=graphs[(family,entity)];j=joints[(family,entity)];native=original[family][g['native_row_index']];orig=dict(family=family,entity=entity,endpoint=endpoint,parent=parent,literal=native[endpoint],raw=native);groups=validate_task(row,orig,registry,g,j,core,ep);assert gr[i]['task_id']==row['task_id'] and json.loads(gr[i]['effective_groups_json'])==groups;loc=old.execute('SELECT * FROM tasks WHERE task_id=?',(row['task_id'],)).fetchone();assert loc and loc['shard_path']==sh['path'] and loc['shard_row']==i and loc['direction']==row['direction'];db.execute('INSERT INTO tasks VALUES(?,?,?,?)',(row['task_id'],row['record_id'],row['direction'],split));(seen_train if split=='train' else seen_test).update(groups);counts['verified_task_views']+=1;counts['verified_task_views:'+split]+=1
  db.commit();print('EXPANSIONRX_INDEPENDENT_TASK_PROOF',n+1,'views',counts['verified_task_views'],flush=True)
 assert not seen_train&seen_test;assert counts['verified_task_views']==61426 and task_bytes==m['total_task_shard_bytes'];missing=db.execute('SELECT count(*) FROM source_cells s LEFT JOIN (SELECT observation_id,count(*) n FROM tasks GROUP BY observation_id) t USING(observation_id) WHERE coalesce(t.n,0)!=s.expected_views').fetchone()[0];assert missing==0;assert dict(disposition)==m['exclusion_dispositions'];old.close();db.close()
 for v in list(plan['inputs'].values())+plan['dependencies']:checked(v)
 guard();receipt=dict(schema='mora3.ExpansionRxIndependentOriginalSourceNativeTaskProof/v1',status='verified_complete_original_source_axis_and_component_tasks_pending_master_union_and_model_qualification',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=ref(P),approval=ref(approval),component=plan['inputs']['component'],counts=dict(counts),all_original_source_CSV_fields_and_native_axis_values_checked=True,all_native_observation_unknown_fields_and_original_endpoint_audit_preserved=True,all_native_literal_censor_units_context_target_direction_ID_location_groups_checked=True,all_source_axis_cell_dispositions_and_task_membership_conserved=True,all_source_known_holdouts_and_existing_graph_exposure_exclusions_preserved=True,new_train_test_known_group_intersections=0,source_axis_cells_with_incorrect_task_multiplicity=missing,task_shard_bytes=task_bytes,independent_source_task_index=ref(O/'independent-native-source-task-proof.sqlite'),global_novelty_or_study_material_assay_identity_not_proven=True,global_master_admitted_tasks=0,model_use=False,substantive_training=False,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);(O/'receipt.json').open('x').write(json.dumps(receipt,indent=2)+'\n');print('COMPLETE_EXPANSIONRX_INDEPENDENT_SOURCE_TASK_PROOF',json.dumps(receipt),flush=True)
if __name__=='__main__':main()
