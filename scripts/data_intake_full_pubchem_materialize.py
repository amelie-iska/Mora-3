"""Materialize only independently indexed eligible new prior/codec views; retain pilot once."""
from pathlib import Path
import sys,json,hashlib,datetime,time,collections,sqlite3,bisect,functools,os
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import CorpusTask
from data_intake_PubChem_tasks import smiles_prior,codec_task
from data_intake_chemical_projection_scope import project_source_graph
from data_intake_full_pubchem_guards import immutable_json,task_identifier
from data_intake_full_pubchem_guards_v2 import BuildGuard
from data_intake_pubchem_native_units import NativeUnitLedger
from data_intake_full_pubchem_selection import create_selection_schema,queue_graph_views
O=R/'outputs/data-intake/20261101-PubChem-native-corpus-v1';P=O/'build-plan.json';C=O/'census/manifest.json';D=O/'materialized'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p
 assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p
def readonly_db(path):return sqlite3.connect('file:'+str(path)+'?mode=ro&immutable=1',uri=True)
def main():
 start=time.perf_counter();plan=json.loads(P.read_text());c=json.loads(C.read_text());assert c['source_train_rows']==8999964 and c['source_validation_audit_only_rows']==999996 and c['pilot_prior_graphs_reused']==27319 and c['pilot_codec_graphs_reused']==26918;assert c['plan']==ref(P);assert len(os.sched_getaffinity(0))<=2;guard=BuildGuard(O,census_cap=2*2**30,total_cap=12*2**30,deadline_epoch=plan['deadline_epoch']);guard.check('materialized',reserve=64*2**20);proofpath=D/'census-independent-proof-v1/receipt.json';proof=json.loads(proofpath.read_text());assert proof['status']=='verified' and proof['census']==ref(C);units=NativeUnitLedger(proof['preserved_pilot_native_unit_ledger']);assert {p.name for p in D.iterdir()}=={'census-independent-proof-v1'};(D/'shards').mkdir();(D/'groups').mkdir();(D/'checkpoints').mkdir();(D/'code').mkdir()
 for p in [Path(__file__),R/'scripts/data_intake_PubChem_tasks.py',R/'scripts/data_intake_full_pubchem_guards.py',R/'scripts/data_intake_full_pubchem_selection.py']:(D/'code'/p.name).write_bytes(p.read_bytes())
 graphpath=check(c['graph_index']);graphs=readonly_db(graphpath);indexpath=D/'source-selection.sqlite';index=sqlite3.connect(indexpath);index.execute('PRAGMA journal_mode=DELETE');index.execute('PRAGMA synchronous=FULL');create_selection_schema(index);index.commit();graphcount=selected=0
 for iso,prior,codec,pilot_prior,pilot_codec in graphs.execute('SELECT iso,prior_row,codec_row,pilot_prior,pilot_codec FROM graphs'):
  if graphcount%5000==0:guard.check('materialized',reserve=64*2**20)
  queue_graph_views(index,iso,prior,codec,pilot_prior,pilot_codec)
  graphcount+=1
  if graphcount%5000==0:index.commit()
  if graphcount%500000==0:print('FULL_PUBCHEM_SELECTION',graphcount,'seconds',round(time.perf_counter()-start,1),flush=True)
 index.commit();assert graphcount==c['admitted_unique_reference_graphs'];counts=index.execute('SELECT SUM(emit_prior),SUM(emit_codec),COUNT(*) FROM selection').fetchone();assert counts[0]==c['new_prior_graphs_to_serialize'] and counts[1]==c['new_codec_graphs_to_serialize'];target_count=counts[0]+2*counts[1];index.close();selection_reader=readonly_db(indexpath);taskpath=D/'task-id-index.sqlite';index=sqlite3.connect(taskpath);index.execute('PRAGMA journal_mode=DELETE');index.execute('PRAGMA synchronous=FULL');index.execute('CREATE TABLE task_ids(id BLOB PRIMARY KEY) WITHOUT ROWID');index.commit()
 source=json.loads(check(plan['source']).read_text());jobs=sorted([j['source_payload'] for j in plan['source_payload_jobs'] if j['mode']=='train'],key=lambda f:f['source_row_start']);starts=[f['source_row_start'] for f in jobs];census={part['source_payload']['path']:part for part in c['parts'] if part['mode']=='train'};master=json.loads(check(plan['master']).read_text());pre=json.loads(check(plan['preflight']).read_text());heldout=set(pre['old_heldout_chemical_guards']);frozen=set(pre['frozen_pilot_validation_chemical_guards']);cache={};verified=set()
 def lookup(ordinal):
  i=bisect.bisect_right(starts,ordinal)-1;assert i>=0;file=jobs[i];assert file['source_row_start']<=ordinal<file['source_row_stop'];path=file['path']
  if path not in cache:
   cache.clear();native=check(file);part=census[path];derived=check(part);cache[path]=(pq.read_table(native).to_pylist(),pq.read_table(derived).to_pylist());verified.add(path)
  native,derived=cache[path];local=ordinal-file['source_row_start'];row=native[local];qc=derived[local];assert row['source_native_file_row_index']==qc['source_native_row']==ordinal and row['source_native_split']=='train' and qc['admission_gate_code']==0 and qc['derived_native_graph_pass'];assert hashlib.sha256(row['native_smiles'].encode()).digest()==qc['source_native_SMILES_sha256'];return row,qc,file,local,census[path]
 buffers=collections.defaultdict(list);shards=[];lookuprefs={};taskcounts=collections.Counter();axis=collections.defaultdict(lambda:dict(min=None,max=0,values=0));occurrences=0;source_ref=plan['source'];pilot_ids=set()
 pilot=json.loads(check(plan['pilot_component']).read_text())
 for sh in pilot['shards']:
  for row in pq.read_table(check(sh),columns=['task_id']).to_pylist():pilot_ids.add(row['task_id'])
 assert len(pilot_ids)==81155
 def flush(family):
  rows=buffers[family]
  if not rows:return
  table=pa.Table.from_pylist(rows);guard.check('materialized',reserve=table.nbytes*2+64*2**20);path=D/'shards'/f'{len(shards):06d}.parquet';pq.write_table(table,path,compression='zstd');assert pq.read_table(path).to_pylist()==rows;shards.append(ref(path,rows=len(rows),source_family=family,split='train'));gp=D/'groups'/path.name;groups=[dict(task_id=r['task_id'],effective_groups_json=r['group_ids_json']) for r in rows];pq.write_table(pa.Table.from_pylist(groups),gp,compression='zstd');assert pq.read_table(gp).to_pylist()==groups;lookuprefs[str(path.relative_to(R))]=ref(gp,rows=len(groups));index.commit();buffers[family]=[];guard.check('materialized')
 def register(row,qc,file,local,censuspart,iso,family,direction,task,groups):
  units.view(direction,row);tid=task_identifier(family,row['source_record_id'],direction);assert tid not in pilot_ids;index.execute('INSERT INTO task_ids VALUES(?)',(bytes.fromhex(tid),));raw=task.canonical_bytes();d=json.loads(raw);assert not any(k in d['conditions'] for k in ['source_record_id','source_native_split','target_length','source_current_canonical_graph_sha256','status','derived_atom_count']);target=next(iter(d['targets'].values()));targetlen=len(target);a=axis[family+':'+direction];a['min']=targetlen if a['min'] is None else min(a['min'],targetlen);a['max']=max(a['max'],targetlen);a['values']+=targetlen;taskcounts[family+':'+direction]+=1
  sr=dict(path=str(Path(file['path']).relative_to(R)),bytes=file['bytes'],sha256=file['sha256'],row_index=local)
  buffers[family].append(dict(task_id=tid,record_id=row['source_record_id'],source_family=family,source_revision=source['source_revision'],source_sha256=source_ref['sha256'],task_name=d['task'],native_schema='mora3.CorpusTask/v1',native_payload=raw,native_payload_sha256=hashlib.sha256(raw).hexdigest(),split='train',split_policy='native_source_trainonly;globalnativevalidationexact+v20heldoutwhole/componentchemistry+frozenpilotvalidationguards_excluded;pilotpreservedonce',group_ids_json=json.dumps(sorted(groups)),evidence_class='source_reference_structure' if direction=='reference_prior' else 'computed_lossless_representation_not_experimental_response',admitted_for_task=True,model_adapter_supported=False,admission_receipt_json=json.dumps(dict(source_ref=d['audit']['source_ref'],census_ref=dict(path=censuspart['path'],sha256=censuspart['sha256'],row_index=local),source_license='Apache-2.0explicitDATA',strict_source_stereo_projection_available=True,original_source_graph_sha256=iso.hex(),no_3D_or_experimental_response_pair=True)),missing_context_json=json.dumps(d['missing']),full_axis_values=targetlen,condition_context_json=json.dumps(dict(assay=None,dose=None,time=None,image_available=False,mask_available=False,expression_available=False,segmentation_available=False))))
  if len(buffers[family])>=1024:flush(family)
 cursor=selection_reader.execute('SELECT source_row,iso,emit_prior,emit_codec FROM selection ORDER BY source_row')
 for ordinal,iso,emit_prior,emit_codec in cursor:
  if occurrences%1000==0:guard.check('materialized',reserve=64*2**20)
  row,qc,file,local,part=lookup(ordinal);assert qc['canonical_isomeric_graph_sha256']==iso;assert hashlib.sha256(row['source_canonical_isomeric_smiles'].encode()).digest()==iso;groups={'molecule:'+qc['molecule_guard_sha256'].hex(),'scaffold:'+qc['scaffold_guard_sha256'].hex(),'chemical_exact_isomeric:'+iso.hex()}
  if '.' in row['native_smiles']:
   projection=project_source_graph(row['native_smiles']);assert projection['exact_projection_eligible'];groups.update(ident[k] for kind,ident in projection['graphs'] for k in ['molecule_group','scaffold_group'])
  assert not groups&heldout and not groups&frozen;sr=dict(native_source=source_ref,native_record=dict(path=str(Path(file['path']).relative_to(R)),bytes=file['bytes'],sha256=file['sha256'],row_index=local),source_record_id=row['source_record_id'],source_native_split='train',source_native_row=ordinal);validated=dict(smiles_eligible=True,selfies_eligible=qc['derived_SELFIES_graph_pass']);assert validated['selfies_eligible']==row['graph_roundtrip_pass']==row['standard_selfies_target_present']
  if emit_prior:
   units.reference_graph(row);register(row,qc,file,local,part,iso,'PubChem-native-SMILES-reference-prior','reference_prior',smiles_prior(row,sr,validated=validated),groups)
  if emit_codec:
   units.codec_graph(row);assert qc['derived_SELFIES_graph_pass']
   for direction in ['smiles_to_selfies','selfies_to_smiles']:register(row,qc,file,local,part,iso,'PubChem-standardSELFIES-codec',direction,codec_task(row,direction,sr,validated=validated),groups)
  occurrences+=1
  if occurrences%5000==0:
   for family in list(buffers):flush(family)
   index.commit();usage=guard.check('materialized');cp=D/'checkpoints'/f'{occurrences:09d}.json';immutable_json(cp,dict(source_selection_rows_done=occurrences,last_source_row=ordinal,task_counts=dict(taskcounts),completed_shards=len(shards),usage=usage,deadline_epoch=plan['deadline_epoch']));(O/'live-progress.json').write_text(json.dumps(dict(phase='full_task_materialization',source_selection_rows_done=occurrences,source_selection_rows_total=counts[2],task_counts=dict(taskcounts),completed_shards=len(shards),usage=usage,seconds=time.perf_counter()-start,deadline_epoch=plan['deadline_epoch']),indent=2)+'\n')
  if occurrences%100000==0:print('FULL_PUBCHEM_TASKS',occurrences,'views',sum(taskcounts.values()),'seconds',round(time.perf_counter()-start,1),flush=True)
 for family in list(buffers):flush(family)
 index.commit();assert occurrences==counts[2] and sum(taskcounts.values())==target_count and sum(s['rows'] for s in shards)==target_count;assert index.execute('SELECT COUNT(*) FROM task_ids').fetchone()[0]==target_count;assert index.execute('PRAGMA integrity_check').fetchone()[0]=='ok';index.close();selection_reader.close();graphs.close();assert units.references['graphs']==c['admitted_unique_reference_graphs'] and units.codecs['graphs']==c['graphs_with_eligible_standardSELFIES'];assert ref(graphpath)==c['graph_index'];usage=guard.check('materialized');manifest=dict(schema='mora3.PubChemWholeNativeNewTaskComponent/v1',status='serialized_pending_independent_whole_raw_census_task_and_union_proof',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=ref(P),census=ref(C),independent_census_proof=ref(proofpath),prior_master=plan['master'],source=source_ref,pilot_component=plan['pilot_component'],all_pilot_task_ids_payloads_inherited_once=True,new_task_views=target_count,new_prior_graphs=counts[0],new_codec_graphs=counts[1],selected_original_source_rows_for_new_views=counts[2],task_counts=dict(taskcounts),new_splits={'train':target_count},full_native_axes=axis,serialized_PubChem_native_units=units.snapshot(),native_units_include_preserved_pilot_once=True,source_files_consumed_for_new_targets=len(verified),shards=shards,effective_group_lookup=lookuprefs,task_shard_bytes=sum(s['bytes'] for s in shards),new_source_selection_index=ref(indexpath),new_task_id_uniqueness_index=ref(taskpath),native_schema='mora3.CorpusTask/v1',model_adapter_supported=False,source_train_occurrences=8999964,source_validation_audit_only_occurrences=999996,all_admitted_unique_reference_graphs=c['admitted_unique_reference_graphs'],all_admitted_codec_graphs=c['graphs_with_eligible_standardSELFIES'],new_graphs_not_independent_experiments=True,optimizer_steps=0,final_mixture_weights=None,phase_usage=usage,deadline_epoch=plan['deadline_epoch'],seconds=time.perf_counter()-start);immutable_json(D/'manifest.json',manifest);guard.check('materialized');print('COMPLETE_FULL_PUBCHEM_TASKS',json.dumps({k:manifest[k] for k in ['new_task_views','new_prior_graphs','new_codec_graphs','selected_original_source_rows_for_new_views','task_shard_bytes','seconds']}),flush=True)
if __name__=='__main__':main()
