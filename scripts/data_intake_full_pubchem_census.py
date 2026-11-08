"""Finite whole-native-source census. New indexes only; no task/model mutation."""
from pathlib import Path
import sys,json,hashlib,collections,datetime,time,sqlite3,concurrent.futures,importlib.metadata,multiprocessing
import pyarrow as pa,pyarrow.parquet as pq
import selfies as sf
from rdkit import Chem,rdBase
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_PubChem_tasks import validate_native
from data_intake_chemical_projection_scope import project_source_graph
from data_intake_full_pubchem_guards import BuildGuard,immutable_json
SCHEMA=pa.schema([('source_native_row',pa.int64()),('source_native_SMILES_sha256',pa.binary(32)),('canonical_isomeric_graph_sha256',pa.binary(32)),('molecule_guard_sha256',pa.binary(32)),('scaffold_guard_sha256',pa.binary(32)),('source_native_parse_pass',pa.bool_()),('source_SELFIES_target_present',pa.bool_()),('derived_native_graph_pass',pa.bool_()),('derived_SELFIES_graph_pass',pa.bool_()),('lossy_stereo_warning',pa.bool_()),('admission_gate_code',pa.uint16())])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p
 assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p
def validate_rows(rows,mode):
 if mode not in ['train','validation']:raise ValueError('explicit source mode required')
 results=[];errors=[];groups=[];counts=collections.Counter()
 for local,row in enumerate(rows):
  native=row['native_smiles'];out=dict(source_native_row=row['source_native_file_row_index'],source_native_SMILES_sha256=hashlib.sha256(native.encode()).digest() if isinstance(native,str) else None,canonical_isomeric_graph_sha256=None,molecule_guard_sha256=None,scaffold_guard_sha256=None,source_native_parse_pass=row['native_current_RDKit_parse_pass'],source_SELFIES_target_present=row['standard_selfies_target_present'],derived_native_graph_pass=False,derived_SELFIES_graph_pass=False,lossy_stereo_warning=False,admission_gate_code=0);gs=[]
  try:
   if row['coordinate_geometry_present'] is not False:raise ValueError('unexpected source geometry claim')
   if row['native_current_RDKit_parse_pass'] is not True:
    with rdBase.BlockLogs():mol=Chem.MolFromSmiles(native) if isinstance(native,str) else None
    if mol is not None:raise ValueError('source native-parse mask contradicts independent parse')
    raise ValueError('source native graph invalid; raw reference retained')
   p=project_source_graph(native);out['lossy_stereo_warning']=p['lossy_stereo_warning']
   if not p['exact_projection_eligible']:raise ValueError('lossy source stereo projection; exact graph prior/codec gated')
   ident=p['graphs'][0][1];iso=ident['canonical_isomeric_smiles'];digest=hashlib.sha256(iso.encode()).hexdigest()
   if iso!=row['source_canonical_isomeric_smiles'] or digest!=row['source_current_canonical_graph_sha256']:raise ValueError('source canonical graph/normalization mismatch; no silent identity change')
   if mode=='train':v=validate_native(row);assert v['identity']==ident;out['derived_SELFIES_graph_pass']=v['selfies_eligible']
   else:out['derived_SELFIES_graph_pass']=False
   out.update(canonical_isomeric_graph_sha256=bytes.fromhex(digest),molecule_guard_sha256=bytes.fromhex(ident['molecule_group'].split(':',1)[1]),scaffold_guard_sha256=bytes.fromhex(ident['scaffold_group'].split(':',1)[1]),derived_native_graph_pass=True)
   gs=sorted({identity[key] for kind,identity in p['graphs'] for key in ['molecule_group','scaffold_group']})
  except Exception as exc:
   out['admission_gate_code']|=32 if out['lossy_stereo_warning'] else 1;errors.append(dict(local_row=local,source_record_id=row['source_record_id'],source_native_row=row['source_native_file_row_index'],error=repr(exc),source_parse_pass=row['native_current_RDKit_parse_pass'],source_status=row['status'],originals_unchanged=True))
  results.append(out);groups.append(gs);counts['rows']+=1;counts['source_parse_pass']+=row['native_current_RDKit_parse_pass'];counts['derived_native_graph_pass']+=out['derived_native_graph_pass'];counts['source_SELFIES_present']+=row['standard_selfies_target_present'];counts['derived_SELFIES_pass']+=out['derived_SELFIES_graph_pass'];counts['lossy_stereo_warning']+=out['lossy_stereo_warning']
 return results,groups,errors,dict(counts)
def worker(job):
 path=check(job['source_payload']);rows=pq.read_table(path).to_pylist();assert len(rows)==job['source_payload']['rows'];start=job['source_payload']['source_row_start'];stop=job['source_payload']['source_row_stop'];assert [r['source_native_file_row_index'] for r in rows]==list(range(start,stop));assert all(r['source_native_split']==job['mode'] for r in rows);out=validate_rows(rows,job['mode']);unexpected=sum(e['source_parse_pass'] for e in out[2]);assert unexpected<=max(50,len(rows)//100),'systematic source/decoder failure, preserve receipt and stop';return out
def ordered_parallel(jobs,deadline):
 pool=concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context("spawn"));pending=collections.deque();iterator=iter(jobs)
 try:
  for _ in range(2):
   job=next(iterator,None)
   if job is not None:pending.append((job,pool.submit(worker,job)))
  while pending:
   job,future=pending.popleft();remaining=deadline-time.time()
   if remaining<=0:raise RuntimeError('finite deadline reached')
   value=future.result(timeout=min(600,remaining));yield job,value;job=next(iterator,None)
   if job is not None:pending.append((job,pool.submit(worker,job)))
 finally:pool.shutdown(wait=True,cancel_futures=True)
def main():
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--plan',required=True);args=a.parse_args();planpath=Path(args.plan);plan=json.loads(planpath.read_text());O=R/plan['output_namespace'];C=O/'census';C.mkdir();(C/'parts').mkdir();(C/'checkpoints').mkdir();(C/'exceptions').mkdir();(C/'code').mkdir();guard=BuildGuard(O,census_cap=2*2**30,total_cap=12*2**30,deadline_epoch=plan['deadline_epoch']);guard.check('census',reserve=8*2**20);start=time.perf_counter();assert len(__import__('os').sched_getaffinity(0))<=2;source=json.loads(check(plan['source']).read_text());assert {n:importlib.metadata.version(n) for n in source['versions']}==source['versions'] and sf.get_semantic_constraints()==source['semantic_constraints']
 for file in [Path(__file__),'scripts/data_intake_full_pubchem_guards.py','scripts/data_intake_PubChem_tasks.py','scripts/data_intake_chemical_projection_scope.py']:
  p=Path(file);p=p if p.is_absolute() else R/p;(C/'code'/p.name).write_bytes(p.read_bytes())
 dbpath=C/'graph-index.sqlite';assert not dbpath.exists();db=sqlite3.connect(dbpath);db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL');db.execute('CREATE TABLE validation_graphs(iso BLOB PRIMARY KEY) WITHOUT ROWID');db.execute('CREATE TABLE graphs(iso BLOB PRIMARY KEY,prior_row INTEGER NOT NULL,codec_row INTEGER,pilot_prior INTEGER NOT NULL,pilot_codec INTEGER NOT NULL) WITHOUT ROWID');db.commit()
 pre=json.loads(check(plan['preflight']).read_text());heldout=set(pre['old_heldout_chemical_guards']);pilot_validation=set(pre['frozen_pilot_validation_chemical_guards']);pilot=json.loads(check(plan['pilot_component']).read_text());ledger=pq.read_table(check(pilot['selected_source_ledger'])).to_pylist();seeded=0;pilot_rows={}
 for row in ledger:
  tids=json.loads(row['admitted_task_ids_json'])
  if not tids:continue
  native=json.loads(row['all_native_source_columns_json']);assert native['source_native_split']=='train';iso=bytes.fromhex(native['source_current_canonical_graph_sha256']);ordinal=native['source_native_file_row_index'];codec=ordinal if native['graph_roundtrip_pass'] else None;db.execute('INSERT INTO graphs VALUES(?,?,?,?,?)',(iso,ordinal,codec,1,int(codec is not None)));seeded+=1;pilot_rows[ordinal]=iso
 assert seeded==27319;db.commit();checkpoints=[];totals=collections.defaultdict(collections.Counter);parts=[];validation_hashes=set();train_next=validation_next=0;new_graph_rows=0;gatecounts=collections.Counter();files=plan['source_payload_jobs'];assert len(files)==2000
 for mode in ['validation','train']:
  jobs=[j for j in files if j['mode']==mode]
  if mode=='train':assert db.execute('SELECT COUNT(*) FROM graphs g JOIN validation_graphs v ON v.iso=g.iso').fetchone()[0]==0
  for seq,(job,result) in enumerate(ordered_parallel(jobs,plan['deadline_epoch'])):
   records,groups,errors,counts=result;guard.check('census',reserve=pa.Table.from_pylist(records,schema=SCHEMA).nbytes*2+64*2**20)
   if mode=='validation':
    assert records[0]['source_native_row']==validation_next;validation_next+=len(records)
    for row in records:
     row['admission_gate_code']|=2
     if row['derived_native_graph_pass']:validation_hashes.add(row['canonical_isomeric_graph_sha256']);db.execute('INSERT OR IGNORE INTO validation_graphs VALUES(?)',(row['canonical_isomeric_graph_sha256'],))
   else:
    assert records[0]['source_native_row']==train_next;train_next+=len(records)
    for row,gs in zip(records,groups):
     if row['derived_native_graph_pass']:
      iso=row['canonical_isomeric_graph_sha256']
      if iso in validation_hashes:row['admission_gate_code']|=4
      if heldout&set(gs):row['admission_gate_code']|=8
      if pilot_validation&set(gs):row['admission_gate_code']|=16
      if row['admission_gate_code']==0:
       ordinal=row['source_native_row'];codec=ordinal if row['derived_SELFIES_graph_pass'] else None;db.execute('INSERT OR IGNORE INTO graphs VALUES(?,?,?,0,0)',(iso,ordinal,codec))
       if codec is not None:db.execute('UPDATE graphs SET codec_row=? WHERE iso=? AND codec_row IS NULL',(ordinal,iso))
     if row['source_native_row'] in pilot_rows:assert row['admission_gate_code']==0 and row['canonical_isomeric_graph_sha256']==pilot_rows[row['source_native_row']],'pilotconservationgatefailed'
     gatecounts[str(row['admission_gate_code'])]+=1
   db.commit();out=C/'parts'/f'{mode}-{seq:04d}.parquet';table=pa.Table.from_pylist(records,schema=SCHEMA);pq.write_table(table,out,compression='zstd');assert pq.read_table(out).to_pylist()==records;part=ref(out,rows=len(records),mode=mode,source_payload=job['source_payload'],source_native_row_start=records[0]['source_native_row'],source_native_row_stop=records[-1]['source_native_row']+1);parts.append(part)
   errorref=None
   if errors:
    e=C/'exceptions'/f'{mode}-{seq:04d}.json';immutable_json(e,dict(source_payload=job['source_payload'],rows=errors,source_unknown_fields_retained_in_immutable_payload=True));errorref=ref(e)
   totals[mode].update(counts);usage=guard.check('census');cp=C/'checkpoints'/f'{mode}-{seq:04d}.json';immutable_json(cp,dict(source_chunk=job['source_payload'],completed_part=part,exceptions=errorref,cumulative=dict(totals[mode]),train_next=train_next,validation_next=validation_next,usage=usage,deadline_epoch=plan['deadline_epoch'],graph_index_mutable_only_during_this_authorized_new_build=True));checkpoints.append(ref(cp));live=dict(phase='whole_native_census',mode=mode,source_chunks_done=len(parts),source_chunks_total=len(files),cumulative={k:dict(v) for k,v in totals.items()},train_next=train_next,validation_next=validation_next,new_graph_index_rows=db.execute('SELECT COUNT(*) FROM graphs').fetchone()[0] if seq%50==0 else None,usage=usage,elapsed_seconds=time.perf_counter()-start,deadline_epoch=plan['deadline_epoch']);(O/'live-progress.json').write_text(json.dumps(live,indent=2)+'\n')
   if seq%20==0:print('FULL_PUBCHEM_CENSUS',mode,seq+1,'rows',train_next,validation_next,'bytes',usage['all_bytes'],'seconds',round(time.perf_counter()-start,1),flush=True)
 assert train_next==8999964 and validation_next==999996;assert totals['train']['source_parse_pass']==8999908 and totals['train']['source_SELFIES_present']==8999210;assert db.execute('SELECT COUNT(*) FROM graphs WHERE pilot_prior=1').fetchone()[0]==27319;assert db.execute('SELECT COUNT(*) FROM graphs WHERE pilot_codec=1').fetchone()[0]==26918;graph_count=db.execute('SELECT COUNT(*) FROM graphs').fetchone()[0];codec_count=db.execute('SELECT COUNT(*) FROM graphs WHERE codec_row IS NOT NULL').fetchone()[0];assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok';db.close();usage=guard.check('census');manifest=dict(schema='mora3.PubChemWholeNativeCensus/v1',status='complete_census_pending_full_raw_hash_independent_replay_and_materialization',plan=ref(planpath),source=plan['source'],prior_master=plan['master'],source_raw_available=9999960,source_train_rows=train_next,source_validation_audit_only_rows=validation_next,source_counts={k:dict(v) for k,v in totals.items()},complete_all_source_chunk_receipts=checkpoints,parts=parts,graph_index=ref(dbpath),admitted_unique_reference_graphs=graph_count,graphs_with_eligible_standardSELFIES=codec_count,pilot_prior_graphs_reused=27319,pilot_codec_graphs_reused=26918,new_prior_graphs_to_serialize=graph_count-27319,new_codec_graphs_to_serialize=codec_count-26918,train_gate_code_counts=dict(gatecounts),gate_code_bits=plan['gate_code_bits'],global_validation_exact_unique_graphs=len(validation_hashes),old_validation_test_and_frozen_pilot_guards_preserved=True,strict_stereo_projection_required=True,no_new_task_shards_yet=True,all_source_occurrences_retained_in_census=True,all_unknown_native_fields_retained_by_full_source_payload_refs=True,model_adapter_supported=False,optimizer_steps=0,all_new_derived_bytes=usage['all_bytes'],phase_cap=2*2**30,deadline_epoch=plan['deadline_epoch'],seconds=time.perf_counter()-start);immutable_json(C/'manifest.json',manifest);guard.check('census');print('COMPLETE_FULL_PUBCHEM_CENSUS',json.dumps({k:manifest[k] for k in ['source_train_rows','source_validation_audit_only_rows','admitted_unique_reference_graphs','graphs_with_eligible_standardSELFIES','new_prior_graphs_to_serialize','new_codec_graphs_to_serialize','all_new_derived_bytes','seconds']}),flush=True)
if __name__=='__main__':main()
