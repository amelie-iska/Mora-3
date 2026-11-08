"""Independent full original SMILES, census/index and pilot-conservation replay."""
from pathlib import Path
import sys,json,hashlib,sqlite3,collections,time,datetime,os
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_chemical_projection_scope import project_source_graph
from data_intake_full_pubchem_guards import immutable_json
from data_intake_full_pubchem_guards_v2 import BuildGuard
from data_intake_pubchem_native_units import NativeUnitLedger
O=R/'outputs/data-intake/20261101-PubChem-native-corpus-v1';P=O/'build-plan.json';C=O/'census/manifest.json';D=O/'materialized/census-independent-proof-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p
 assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p
def raw_values(path):
 for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,columns=['smiles']):yield from batch.column(0).to_pylist()
def main():
 start=time.perf_counter();plan=json.loads(P.read_text());c=json.loads(C.read_text());assert c['plan']==ref(P);assert len(os.sched_getaffinity(0))<=2;guard=BuildGuard(O,census_cap=2*2**30,total_cap=12*2**30,deadline_epoch=plan['deadline_epoch']);guard.check('materialized',reserve=64*2**20);D.mkdir(parents=True);(D/'code').mkdir();(D/'code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes());source=json.loads(check(plan['source']).read_text());pre=json.loads(check(plan['preflight']).read_text());heldout=set(pre['old_heldout_chemical_guards']);frozen=set(pre['frozen_pilot_validation_chemical_guards']);rawassets={}
 for ch in source['chunks']:
  cp=Path(ch['path']);assert sha(cp)==ch['sha256'];x=json.loads(cp.read_text());a=x['job']['source_asset'];rawassets[a['source_native_split']]=a
 for cp in c['complete_all_source_chunk_receipts']:
  receipt=json.loads(check(cp).read_text());check(receipt['completed_part']);
  if receipt['exceptions'] is not None:check(receipt['exceptions'])
 val=set();train_hashes=bytearray(8999964*32);train_gates=bytearray(8999964);train_SF=bytearray(8999964);source_counts=collections.defaultdict(collections.Counter);eligible=0;raw_count=collections.Counter();component_rows=0;totalparts=0
 for mode in ['validation','train']:
  asset=rawassets[mode];raw=check(asset);assert sha(raw)==asset['expected_sha256'];assert pq.ParquetFile(raw).metadata.num_rows==asset['expected_native_rows'];original=iter(raw_values(raw));position=0;parts=sorted((part for part in c['parts'] if part['mode']==mode),key=lambda p:p['source_native_row_start'])
  for part in parts:
   assert part['source_native_row_start']==position;rows=pq.read_table(check(part)).to_pylist();native_ref=part['source_payload'];check(native_ref);assert len(rows)==part['rows']==native_ref['rows'];native_iter=iter(pq.read_table(native_ref['path']).to_pylist());assert part['source_native_row_stop']==position+len(rows)
   for row in rows:
    value=next(original);native=next(native_iter);assert native['native_smiles']==value and native['source_native_file_row_index']==row['source_native_row']==position and native['source_native_split']==mode;expectedhash=hashlib.sha256(value.encode()).digest() if isinstance(value,str) else None;assert row['source_native_SMILES_sha256']==expectedhash;assert row['source_native_parse_pass']==native['native_current_RDKit_parse_pass'] and row['source_SELFIES_target_present']==native['standard_selfies_target_present'];iso=row['canonical_isomeric_graph_sha256'];passed=row['derived_native_graph_pass'];code=row['admission_gate_code'];source_counts[mode]['rows']+=1;source_counts[mode]['native_graph_pass']+=passed;source_counts[mode]['native_parse_flag']+=native['native_current_RDKit_parse_pass'];source_counts[mode]['source_codec_present']+=native['standard_selfies_target_present'];source_counts[mode]['derived_codec_pass']+=row['derived_SELFIES_graph_pass']
    if not passed:assert code&(1|32)
    else:
     assert native['native_current_RDKit_parse_pass'] and iso==bytes.fromhex(native['source_current_canonical_graph_sha256']) and not row['lossy_stereo_warning'];assert row['molecule_guard_sha256'] is not None and row['scaffold_guard_sha256'] is not None
     if mode=='validation':assert code==2;val.add(iso);assert row['derived_SELFIES_graph_pass'] is False
     else:
      groups={'molecule:'+row['molecule_guard_sha256'].hex(),'scaffold:'+row['scaffold_guard_sha256'].hex()}
      if '.' in value:
       projection=project_source_graph(value);assert projection['exact_projection_eligible'];assert hashlib.sha256(projection['graphs'][0][1]['canonical_isomeric_smiles'].encode()).digest()==iso;groups.update(ident[k] for kind,ident in projection['graphs'] for k in ['molecule_group','scaffold_group']);component_rows+=1
      expected=(4 if iso in val else 0)|(8 if groups&heldout else 0)|(16 if groups&frozen else 0);assert code==expected;assert row['derived_SELFIES_graph_pass']==native['graph_roundtrip_pass']==native['standard_selfies_target_present']
    if mode=='train':
     train_gates[position]=code;train_SF[position]=row['derived_SELFIES_graph_pass'];
     if iso is not None:train_hashes[position*32:(position+1)*32]=iso
     eligible+=passed and code==0
    position+=1;raw_count[mode]+=1
   assert next(native_iter,None) is None;totalparts+=1;guard.check('materialized');
   if totalparts%20==0:print('FULL_CENSUS_RAW_REPLAY',mode,position,'parts',totalparts,'seconds',round(time.perf_counter()-start,1),flush=True)
  sentinel=object();assert next(original,sentinel) is sentinel and position==asset['expected_native_rows']
 assert dict(raw_count)=={'validation':999996,'train':8999964} and totalparts==2000 and len(val)==c['global_validation_exact_unique_graphs'];indexpath=check(c['graph_index']);db=sqlite3.connect('file:'+str(indexpath)+'?mode=ro&immutable=1',uri=True);assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok';assert db.execute('SELECT COUNT(*) FROM validation_graphs').fetchone()[0]==len(val);assert db.execute('SELECT COUNT(*) FROM graphs g JOIN validation_graphs v ON g.iso=v.iso').fetchone()[0]==0
 pilot=json.loads(check(plan['pilot_component']).read_text());pilot_ledger=pq.read_table(check(pilot['selected_source_ledger'])).to_pylist();pilot_map={};pilot_units=NativeUnitLedger()
 for row in pilot_ledger:
  if json.loads(row['admitted_task_ids_json']):
   native=json.loads(row['all_native_source_columns_json']);pilot_map[bytes.fromhex(native['source_current_canonical_graph_sha256'])]=native;pilot_units.reference_graph(native);pilot_units.view('reference_prior',native)
   if native['graph_roundtrip_pass']:
    pilot_units.codec_graph(native);pilot_units.view('smiles_to_selfies',native);pilot_units.view('selfies_to_smiles',native)
 count=codec=pilot_prior=pilot_codec=0
 for iso,prior,encoding,oldprior,oldcodec in db.execute('SELECT iso,prior_row,codec_row,pilot_prior,pilot_codec FROM graphs'):
  assert train_gates[prior]==0 and train_hashes[prior*32:(prior+1)*32]==iso
  if encoding is not None:assert train_gates[encoding]==0 and train_SF[encoding] and train_hashes[encoding*32:(encoding+1)*32]==iso;codec+=1
  if oldprior:
   native=pilot_map[iso];assert prior==native['source_native_file_row_index'];pilot_prior+=1
  if oldcodec:
   native=pilot_map[iso];assert native['graph_roundtrip_pass'] and encoding==native['source_native_file_row_index'];pilot_codec+=1
  count+=1
 assert count==c['admitted_unique_reference_graphs'] and codec==c['graphs_with_eligible_standardSELFIES'] and pilot_prior==27319 and pilot_codec==26918
 cursor=db.cursor();membership=0
 for row in range(8999964):
  if train_gates[row]!=0:continue
  iso=bytes(train_hashes[row*32:(row+1)*32]);found=cursor.execute('SELECT prior_row,codec_row,pilot_prior,pilot_codec FROM graphs WHERE iso=?',(iso,)).fetchone();assert found is not None;prior,encoding,oldprior,oldcodec=found;assert oldprior or prior<=row
  if train_SF[row]:assert encoding is not None and (oldcodec or encoding<=row)
  membership+=1
  if row%500000==0:print('FULL_CENSUS_ELIGIBLE_INDEX',row,'members',membership,'seconds',round(time.perf_counter()-start,1),flush=True);guard.check('materialized')
 assert membership==eligible;db.close();assert ref(indexpath)==c['graph_index'];usage=guard.check('materialized',reserve=2**20);result=dict(schema='mora3.PubChemWholeNativeCensusIndependentProof/v1',status='verified',plan=ref(P),census=ref(C),source=plan['source'],master=plan['master'],all_original_raw_source_SMILES_rows_replayed=dict(raw_count),all2000_census_and_native_payload_bytes_SHA_rows_refs_checked=True,all_native_original_SMILES_exact_and_source_derived_context_fields_preserved_in_original_refs=True,source_counts={k:dict(v) for k,v in source_counts.items()},native_validation_audit_only=True,validation_unique_graphs=len(val),all_train_whole_or_component_guards_replayed_against_old_heldout_and_frozen_validation=True,source_component_graph_rows_independently_reparsed=component_rows,all_source_native_SF_graph_QC='fullphaseAcensusverifiedeveryeligibleencoding_againstsourcecanonicalgraph;thisraw/indexreplaydoesnotclaimasecondfullSFdecode',all_eligible_original_source_occurrences_accounted_in_graph_index=membership,all_graph_index_rows_source_and_encoding_masks_bound=count,all_graphs_with_supported_codec=codec,all27319_pilot_prior_and26918codec_graph_refs_conserved=True,preserved_pilot_native_unit_ledger=pilot_units.snapshot(),native_units_are_not_actual_model_tokens=True,no_new_tasks_yet=True,all_graph_index_SHA_unchanged=True,phase_usage=usage,deadline_epoch=plan['deadline_epoch'],seconds=time.perf_counter()-start);immutable_json(D/'receipt.json',result);guard.check('materialized');print('COMPLETE_FULL_CENSUS_INDEPENDENT_PROOF',json.dumps({k:result[k] for k in ['all_original_raw_source_SMILES_rows_replayed','all_eligible_original_source_occurrences_accounted_in_graph_index','all_graph_index_rows_source_and_encoding_masks_bound','seconds']}),flush=True)
if __name__=='__main__':main()
