"""Bounded post-proof audit links to existing native response/property contexts."""
from pathlib import Path
import collections,datetime,hashlib,json,os,sqlite3,sys,time
import pyarrow as pa
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_chemical_projection_scope import project_source_graph
from data_intake_full_pubchem_guards import task_identifier,immutable_json
from data_intake_full_pubchem_guards_v2 import BuildGuard
from data_intake_reference_graph_context_scope import link_scope
B=R/'outputs/data-intake/20261101-PubChem-native-corpus-v1'
M=B/'materialized/independent-final-v1/master-corpus-manifest-v21-PubChem-whole-native-train-v1.json'
F=B/'materialized/independent-final-v1/receipt.json'
O=B/'materialized/reference-graph-context-registry-v1'
A=R/'outputs/data-intake/20261101-PubChem-corpus-pilot-v1/cross-source-chemical-evidence-v2/manifest.json'
FAMILIES={'FreeSolv','Tox21-original-source-call-prior','ChEMBL37-AZ-experimental-logD7_4','Rosetta-CDRP-molecule-profile-CellPainting','Rosetta-CDRP-molecule-profile-L1000'}
CAP=16*2**20

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p
 if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('source reference changed: '+str(p))
 return p

def fields(value,prefix):
 if isinstance(value,dict):
  for key,value2 in value.items():
   role=prefix+'.'+key
   if 'smiles' in key.lower() and isinstance(value2,str):yield role,value2
   elif isinstance(value2,(dict,list)):yield from fields(value2,role)
 elif isinstance(value,list):
  for i,value2 in enumerate(value):
   if isinstance(value2,(dict,list)):yield from fields(value2,prefix+'.'+str(i))

def main():
 start=time.perf_counter();plan=json.loads((B/'build-plan.json').read_text());os.sched_setaffinity(0,set(plan.get('CPU_affinity') or json.loads((B/'launch-receipt.json').read_text())['CPU_affinity']));pa.set_cpu_count(2);pa.set_io_thread_count(1)
 guard=BuildGuard(B,census_cap=2*2**30,total_cap=12*2**30,deadline_epoch=plan['deadline_epoch']);guard.check('materialized',reserve=CAP)
 # Only a completed whole-source task proof permits links into the new master.
 proof=json.loads(F.read_text());m=json.loads(M.read_text());assert proof['status']=='verified' and m['verification']==str(F.relative_to(R))
 for receipt in (B/'phase-launches').glob('*.json'):
  for pid in [json.loads(receipt.read_text()).get(k) for k in ('pid','python_worker_pid')]:
   p=Path('/proc')/str(pid)/'cmdline'
   if p.exists() and b'data_intake_full_pubchem_' in p.read_bytes():raise RuntimeError('earlier full-build phase still live')
 c=json.loads((B/'census/manifest.json').read_text());dbp=check(c['graph_index']);db=sqlite3.connect('file:'+str(dbp)+'?mode=ro&immutable=1',uri=True);a=json.loads(A.read_text());old_shards={s['path']:s for s in m['shards']};O.mkdir();(O/'code').mkdir();(O/'code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes());(O/'code/data_intake_reference_graph_context_scope.py').write_bytes((R/'scripts/data_intake_reference_graph_context_scope.py').read_bytes())
 scanned=collections.Counter();with_graph=collections.Counter();quarantined=collections.Counter();links=[];contexts=set();views=set();graphkeys=set();source_refs=[];source_ref_seen=set();projection_quarantines=[]
 for sh in a['source_shard_refs']:
  assert old_shards[sh['path']]==sh
  if sh.get('source_family') not in FAMILIES|{None}:continue
  p=check(sh);pf=pq.ParquetFile(p);families=set(pf.read(columns=['source_family'])['source_family'].to_pylist())
  if not families&FAMILIES:continue
  source_refs.append(sh);source_ref_seen.add(sh['path']);columns=[k for k in ['task_id','record_id','source_family','split','native_payload','native_payload_sha256','condition_context_json'] if k in pf.schema_arrow.names];rows=pf.read(columns=columns).to_pylist()
  for local,row in enumerate(rows):
   family=row['source_family']
   if family not in FAMILIES:continue
   scanned[family]+=1
   if row['split']!='train':continue
   payload=row['native_payload'];assert hashlib.sha256(payload).hexdigest()==row['native_payload_sha256'];d=json.loads(payload);found=list(fields(d,'payload'))
   if found:with_graph[family]+=1
   roles_seen=set()
   for role,smi in found:
    # Roles retain their source location; audit/target structures are never predictor conditions.
    native_role=role.removeprefix('payload.')
    try:pgraph=project_source_graph(smi)
    except Exception as exc:
     quarantined[family]+=1;projection_quarantines.append(dict(source_family=family,record_id=row['record_id'],task_id=row['task_id'],field_role=native_role,source_original_value=smi,reason='source field cannot form verified exact reference graph: '+repr(exc),source_ref=dict(path=sh['path'],sha256=sh['sha256'],row_index=local),original_task_unchanged=True));continue
    if not pgraph['exact_projection_eligible']:
     quarantined[family]+=1;projection_quarantines.append(dict(source_family=family,record_id=row['record_id'],task_id=row['task_id'],field_role=native_role,source_original_value=smi,reason=pgraph['match_scope'],parser_messages=pgraph['parser_messages'],source_ref=dict(path=sh['path'],sha256=sh['sha256'],row_index=local),original_task_unchanged=True));continue
    for component,ident in pgraph['graphs']:
     policy=link_scope(exact=pgraph['exact_projection_eligible'],component=component,split=row['split'],field_role=native_role)
     if not policy['admitted_reference_graph_link']:continue
     iso=hashlib.sha256(ident['canonical_isomeric_smiles'].encode()).digest();key=(iso,native_role)
     if key in roles_seen:continue
     roles_seen.add(key);match=db.execute('SELECT prior_row,codec_row,pilot_prior,pilot_codec FROM graphs WHERE iso=?',(iso,)).fetchone()
     if match is None:continue
     prior,codec,pilotprior,pilotcodec=match;rid='HF:sagawa/pubchem-10m-canonicalized@f83219601635a0a80fc99c13a9ca37f99ef34f0a:train:'+str(prior)
     links.append(dict(reference_isomeric_graph_sha256=iso.hex(),canonical_isomeric_reference_graph=ident['canonical_isomeric_smiles'],PubChem_reference_record_id=rid,PubChem_reference_prior_task_id=task_identifier('PubChem-native-SMILES-reference-prior',rid,'reference_prior'),PubChem_source_native_train_row=prior,PubChem_codec_source_native_train_row=codec,PubChem_prior_from_preserved_pilot=bool(pilotprior),PubChem_codec_from_preserved_pilot=bool(pilotcodec),existing_source_family=family,existing_record_id=row['record_id'],existing_task_id=row['task_id'],source_view_direction=d['task'],source_split='train',source_original_SMILES=smi,source_graph_field_role=native_role,source_native_payload_sha256=row['native_payload_sha256'],source_payload_ref_json=json.dumps(dict(path=sh['path'],sha256=sh['sha256'],row_index=local)),source_context_json=row.get('condition_context_json') or '{}',link_policy_json=json.dumps(policy),source_condition_molecule_available=policy['molecule_is_explicit_condition'],same_cell_or_material_or_dose_time_pair=False,cross_view_attention_allowed=False,new_task_views=0))
     contexts.add((family,row['record_id']));views.add(row['task_id']);graphkeys.add(iso)
  guard.check('materialized')
  if len(links)>50000:raise RuntimeError('bounded 50k link ceiling reached; preserve partial code/receipts, do not truncate')
 if links:
  p=O/'existing-view-reference-graph-links.parquet';table=pa.Table.from_pylist(links);pq.write_table(table,p,compression='zstd');assert pq.read_table(p).to_pylist()==links;linkref=ref(p,rows=len(links))
 else:linkref=None
 immutable_json(O/'projection-quarantines.json',projection_quarantines)
 manifest=dict(schema='mora3.ReferenceGraphExistingContextRegistry/v2',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),master=ref(M),full_source_task_proof=ref(F),completed_graph_census=ref(B/'census/manifest.json'),source_task_shards=source_refs,source_family_scope=sorted(FAMILIES),source_task_views_scanned=dict(scanned),source_train_views_with_graph_fields=dict(with_graph),nonexact_source_graph_field_roles_excluded=dict(quarantined),source_graph_role_quarantine_refs=ref(O/'projection-quarantines.json'),exact_reference_graphs=len(graphkeys),distinct_original_source_context_records=len(contexts),existing_task_views=len(views),field_role_links=len(links),links=linkref,source_payload_and_reference_graph_keys_roundtrip=True,no_new_task_views_or_experiments=True,no_same_cell_or_material_lot_or_dose_time_or_replicate_pair=True,graph_hashes_and_source_target_structures_audit_only=True,cross_view_attention_allowed=False,allowed_use='audit/sampling candidate only; each existing task retains its own inputs/targets/masks and is separately encoded; hidden structure targets cannot enter another view through a shared graph or cross-sample attention',source_images_masks_or_spatial_entities_added=0,actual_optimizer_use=None,actual_model_token_totals=None,seconds=time.perf_counter()-start,cap_bytes=CAP)
 immutable_json(O/'manifest.json',manifest);usage=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());assert usage<CAP;guard.check('materialized');immutable_json(O/'independent-count-and-size-receipt.json',dict(master=ref(M),registry=ref(O/'manifest.json'),links=linkref,all_registry_bytes_before_this_receipt=usage,cap_bytes=CAP,source_context_count_unit='original context identifiers, not proof of independent experiments',all_source_shard_refs_hash_checked=True,new_task_views=0));assert sum(p.stat().st_size for p in O.rglob('*') if p.is_file())<CAP;guard.check('materialized');print('COMPLETE_REFERENCE_CONTEXT_REGISTRY',json.dumps({k:manifest[k] for k in ['exact_reference_graphs','distinct_original_source_context_records','existing_task_views','field_role_links','seconds']}),flush=True)

if __name__=='__main__':main()
