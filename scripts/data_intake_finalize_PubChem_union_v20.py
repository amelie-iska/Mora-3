"""Independent original-row, typed graph/codec, group and master union verification."""
from pathlib import Path
import sys,json,hashlib,collections,datetime,time,shutil,functools,bisect
import pyarrow.parquet as pq,pyarrow.compute as pc
from rdkit import Chem,rdBase
import selfies as sf
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import CorpusTask,canonical,chemical_identity
B=R/'outputs/data-intake/20261101-PubChem-corpus-pilot-v1';O=B/'independent-master-union-v20';P=B/'manifest.json';M=R/'outputs/data-intake/20261101-NCI-AIDS-original-v1/independent-master-union-v19/master-corpus-manifest-v19-original-NCI-assays-v1.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p
 assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p
def write(p,x):p.open('x').write(json.dumps(x,indent=2)+'\n')
@functools.lru_cache(maxsize=2)
def native_rows(path,digest):
 p=R/path;assert sha(p)==digest;return pq.read_table(p).to_pylist()
def guard():
 n=sum(p.stat().st_size for p in B.rglob('*') if p.is_file());assert n<512*2**20;assert shutil.disk_usage(R).free>200*2**30;return n
def main():
 start=time.perf_counter();O.mkdir();(O/'code').mkdir();(O/'code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes());p=json.loads(P.read_text());m=json.loads(M.read_text());assert ref(M)==p['prior_master'];source=json.loads(check(p['source']).read_text());ledger=pq.read_table(check(p['selected_source_ledger'])).to_pylist();assert len(ledger)==p['selected_source_occurrences']<=100000;byid={r['source_record_id']:r for r in ledger};assert len(byid)==len(ledger)
 selected_by_split=collections.defaultdict(dict)
 for row in ledger:selected_by_split[row['source_native_split']][row['source_native_file_row_index']]=row
 raw_seen=0;raw_files={};full_validation=set()
 for ch in source['chunks']:
  q=Path(ch['path']);assert sha(q)==ch['sha256'];chunk=json.loads(q.read_text());asset=chunk['job']['source_asset'];raw_files[asset['source_native_split']]=asset
  if asset['source_native_split']=='validation':
   for file in chunk['payloads']:
    t=pq.read_table(check(file),columns=['source_current_canonical_graph_sha256']);full_validation.update(v for v in t.column(0).to_pylist() if v is not None)
 for split,asset in raw_files.items():
  raw=check(asset);assert sha(raw)==asset['expected_sha256'];items=selected_by_split[split];indices=sorted(items);offset=0
  for batch in pq.ParquetFile(raw).iter_batches(batch_size=65536,columns=['smiles']):
   lo=bisect.bisect_left(indices,offset);hi=bisect.bisect_left(indices,offset+len(batch))
   for index in indices[lo:hi]:
    item=items[index];native=json.loads(item['all_native_source_columns_json']);assert batch.column(0)[index-offset].as_py()==native['native_smiles'];assert native['source_native_split']==split and native['source_native_file_row_index']==index;raw_seen+=1
   offset+=len(batch)
  assert offset==asset['expected_native_rows']
 assert raw_seen==len(ledger)
 validated={};validation_groups=set();native_table_refs={};source_status=collections.Counter();source_parse=0
 for i,row in enumerate(ledger):
  sr=json.loads(row['source_ref_json']);native=native_rows(sr['path'],sr['sha256'])[sr['row_index']];assert native==json.loads(row['all_native_source_columns_json']) and native['source_record_id']==row['source_record_id'];native_table_refs[sr['path']]=sr['sha256'];source_status[native['status']]+=1;source_parse+=native['native_current_RDKit_parse_pass'];assert native['coordinate_geometry_present'] is False
  with rdBase.BlockLogs():mol=Chem.MolFromSmiles(native['native_smiles']) if native['native_smiles'] else None
  assert (mol is not None)==native['native_current_RDKit_parse_pass']
  if mol is None:assert row['independently_validated_SMILES'] is False;continue
  iso=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True);assert iso==native['source_canonical_isomeric_smiles'] and hashlib.sha256(iso.encode()).hexdigest()==native['source_current_canonical_graph_sha256']
  try:ident=chemical_identity(native['native_smiles'])
  except Exception:
   assert row['independently_validated_SMILES'] is False and not json.loads(row['admitted_task_ids_json']);continue
  assert row['independently_validated_SMILES'] and ident==json.loads(row['derived_identity_json']);groups={ident['molecule_group'],ident['scaffold_group'],'chemical_exact_isomeric:'+native['source_current_canonical_graph_sha256']};validated[row['source_record_id']]=dict(native=native,groups=groups)
  if native['source_native_split']=='validation':validation_groups.update(groups)
  assert native['graph_roundtrip_pass']==native['standard_selfies_target_present']==row['independently_validated_SELFIES']
  if native['graph_roundtrip_pass']:
   decoded=sf.decoder(native['standard_selfies'])
   with rdBase.BlockLogs():back=Chem.MolFromSmiles(decoded)
   assert back is not None and Chem.MolToSmiles(back,canonical=True,isomericSmiles=True)==iso
  else:assert native['standard_selfies'] is None
  if i%10000==0:print('PUBCHEM_INDEPENDENT_SELECTED',i+1,'seconds',round(time.perf_counter()-start,2),flush=True)
 assert dict(source_status)==p['selected_status_counts'] and source_parse==p['selected_native_parse_pass'] and len(validated)==p['selected_independent_graph_quality_pass'];qt=pq.read_table(check(p['source_quarantine_index'])).to_pylist();assert len(qt)==843 and sum(q['native_current_RDKit_parse_pass'] for q in qt)==782;assert {q['source_record_id'] for q in qt}<=set(byid)
 # Read all new native payloads and bind source references and typed conditions.
 newids=set();newgroups=set();newcounts=collections.Counter();records=collections.defaultdict(set);allnew={};data_bytes=0
 for sh in p['shards']:
  path=check(sh);data_bytes+=path.stat().st_size;rows=pq.read_table(path).to_pylist();assert len(rows)==sh['rows'];gp=check(p['effective_group_lookup'][sh['path']]);grows=pq.read_table(gp).to_pylist();assert len(grows)==len(rows)
  for row,gr in zip(rows,grows):
   tid=row['task_id'];assert tid==gr['task_id'] and tid not in newids;newids.add(tid);rid=row['record_id'];source_row=byid[rid];native=validated[rid]['native'];groups=validated[rid]['groups'];assert row['split']=='train' and row['admitted_for_task'] and row['model_adapter_supported'] is False and row['native_schema']=='mora3.CorpusTask/v1';assert set(json.loads(gr['effective_groups_json']))==set(json.loads(row['group_ids_json']))==groups;newgroups.update(groups);assert tid in json.loads(source_row['admitted_task_ids_json']);assert not json.loads(source_row['gate_causes_json']);assert native['source_native_split']=='train' and native['source_current_canonical_graph_sha256'] not in full_validation and not groups&validation_groups
   raw=row['native_payload'];assert hashlib.sha256(raw).hexdigest()==row['native_payload_sha256'] and CorpusTask.from_bytes(raw).canonical_bytes()==raw;d=json.loads(raw);assert d['task']==row['task_name'];sr=d['audit']['source_ref'];assert sr['native_source']==p['source'] and sr['source_record_id']==rid and sr['native_record']==json.loads(source_row['source_ref_json']) and sr['source_native_split']=='train';family=row['source_family']
   if family=='PubChem-native-SMILES-reference-prior':direction='reference_prior';expected_conditions={'prior_domain':'molecular_reference_native_SMILES'};expected_targets={'native_smiles':native['native_smiles']};assert d['audit']['evidence_class']=='source_reference_connectivity_not_experimental_response'
   else:
    assert family=='PubChem-standardSELFIES-codec' and native['graph_roundtrip_pass'];direction=d['task'].removeprefix('computed_standard_SELFIES_');assert d['audit']['evidence_class']=='computed_representation_of_source_reference_graph_not_experiment'
    if direction=='smiles_to_selfies':expected_conditions={'representation_task':'standard_SELFIES_codec','native_smiles':native['native_smiles']};expected_targets={'standard_selfies':native['standard_selfies']}
    else:assert direction=='selfies_to_smiles';expected_conditions={'representation_task':'standard_SELFIES_codec','standard_selfies':native['standard_selfies']};expected_targets={'source_canonical_isomeric_smiles':native['source_canonical_isomeric_smiles']}
   assert d['conditions']==expected_conditions and d['targets']==expected_targets;assert tid==hashlib.sha256(canonical([family,rid,direction])).hexdigest();base=CorpusTask.from_bytes(raw).input_bytes();assert CorpusTask(d['task'],d['conditions'],{'mutated_hidden':'X'},{'mutated_ID_and_target_length':999},['changed_missing']).input_bytes()==base;assert not any(k in json.loads(row['condition_context_json']) for k in ['source_record_id','native_length','target_length','source_native_split']);allnew[tid]=(rid,family,direction);newcounts[family+':'+direction]+=1;records[family].add(rid)
 assert dict(newcounts)==p['task_counts'] and len(newids)==p['admitted_task_views'] and data_bytes==p['task_shard_bytes'];assert sum(len(json.loads(r['admitted_task_ids_json'])) for r in ledger)==len(newids)
 # Whole inherited bytes/footer allocations, exact old-ID collisions and heldout groups.
 oldheldout=set();oldchemical={};old_ids_intersections=0;oldrows=oldbytes=oldgroupbytes=0
 for i,sh in enumerate(m['shards']):
  path=check(sh);gp=check(m['effective_group_lookup'][sh['path']]);assert pq.ParquetFile(path).metadata.num_rows==sh['rows'];oldrows+=sh['rows'];oldbytes+=sh['bytes'];oldgroupbytes+=gp.stat().st_size;t=pq.read_table(path,columns=['task_id','split']);old_ids_intersections+=len(set(t['task_id'].to_pylist())&newids);g=pq.read_table(gp,columns=['task_id','effective_groups_json']);assert len(g)==sh['rows'];assert len(set(g['task_id'].to_pylist()))==len(g) and set(t['task_id'].to_pylist())==set(g['task_id'].to_pylist())
  column=g['effective_groups_json'];mask=pc.or_(pc.match_substring(column,'molecule:'),pc.match_substring(column,'scaffold:'))
  if pc.any(mask).as_py():
   splits=dict(zip(t['task_id'].to_pylist(),t['split'].to_pylist()))
   for gr in g.filter(mask).to_pylist():
    split=splits[gr['task_id']]
    for group in json.loads(gr['effective_groups_json']):
     if group.startswith(('molecule:','scaffold:')):assert group not in oldchemical or oldchemical[group]==split;oldchemical[group]=split
     if split in ['validation','test']:oldheldout.add(group)
  if i%1000==0:print('PUBCHEM_INHERITED',i+1,'rows',oldrows,'seconds',round(time.perf_counter()-start,2),flush=True)
 assert oldrows==m['serialized_tasks'] and oldbytes==m['total_shard_bytes'] and old_ids_intersections==0 and not oldheldout&newgroups
 # Independently replay every admission cause, including representation-specific eligibility.
 seen={};gatecounts=collections.Counter();admitted_graphs=unsupported_admitted=0
 for row in ledger:
  rid=row['source_record_id'];native=json.loads(row['all_native_source_columns_json']);v=validated.get(rid);expected=[]
  if v is None:expected.append('native_graph_or_scaffold_independent_derivation_failed')
  if native['source_native_split']!='train':expected.append('native_source_validation_retained_as_audit_candidate_only')
  if v:
   groups=v['groups'];prior={oldchemical[g] for g in groups if g in oldchemical};iso=native['source_current_canonical_graph_sha256']
   if prior-{'train'}:expected.append('authoritative_master_heldout_chemical_or_scaffold')
   if iso in full_validation:expected.append('global_source_native_validation_exact_graph')
   if groups&validation_groups:expected.append('selected_native_validation_chemical_or_scaffold')
   if iso in seen:expected.append('pilot_exact_isomeric_graph_duplicate')
  assert expected==json.loads(row['gate_causes_json']);gatecounts.update(expected)
  ids=json.loads(row['admitted_task_ids_json'])
  if not expected:
   seen[iso]=rid;admitted_graphs+=1;unsupported_admitted+=not native['graph_roundtrip_pass'];assert len(ids)==(3 if native['graph_roundtrip_pass'] else 1)
  else:assert not ids
 assert dict(gatecounts)==p['quarantine_or_exclusion_counts'] and admitted_graphs==p['selected_admitted_counts']['admitted_unique_isomeric_graphs'] and unsupported_admitted==p['selected_admitted_counts']['valid_SMILES_without_SELFIES_admitted']
 receipt=dict(schema='mora3.PubChemOriginalNativeTaskIndependentVerification/v1',status='verified',prior_master=ref(M),new_component=ref(P),all_selected_original_raw_SMILES_by_exact_split_and_row_index=raw_seen,complete_raw_source_LFS_SHA256s_rechecked=True,all_selected_full_native_typed_rows_against_ledger_and_originalrefs=True,selected_independent_native_graph_scaffold_and_supported_SELFIES_roundtrip=len(validated),all843sourceSELFIESquarantines_preserved=True,all782validSMILESunsupportedSF_selected_task_specific=True,validSMILESunsupportedSF_admitted=unsupported_admitted,all_new_condition_target_source_refs_and_mutation_invariance=len(newids),admitted_unique_isomeric_graphs=admitted_graphs,new_task_counts=dict(newcounts),new_task_views=len(newids),new_task_bytes=data_bytes,new_shards=len(p['shards']),new_splits={'train':len(newids)},old_task_ID_intersections=old_ids_intersections,new_train_old_valtest_group_intersections=0,inherited_all_shard_and_group_bytes_SHA_footer_rows_verified=True,inherited_full_semantic_and_old_groupproof='boundv19receipt;notreclaimed_as_newfullrawQC',source_native_validation_all_exact_graph_guards=len(full_validation),no_new_heldouts=True,all_native_random_split_or_broadblindclaims=False,new_agent_performed_experiments=0,optimizer_steps=0,seconds=time.perf_counter()-start)
 write(O/'receipt.json',receipt);m['shards']+=p['shards'];m['effective_group_lookup'].update(p['effective_group_lookup']);m['serialized_tasks']+=len(newids);m['total_shard_bytes']+=data_bytes;m['split_counts']['train']+=len(newids);m['native_schema_counts']['mora3.CorpusTask/v1']+=len(newids);m['adapter_support_counts']['mora3.CorpusTask/v1;model_adapter_supported=False']+=len(newids);m['serialized_distinct_record_counts_by_source'].update({k:len(v) for k,v in records.items()});m.update(build_id='20261101-PubChem-native-codec-pilot-v20',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_manifest_sha256=sha(M),supersedes_for_consumption=str(M.relative_to(R)),source_components_base_manifest=str(M.relative_to(R)),additional_PubChem_native_codec_pilot=str(P.relative_to(R)),verification=str((O/'receipt.json').relative_to(R)));m['verification_layers'].append(str((O/'receipt.json').relative_to(R)));m['source_curation_manifests'].append(ref(P))
 for family in records:
  count=sum(n for k,n in newcounts.items() if k.startswith(family+':'));m['source_ledger'].append(dict(source_family=family,inventory_id='INV-002',release='HF@f83219601635a0a80fc99c13a9ca37f99ef34f0a;PubChembiologicalreleaseunknown',license='Apache-2.0explicitDATA',available_source_occurrences=9999960,selected_source_occurrences=len(ledger),admitted_records=len(records[family]),serialized_tasks=count,count_unit='uniqueisomericreferencegraphswithinboundedpilot;codecviewsnotindependentobservations',source_component=ref(P),model_adapter_supported=False,source_splits='native_trainonly;globalnativevalidationexact+oldheldoutchemistry+selectednativevalidationcoarsescaffolds_excluded;nativevalidationauditonly',observed_geometry_or_experimental_response=False));m['contexts'][family]=dict(rows=count,full_native_character_axes=True,model_token_count=None,evidence='sourceconnectivityreference_or_computedstandardSELFIESrepresentation',missing='3D/property/response/dose/time/cellularimage/expression/segmentationunavailable',source_native_random_split='auditonly',model_adapter_supported=False)
 m['open_gates'].append('PubChemboundedpilotonly;full10Mcorpusunlaunched;originalGroupSELFIESnotthisstandardgrammar;sourcebiologicalrelease/CIDsunknown;noassay/image/responsejoins;nativevalidationnotblindscaffoldeval;all782validSMILESunsupportedSFtask-specific;modelreader/objective/fullmixtureweightsremainmodel-owned');m['mixture_proposal']['PubChem_native_codec_pilot']=dict(status='provisionalunactivated',weights=None,requires='nativefullSMILES/standardSELFIESsource-safecodecandpriorreaderwithtargetexclusion;noexperimentalresponseor3Dclaims')
 target=O/'master-corpus-manifest-v20-PubChem-native-codec-pilot-v1.json';write(target,m);bytes_now=guard();write(O/'budget-and-publication-receipt.json',dict(master=ref(target),task_views=m['serialized_tasks'],shards=len(m['shards']),data_bytes=m['total_shard_bytes'],split_counts=m['split_counts'],all_new_pilot_derived_bytes_before_this_small_receipt=bytes_now,cap=512*2**20,full10Mbuild_launched=False));guard();print('COMPLETE',target.relative_to(R),m['serialized_tasks'],len(m['shards']),m['total_shard_bytes'],flush=True)
if __name__=='__main__':main()
