"""Full relevant admitted chemical-task scan; identity evidence never authorizes assay joins."""
from pathlib import Path
import sys,json,hashlib,collections,datetime,time,functools,shutil
import pyarrow as pa,pyarrow.parquet as pq
from rdkit import Chem,rdBase
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import chemical_identity
from data_intake_chemical_projection_scope import project_source_graph
B=R/'outputs/data-intake/20261101-PubChem-corpus-pilot-v1';O=B/'cross-source-chemical-evidence-v2';M=B/'independent-master-union-v20/master-corpus-manifest-v20-PubChem-native-codec-pilot-v1.json'
FAMILIES={'FreeSolv','ReactZyme','RGD1-computed-energy','BioGRID','Rosetta-CDRP-molecule-profile-CellPainting','Rosetta-CDRP-molecule-profile-L1000','Tox21-original-source-call-prior','NCI-AIDS-source-conclusion-prior','NCI-AIDS-source-endpoint-prior','NCI-AIDS-paired-assay-profile-prior','ChEMBL37-AZ-experimental-logD7_4','Reactome98-static-pathway-prior'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=R/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p
def write(p,x):p.open('x').write(json.dumps(x,indent=2)+'\n')
def fields(value,path):
 if isinstance(value,dict):
  for key,v in value.items():
   if 'smiles' in key.lower() and isinstance(v,str):yield path+'.'+key,v
   elif isinstance(v,(dict,list)):yield from fields(v,path+'.'+key)
 elif isinstance(value,list):
  for i,v in enumerate(value):
   if isinstance(v,(dict,list)):yield from fields(v,path+f'[{i}]')
@functools.lru_cache(maxsize=250000)
def graphs(smiles,unknown):return project_source_graph(smiles,source_unknown_stereo=unknown)
def main():
 start=time.perf_counter();O.mkdir();(O/'code').mkdir();(O/'code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes());m=json.loads(M.read_text());assert sha(M)=='fc47edd361ce3a5dc7088995670323f0cd87803d6b97184fbb9703dcde4b2d2f';p=json.loads((B/'manifest.json').read_text());ledger=pq.read_table(check(p['selected_source_ledger'])).to_pylist();query={};iso_to_queries=collections.defaultdict(set);coarse_to_queries=collections.defaultdict(set);admitted=set()
 for row in ledger:
  if not row['independently_validated_SMILES']:continue
  native=json.loads(row['all_native_source_columns_json']);ident=json.loads(row['derived_identity_json']);rid=row['source_record_id'];key=hashlib.sha256(ident['canonical_isomeric_smiles'].encode()).hexdigest();query[rid]=dict(canonical_isomeric_graph_sha256=key,molecule_group=ident['molecule_group']);iso_to_queries[key].add(rid);coarse_to_queries[ident['molecule_group']].add(rid)
  if json.loads(row['admitted_task_ids_json']):admitted.add(rid)
 edges={};failures=[];stereo_quarantines={};source_rows=collections.Counter();rows_with_structure=collections.Counter();structures=collections.Counter();no_graphs=collections.defaultdict(set);scanrefs=[];old_heldout_exact_admitted=set();old_heldout_coarse_admitted=set();old_heldout_component_exact_admitted=set()
 for i,sh in enumerate(m['shards']):
  if sh.get('source_family') not in FAMILIES|{None}:continue
  path=check(sh);scanrefs.append(sh);rows=pq.read_table(path,columns=['task_id','record_id','source_family','split','native_schema','native_payload','native_payload_sha256']).to_pylist()
  for local,row in enumerate(rows):
   family=row['source_family']
   if family not in FAMILIES:continue
   source_rows[family]+=1
   if row['native_schema']!='mora3.CorpusTask/v1':continue
   assert hashlib.sha256(row['native_payload']).hexdigest()==row['native_payload_sha256'];d=json.loads(row['native_payload']);found=list(fields(d['conditions'],'conditions'))+list(fields(d['targets'],'targets'))
   if not found:no_graphs[family].add(row['record_id']);continue
   rows_with_structure[family]+=1;unknown_stereo=family.startswith('NCI-')
   for role,smi in found:
    structures[family]+=1
    try:projection=graphs(smi,unknown_stereo);derived=projection['graphs']
    except Exception as exc:failures.append(dict(source_family=family,record_id=row['record_id'],task_id=row['task_id'],chemical_role=role,exception=repr(exc),task_payload_unchanged=True));continue
    if projection['lossy_stereo_warning']:stereo_quarantines[(family,row['record_id'],role)]=dict(source_family=family,record_id=row['record_id'],task_id=row['task_id'],field_role=role,original_source_smiles=smi,parser_messages_json=json.dumps(projection['parser_messages']),exact_projection_eligible=False,source_payload_ref_json=json.dumps(dict(path=sh['path'],sha256=sh['sha256'],row_index=local)),original_task_unchanged=True)
    for component_role,ident in derived:
     iso=hashlib.sha256(ident['canonical_isomeric_smiles'].encode()).hexdigest();iso_queries=iso_to_queries.get(iso,set());coarse_queries=coarse_to_queries.get(ident['molecule_group'],set())
     if not iso_queries and not coarse_queries:continue
     # Equality of native graph representation cannot resolve unknown source stereo/lot.
     match_scope=projection['match_scope']
     edgekey=(family,row['record_id'],role,component_role,iso)
     if edgekey in edges:continue
     exact=sorted(iso_queries);coarse=sorted(coarse_queries);admitted_iso=sorted(admitted&iso_queries);admitted_coarse=sorted(admitted&coarse_queries)
     edges[edgekey]=dict(source_family=family,source_record_id=row['record_id'],source_task_id=row['task_id'],source_split=row['split'],chemical_field_role=role,component_scope=component_role,source_isomeric_graph_sha256=iso,source_canonical_isomeric_smiles=ident['canonical_isomeric_smiles'],source_molecule_group=ident['molecule_group'],source_stereochemical_identity_complete=False if not projection['exact_projection_eligible'] else None,source_original_smiles=smi,parser_messages_json=json.dumps(projection['parser_messages']),exact_isomeric_projection_eligible=projection['exact_projection_eligible'],identity_evidence_scope=match_scope,PubChem_exact_representation_original_refs_json=json.dumps(exact),PubChem_coarse_connectivity_original_refs_json=json.dumps(coarse),PubChem_admitted_exact_original_refs_json=json.dumps(admitted_iso),PubChem_admitted_coarse_original_refs_json=json.dumps(admitted_coarse),source_payload_ref_json=json.dumps(dict(path=sh['path'],sha256=sh['sha256'],row_index=local,native_payload_sha256=row['native_payload_sha256'])),source_original_assay_join_authorized=False,new_labels_or_experiments_created=False)
     if row['split'] in ['validation','test']:
      old_heldout_coarse_admitted.update(admitted_coarse)
      if component_role=='whole_source_graph' and projection['exact_projection_eligible']:old_heldout_exact_admitted.update(admitted_iso)
      elif component_role!='whole_source_graph':old_heldout_component_exact_admitted.update(admitted_iso)
  if i%250==0:print('PUBCHEM_EXACT_CROSSSOURCE',i+1,'rows',sum(source_rows.values()),'edges',len(edges),'seconds',round(time.perf_counter()-start,2),flush=True)
 sp=O/'lossy-source-stereo-projection-scopes.parquet';sr=list(stereo_quarantines.values());pq.write_table(pa.Table.from_pylist(sr),sp,compression='zstd');assert pq.read_table(sp).to_pylist()==sr
 out=list(edges.values());path=O/'source-graph-provenance-overlap-edges.parquet';pq.write_table(pa.Table.from_pylist(out),path,compression='zstd');assert pq.read_table(path).to_pylist()==out;fail=O/'derived-group-exceptions.json';write(fail,dict(rows=failures,policy='sourcepayloadsunchanged;derivedauditfailureisnotchemicalrepair_or_newadmission'))
 stats={}
 for family in sorted(FAMILIES):
  es=[e for e in out if e['source_family']==family];exact={q for e in es if e['component_scope']=='whole_source_graph' and e['identity_evidence_scope'].startswith('literal_isomeric') for q in json.loads(e['PubChem_exact_representation_original_refs_json'])};coarse={q for e in es for q in json.loads(e['PubChem_coarse_connectivity_original_refs_json'])};stats[family]=dict(scanned_task_views=source_rows[family],task_views_with_graph=rows_with_structure[family],source_graph_field_views=structures[family],distinct_source_record_graph_roles_matched=len(es),PubChem_original_rows_with_literal_isomeric_whole_graph_guard=len(exact),PubChem_admitted_rows_with_literal_isomeric_whole_graph_guard=len(exact&admitted),PubChem_original_rows_with_coarse_or_component_guard=len(coarse),PubChem_admitted_rows_with_coarse_or_component_guard=len(coarse&admitted),source_records_with_no_native_graph=len(no_graphs[family]),source_unknown_stereo=family.startswith('NCI-'),lossy_source_stereo_graph_roles=sum(q['source_family']==family for q in stereo_quarantines.values()))
 result=dict(schema='mora3.PubChemCrossSourceChemicalEvidenceAudit/v2',status='complete_readonly_identity_guard_audit',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),master=ref(M),component=ref(B/'manifest.json'),selected_valid_source_occurrences=len(query),admitted_original_source_graph_rows=len(admitted),whole_admitted_chemical_task_families_scanned=stats,source_shard_refs=scanrefs,matched_edges=ref(path,rows=len(out)),lossy_source_stereo_scope_rows=ref(sp,rows=len(sr)),strict_lossy_parser_messages_not_promoted_to_exact_identity=True,prior_v1_audit_retained_not_source_material_identity=True,derived_exceptions=ref(fail),admitted_newtrain_exact_whole_graph_existing_valtest_original_rows=len(old_heldout_exact_admitted),admitted_newtrain_coarse_or_component_existing_valtest_original_rows=len(old_heldout_coarse_admitted),admitted_newtrain_component_exact_existing_valtest_original_rows=len(old_heldout_component_exact_admitted),source_algorithm_or_stereochemical_graph_match_not_assay_or_samecell_join=True,NCI_unknown_stereo_never_exact_material_identity=True,BioGRID_without_native_graph_not_joined_by_name_InChIKey_or_gene=True,graph_stereo_isotope_charge_not_removed_from_exact_comparison=True,nonstereo_group_only_leakage_guard_not_material_identity=True,source_derived_atommap_or_redundantH_normalization_does_not_mutate_originals=True,no_new_admission_or_weights_or_splits=True,seconds=time.perf_counter()-start)
 write(O/'manifest.json',result);n=sum(p.stat().st_size for p in B.rglob('*') if p.is_file());assert n<512*2**20 and shutil.disk_usage(R).free>200*2**30;print('COMPLETE',json.dumps({k:result[k] for k in ['selected_valid_source_occurrences','admitted_newtrain_exact_whole_graph_existing_valtest_original_rows','admitted_newtrain_coarse_or_component_existing_valtest_original_rows','admitted_newtrain_component_exact_existing_valtest_original_rows','seconds']}),flush=True)
if __name__=='__main__':main()
