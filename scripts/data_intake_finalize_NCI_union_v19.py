"""Independent original NCI byte/schema/task/chemical-group audit and immutable master v19."""
from pathlib import Path
import sys,json,csv,hashlib,collections,datetime,time,subprocess,functools,shutil,re
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import CorpusTask,chemical_identity,connected_groups,group_split
from data_intake_admit_uniprot_annotations import resolved_shard_rows
from data_intake_corpus_audit_metadata import adapter_support
B=R/'outputs/data-intake/20261101-NCI-AIDS-original-v1';O=B/'independent-master-union-v19';M=R/'outputs/data-intake/20261101-RNAcentral-pilot-union-v1/master-corpus-manifest-v18-RNAcentral-native-pilot-v1.json';P=B/'admitted-source-tasks-v1/manifest.json'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  while chunk:=stream.read(4*2**20):h.update(chunk)
 return h.hexdigest()
def ref(path,**kw):return dict(path=str(path.relative_to(R)),bytes=path.stat().st_size,sha256=sha(path),**kw)
def check(record):
 path=R/record['path'];assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256'];return path
@functools.lru_cache(maxsize=3)
def table_rows(path,digest):
 p=R/path;assert sha(p)==digest;return pq.read_table(p).to_pylist()

def main():
 start=time.perf_counter();O.mkdir();(O/'code').mkdir();(O/'code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes());m=json.loads(M.read_text());master_hash=sha(M);p=json.loads(P.read_text());cpath=check(p['curated_original']);c=json.loads(cpath.read_text());acq=json.loads((B/'manifest.json').read_text())
 for asset in acq['assets']:check(asset['raw'])
 # Independent entire compressed .Z -> decoded byte conservation (UNIX compress
 # has no gzip CRC; do not claim a checksum that the source format does not have).
 decoded=check(c['source_native_SDF']);proc=subprocess.Popen(['gzip','-dc',str(B/'raw/aido99sd.bin')],stdout=subprocess.PIPE,stderr=subprocess.PIPE);hasher=hashlib.sha256();nbytes=0
 while chunk:=proc.stdout.read(4*2**20):hasher.update(chunk);nbytes+=len(chunk)
 assert proc.wait()==0 and nbytes==decoded.stat().st_size and hasher.hexdigest()==c['source_native_SDF']['sha256']
 source_structures={};structure_rows=0;valid_graphs=0;quarantined_graphs=0;native_offsets=0
 with decoded.open('rb') as raw:
  for file in c['structure_files']:
   rows=pq.read_table(check(file)).to_pylist();assert len(rows)==file['rows']
   for i,row in enumerate(rows):
    assert row['native_byte_offset']==native_offsets;block=raw.read(row['native_record_bytes']);native_offsets+=len(block);assert block==row['all_native_record_bytes'] and hashlib.sha256(block).hexdigest()==row['native_record_sha256'] and row['source_record_index']==structure_rows and block.rstrip().endswith(b'$$$$');text=block.decode('ascii');lines=text.splitlines();match=re.search(r'>\s*<NSC>\s*\r?\n([^\r\n]+)',text);assert match and match[1].strip()==row['source_NSC'];assert row['source_NSC'] not in source_structures;source_structures[row['source_NSC']]=dict(ref=dict(path=file['path'],sha256=file['sha256'],row_index=i),row=row)
    if row['graph_projection_status']=='graph_projection_pass':
     projected=json.loads(row['projection_json']);na,nb=int(lines[3][:3]),int(lines[3][3:6]);assert projected['native_atom_count']==na and projected['native_bond_count']==nb and projected['native_atom_lines']==lines[4:4+na] and projected['native_bond_lines']==lines[4+na:4+na+nb] and projected['native_record_sha256']==row['native_record_sha256'];assert projected['source_stereochemistry_known'] is False and projected['derived_graph_coordinates_used'] is False and projected['source_geometry_role']=='source_computed_Corina_no_optimization_not_observed';valid_graphs+=1
    else:assert row['graph_projection_error'] is not None;quarantined_graphs+=1
    structure_rows+=1
  assert raw.read()==b''
 assert structure_rows==42687 and valid_graphs==42076 and quarantined_graphs==611 and native_offsets==nbytes
 # All original padded source CSV dictionaries, not merely selected columns.
 native_tables={};native_projection={};source_qc=collections.Counter();diagnostics=collections.Counter()
 for filename,record in [('aids_conc_may04.txt',c['class_file']),('aids_ec50_may04.txt',c['endpoint_files'][0]),('aids_ic50_may04.txt',c['endpoint_files'][1])]:
  rows=pq.read_table(check(record)).to_pylist();native_tables[record['path']]=rows
  with (B/'raw'/filename).open() as stream:original=list(csv.DictReader(stream))
  assert len(rows)==len(original)==record['rows']
  for i,(row,orig) in enumerate(zip(rows,original)):
   assert row['source_row_index']==i and json.loads(row['source_native_fields_json'])==orig and row['source_NSC']==orig['NSC'].strip();clean={k.strip():v.strip() for k,v in orig.items()}
   if filename=='aids_conc_may04.txt':assert row['source_conclusion']==clean['Conclusion'] and row['secondary_binary_projection_CA_CM_positive']==int(clean['Conclusion'] in ['CA','CM']);source_qc['source_conclusion_records']+=1
   else:
    d=json.loads(row['projection_json']);endpoint='EC50' if 'ec50' in filename else 'IC50';assert d['source_value']==float(clean['Log10'+endpoint]) and d['source_ConcUnit']==clean['ConcUnit'] and d['source_flag']==clean['Flag'];point=clean['Flag']=='=' and clean['ConcUnit'] in ['M','u'];assert d['point_target_available']==point;assert d['source_log10_highest_concentration']==float(clean['Log10HiConc']) and d['source_NumExp']==int(clean['NumExp']) and d['source_stddev']==float(clean['StdDev']);native_projection[row['source_record_id']]=d;source_qc[endpoint+'_source_records']+=1;source_qc[endpoint+'_source_point_masks']+=point
    if point and d['source_value']>d['source_log10_highest_concentration']:diagnostics[endpoint+'_reported_mean_above_reported_highest_concentration']+=1;assert d['source_value']-d['source_log10_highest_concentration']<=0.020000001
    if d['source_stddev']==0:assert not d['positive_uncertainty_target_available']
 contexts=pq.read_table(check(c['context_links'])).to_pylist();native_tables[c['context_links']['path']]=contexts
 # Source reference lookup is keyed by hash and literal row index, not names.
 def outcome(r):
  rows=native_tables.get(r['path'])
  if rows is None:rows=table_rows(r['path'],r['sha256'])
  else:
   expected=next(x for x in [c['class_file'],*c['endpoint_files'],c['context_links']] if x['path']==r['path']);assert r['sha256']==expected['sha256']
  return rows[r['row_index']]
 def expected_view(d,family,sr,direction):
  nsc=sr['source_NSC'];graph=source_structures[nsc];assert sr['structure']==dict(graph['ref'],native_record_sha256=graph['row']['native_record_sha256']);assert graph['row']['graph_projection_status']=='graph_projection_pass';smi=json.loads(graph['row']['projection_json'])['derived_nonstereo_connectivity_smiles'];base=dict(assay_panel='NCI_public_AIDS_screen_source_protocol',source_stereochemistry_known=False)
  if family=='NCI-AIDS-source-conclusion-prior':
   original=outcome(sr['outcome']);assert original['source_NSC']==nsc;label=original['source_conclusion'];expected_conditions=dict(base,molecule_source_connectivity_smiles=smi) if direction=='prediction' else dict(base,desired_source_conclusion=label);expected_targets=dict(source_conclusion=label) if direction=='prediction' else dict(molecule_source_connectivity_smiles=smi)
  elif family=='NCI-AIDS-source-endpoint-prior':
   original=outcome(sr['outcome']);assert original['source_NSC']==nsc;point=native_projection[original['source_record_id']];assert point['point_target_available'];readout='protection_of_infected_cells' if point['endpoint']=='EC50' else 'growth_inhibition_of_uninfected_cells';base.update(assay_readout=readout,concentration_unit=point['source_ConcUnit'],source_log10_highest_tested_concentration=point['source_log10_highest_concentration']);expected_conditions=dict(base,molecule_source_connectivity_smiles=smi) if direction=='prediction' else dict(base,desired_source_log10_endpoint_mean=point['source_value']);expected_targets=dict(source_log10_endpoint_mean=point['source_value']) if direction=='prediction' else dict(molecule_source_connectivity_smiles=smi)
  else:
   assert family=='NCI-AIDS-paired-assay-profile-prior';source=[outcome(r) for r in sr['outcomes']];ec,ic=[native_projection[r['source_record_id']] for r in source];assert [r['source_NSC'] for r in source]==[nsc,nsc] and ec['endpoint']=='EC50' and ic['endpoint']=='IC50' and ec['point_target_available'] and ic['point_target_available'] and ec['compound_highestdose_context_key']==ic['compound_highestdose_context_key'];link=outcome(sr['context_link']);assert link['exact_one_per_endpoint_and_both_source_point_masks'] and link['source_NSC']==nsc and link['same_cell_or_replicate_link'] is False;values=[ec['source_value'],ic['source_value']];base.update(endpoint_axis=['infected_cell_protection_EC50','uninfected_cell_growth_IC50'],concentration_unit=ec['source_ConcUnit'],source_log10_highest_tested_concentration=ec['source_log10_highest_concentration']);expected_conditions=dict(base,molecule_source_connectivity_smiles=smi) if direction=='prediction' else dict(base,desired_source_log10_endpoint_means=values);expected_targets=dict(source_log10_endpoint_means=values) if direction=='prediction' else dict(molecule_source_connectivity_smiles=smi);assert d['audit']['same_cell_or_replicate_pair'] is False and d['audit']['selectivity_ratio_or_therapeutic_index_derived'] is False
  assert d['conditions']==expected_conditions and d['targets']==expected_targets;return nsc,smi
 allocation_rows=pq.read_table(check(p['chemical_group_allocation'])).to_pylist();allocation={row['source_NSC']:row for row in allocation_rows};identity_by_NSC={}
 for nsc,row in allocation.items():
  graph=source_structures[nsc]['row'];smi=json.loads(graph['projection_json'])['derived_nonstereo_connectivity_smiles'];ident=chemical_identity(smi);expected={ident['molecule_group'],ident['scaffold_group'],'NCI-source-NSC:'+nsc};assert set(json.loads(row['groups_json']))==expected;identity_by_NSC[nsc]=ident
 recomputed=connected_groups((nsc,json.loads(row['groups_json'])) for nsc,row in allocation.items());assert all(recomputed[nsc]==row['component_id'] for nsc,row in allocation.items());old_chemical_allocations={}
 new_paths={s['path'] for s in p['shards']};assert not new_paths&{s['path'] for s in m['shards']};m['shards']+=p['shards'];m['effective_group_lookup'].update(p['effective_group_lookup']);m['serialized_tasks']+=p['admitted_task_views'];m['total_shard_bytes']+=p['task_shard_bytes']
 for split,count in p['split_counts'].items():m['split_counts'][split]+=count
 m['native_schema_counts']['mora3.CorpusTask/v1']+=p['admitted_task_views'];m['adapter_support_counts']['mora3.CorpusTask/v1;model_adapter_supported=False']+=p['admitted_task_views'];m['serialized_distinct_record_counts_by_source'].update(p['admitted_source_record_counts_by_family'])
 ids=set();splits=collections.Counter();schemas=collections.Counter();adapters=collections.Counter();source_counts=collections.Counter();source_records=collections.defaultdict(set);allgroups=collections.defaultdict(set);new_nsc=collections.defaultdict(set);new_molecules=collections.defaultdict(set);new_families=collections.Counter();new_family_splits=collections.defaultdict(collections.Counter);data_bytes=group_bytes=0;new_payloads=0
 for index,shard in enumerate(m['shards']):
  sp=check(shard);gp=check(m['effective_group_lookup'][shard['path']]);data_bytes+=sp.stat().st_size;group_bytes+=gp.stat().st_size;grows=resolved_shard_rows(sp,gp,shard);is_new=shard['path'] in new_paths;cols=['task_id','record_id','source_family','split','native_schema']
  if 'model_adapter_supported' in pq.ParquetFile(sp).schema_arrow.names:cols.append('model_adapter_supported')
  if is_new:cols+=['native_payload','native_payload_sha256','admitted_for_task','task_name','condition_context_json']
  rows=pq.read_table(sp,columns=cols).to_pylist();assert len(rows)==shard['rows']==len(grows)
  for row,gr in zip(rows,grows):
   tid=row['task_id'];assert tid==gr['task_id'] and tid not in ids;ids.add(tid);split=row['split'];splits[split]+=1;schemas[row['native_schema']]+=1;adapters[row['native_schema']+';model_adapter_supported='+str(adapter_support(row))]+=1;source_counts[row['source_family']]+=1;source_records[row['source_family']].add(row['record_id']);groups=json.loads(gr['effective_groups_json']);allgroups[split].update(groups)
   if not is_new:
    for group in groups:
     if group.startswith(('molecule:','scaffold:')):assert group not in old_chemical_allocations or old_chemical_allocations[group]==split;old_chemical_allocations[group]=split
    continue
   payload=row['native_payload'];assert hashlib.sha256(payload).hexdigest()==row['native_payload_sha256'] and CorpusTask.from_bytes(payload).canonical_bytes()==payload and row['admitted_for_task'] and row['model_adapter_supported'] is False;d=json.loads(payload);assert d['task']==row['task_name'];direction='prediction' if '_prediction_' in d['task'] else 'generation';sr=d['audit']['source_ref'];assert sr['native_source']==p['curated_original'];nsc,smi=expected_view(d,row['source_family'],sr,direction);assert 'NCI-source-NSC:'+nsc in groups and identity_by_NSC[nsc]['molecule_group'] in groups and identity_by_NSC[nsc]['scaffold_group'] in groups and allocation[nsc]['component_id'] in groups and allocation[nsc]['split']==split;prefix={'NCI-AIDS-source-conclusion-prior':'source_NCI_assay_conclusion_','NCI-AIDS-source-endpoint-prior':'source_NCI_assay_endpoint_','NCI-AIDS-paired-assay-profile-prior':'source_NCI_paired_assay_profile_'}[row['source_family']];assert d['task']==prefix+direction+'_prior';assert not any(key in json.loads(row['condition_context_json']) for key in ['NSC','actual_original_source_NSC','chemical_structure_native_character_count','target_length']);new_nsc[split].add(nsc);new_molecules[split].update(g for g in groups if g.startswith('molecule:'));new_families[row['source_family']+':'+direction]+=1;new_family_splits[row['source_family']][split]+=1;new_payloads+=1
  if index%1000==0:print('NCI_UNION',index+1,'tasks',len(ids),'new',new_payloads,'seconds',round(time.perf_counter()-start,2),flush=True)
 for nsc,row in allocation.items():
  prior={old_chemical_allocations[group] for group in json.loads(row['groups_json']) if group in old_chemical_allocations};assert set(json.loads(row['prior_anchor_splits_json']))==prior
  if row['split'] is None:assert len(prior)>1
  else:assert len(prior)<=1 and row['split']==(next(iter(prior)) if prior else group_split(row['component_id'],seed=m['seed']))
 assert len(ids)==m['serialized_tasks'] and dict(splits)==m['split_counts'] and dict(schemas)==m['native_schema_counts'] and dict(adapters)==m['adapter_support_counts'] and data_bytes==m['total_shard_bytes'];assert {k:len(v) for k,v in source_records.items()}==m['serialized_distinct_record_counts_by_source'];assert new_payloads==p['admitted_task_views'] and dict(new_families)==p['task_views_by_family_direction'];intersections={a+'__'+b:len(allgroups[a]&allgroups[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]};assert not any(intersections.values());assert sha(M)==master_hash
 receipt=dict(schema='mora3.NCIOriginalAssayIndependentCorpusVerification/v1',status='verified',prior_master=ref(M),new_component=ref(P),all_original_compressed_to_decoded_bytes_and_SHA_conserved=nbytes,UNIX_compress_has_no_gzip_CRC=True,all_native_SDF_record_bytes_offsets_SHA_and_raw_elements_bond_lines_compared=structure_rows,native_valid_graphs=valid_graphs,native_graph_quarantines_preserved=quarantined_graphs,original_provider_structure_count_claim=42390,actual_native_unique_NSC_and_record_count=42687,provider_claim_difference=297,all_original_csv_fields_whole_rows_against_native_typed=dict(source_qc),numeric_source_reporting_diagnostics=dict(diagnostics),numeric_diagnostics_policy='threeIC50means0.01..0.02logabove_REPORTED_maxdose;compatiblewithreportedmax0.1precision_inferenceonly;valuesunchanged;nohardboundsorroundingrepair',all_new_task_source_condition_target_scope_unit_and_pointmask_bindings=new_payloads,new_task_counts_by_family_direction=dict(new_families),new_split_counts_by_family={k:dict(v) for k,v in new_family_splits.items()},new_distinct_NSCs_by_split={k:len(v) for k,v in new_nsc.items()},new_distinct_nonstereo_molecule_groups_by_split={k:len(v) for k,v in new_molecules.items()},all_master_shard_group_bytes_hashes_and_envelope_metadata_checked=True,serialized_tasks=len(ids),shards=len(m['shards']),data_bytes=data_bytes,effective_group_bytes=group_bytes,split_counts=dict(splits),native_schema_counts=dict(schemas),adapter_support_counts=dict(adapters),source_tasks=dict(source_counts),known_effective_group_split_intersections=intersections,public_within_source_collection_chemical_holdout_not_newstudy_or_blind=True,all_source_experiment_independence_or_actual_optimizer_usage_asserted=False,old_task_payload_semantics='inheritedboundpriorreceipts;completebytesallocationmetadatagroupsrechecked',new_agent_performed_biological_experiments=0,underlying_independent_experiment_count=None,source_historical_experimental_summary_evidence_expanded=True,new_task_views_not_independent_experiments=True,optimizer_steps=0,seconds=time.perf_counter()-start)
 rp=O/'receipt.json';rp.open('x').write(json.dumps(receipt,indent=2)+'\n');m.update(build_id='20261101-original-NCI-assay-corpus-v19',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_manifest_sha256=master_hash,supersedes_for_consumption=str(M.relative_to(R)),source_components_base_manifest=str(M.relative_to(R)),additional_NCI_original_assay_admission=str(P.relative_to(R)),verification=str(rp.relative_to(R)));m['verification_layers'].append(str(rp.relative_to(R)));m['source_curation_manifests'].append(ref(P))
 for family,records_count in p['admitted_source_record_counts_by_family'].items():
  m['source_ledger'].append(dict(source_family=family,inventory_id='INV-035',release='officialCurrentPublicSnapshotSHA;provider1999structure/2004assayclaims;historicbyteequivalenceunproven',rights='NCIpublicnonconfidentialdata_and_NCIcopyrightreusepolicy_withcredit',source_component=ref(P),source_available_counts=p['source_available_counts'],admitted_records=records_count,serialized_tasks=sum(v for k,v in new_families.items() if k.startswith(family+':')),split_counts=dict(new_family_splits[family]),count_unit='nativeNSCconclusion_or_endpointsummary_or_derivedpairedcontext;directionsareviews;sourcefamiliesoverlap',paired_profiles_not_new_measurements=True,independent_experiment_count=None,full_native_labels_and_fields_retained=True,unknown_cell_strain_time_lot_and_stereo_masks=True))
  m['contexts'][family]=dict(rows=sum(v for k,v in new_families.items() if k.startswith(family+':')),source_record_units=records_count,class_axis=['CA','CM','CI'] if 'conclusion' in family else None,endpoint_mean_axis=2 if 'paired' in family else 1,source_concentration_units=['M','u'] if 'conclusion' not in family else None,source_value_scale='log10concentration_nativeheader' if 'conclusion' not in family else 'source_manual_ternary_conclusion',model_token_count=None,no_native_feature_or_source_label_truncation=True)
 m['open_gates'].append('NCIpublicwithincollectionchemicalholdoutsarenotblind/newstudy;exactcell/strain/duration/lot/stereochemistryunknown;all611nativegraphquarantinesand1779missingclassgraphrefsretained;providerstructurecount+297discrepancypreserved;3IC50meansslightlyabovereportedroundedmaxdosevaluesunchanged;nativeNCIreader/heads/mixedweightsunactivated;noCorinaobservedgeometryorclinicalvalidation')
 m['mixture_proposal']['NCI_original_assay_allocation']=dict(status='provisional_unactivated;sourceexperimentalclass/summarypriorswithreferencegraphs;overlapoldMolNetnotduplicated',weights=None,requires='nativegraph/sourceassayadapter;CA/CM/CIandlog10M/u_evidence/missingmasks;publicchemicalholdoutquality_and_study/lotlimitations;nooraclelabels/IDs/targetlength/Corinafeatures')
 target=O/'master-corpus-manifest-v19-original-NCI-assays-v1.json';target.open('x').write(json.dumps(m,indent=2)+'\n');assert sum(f.stat().st_size for f in B.rglob('*') if f.is_file() and 'raw' not in f.parts and 'research' not in f.parts)<512*2**20;assert shutil.disk_usage(R).free>200*2**30;print('COMPLETE',str(target.relative_to(R)),len(ids),len(m['shards']),data_bytes,flush=True)
if __name__=='__main__':main()
