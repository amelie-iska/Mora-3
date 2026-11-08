"""Original NCI assay task shards with explicit unit/point/graph and chemical-group gates."""
from pathlib import Path
import sys,json,hashlib,csv,collections,datetime,time,shutil,os
import pyarrow as pa,pyarrow.parquet as pq,pyarrow.compute as pc
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import CorpusTask,canonical,chemical_identity,connected_groups,group_split
from data_intake_NCI_AIDS_tasks import conclusion_task,endpoint_task,joint_task
from data_intake_NCI_AIDS_native import parse_endpoint
B=R/'outputs/data-intake/20261101-NCI-AIDS-original-v1';O=B/'admitted-source-tasks-v1';M=R/'outputs/data-intake/20261101-RNAcentral-pilot-union-v1/master-corpus-manifest-v18-RNAcentral-native-pilot-v1.json';CAP=512*2**20

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=R/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p

def main():
 start=time.perf_counter();O.mkdir();(O/'shards').mkdir();(O/'groups').mkdir();(O/'code').mkdir()
 for name in [Path(__file__).name,'data_intake_NCI_AIDS_tasks.py','data_intake_NCI_AIDS_native.py']:(O/'code'/name).write_bytes((R/'scripts'/name).read_bytes())
 cp=B/'curated-native-v2/manifest.json';c=json.loads(cp.read_text());m=json.loads(M.read_text());curated_ref=ref(cp);master_ref=ref(M);assert c['native_structure_counts']==dict(graph_projection_pass=42076,source_records=42687,derived_graph_quarantines=611);source_graphs={};graph_failures=[]
 for file in c['structure_files']:
  path=check(file)
  for i,row in enumerate(pq.read_table(path).to_pylist()):
   if row['graph_projection_status']!='graph_projection_pass':continue
   native=json.loads(row['projection_json']);nsc=native['native_NSC'];smi=native['derived_nonstereo_connectivity_smiles'];assert nsc not in source_graphs
   try:identity=chemical_identity(smi)
   except Exception as exc:graph_failures.append(dict(source_NSC=nsc,cause='chemical_group_derivation_failed',error=repr(exc)));continue
   source_graphs[nsc]=dict(smiles=smi,identity=identity,source_record_id=row['source_record_id'],ref=dict(path=file['path'],sha256=file['sha256'],row_index=i,native_record_sha256=row['native_record_sha256']))
 assert len(source_graphs)+len(graph_failures)==42076
 classes=pq.read_table(check(c['class_file'])).to_pylist();class_by_nsc={r['source_NSC']:r for r in classes};original_by_group=collections.defaultdict(list)
 for nsc,graph in source_graphs.items():
  if nsc in class_by_nsc:original_by_group[graph['identity']['molecule_group']].append(dict(source_NSC=nsc,source_conclusion=class_by_nsc[nsc]['source_conclusion']))
 raw=R/'outputs/data-intake/20261031-subagent/raw/moleculenet/HIV.csv';assert sha(raw)=='9ffa7fe57dc86c342627ee1d5255e937e2ab812393c73c4d16c697022f6e1d22';overlap=[];overlap_counts=collections.Counter()
 with raw.open() as stream:
  for i,row in enumerate(csv.DictReader(stream)):
   assert row['activity'] in ['CA','CM','CI'] and int(row['HIV_active'])==int(row['activity'] in ['CA','CM']);error=None;identity=None
   try:identity=chemical_identity(row['smiles'])
   except Exception as exc:error=repr(exc)
   linked=original_by_group.get(identity['molecule_group'],[]) if identity else [];labels={r['source_conclusion'] for r in linked};overlap_counts['MolNet_rows']+=1;overlap_counts['MolNet_group_parse_quarantines']+=error is not None;overlap_counts['MolNet_rows_with_original_native_graph_guard']+=bool(linked);overlap_counts['MolNet_rows_with_any_same_class_source_graph']+=row['activity'] in labels;overlap_counts['MolNet_rows_with_same_graph_other_class']+=bool(labels-{row['activity']});overlap.append(dict(MolNet_row=i,MolNet_native_activity=row['activity'],MolNet_binary_label=int(row['HIV_active']),chemical_group=identity['molecule_group'] if identity else None,source_original_NSClabel_candidates_json=json.dumps(linked),same_nonstereo_graph_guard_not_exact_stereo_or_NSClot_identity=True,source_record_correspondence_not_asserted=True,MolNet_training_admitted=False,derivation_error=error))
 p=O/'MolNet-original-NCI-chemical-label-guard-review.parquet';pq.write_table(pa.Table.from_pylist(overlap),p,compression='zstd');assert pq.read_table(p).to_pylist()==overlap;overlap_ref=ref(p,rows=len(overlap));del overlap
 # Chemical groups carry old allocations. Public within-collection chemical holdouts
 # are provisional, never a blind/new-study claim; unknown study/lot IDs stay missing.
 anchors={};anchor_refs=[]
 for index,sh in enumerate(m['shards']):
  gref=m['effective_group_lookup'][sh['path']];gp=check(gref);table=pq.read_table(gp,columns=['task_id','effective_groups_json']);column=table['effective_groups_json'];mask=pc.or_(pc.match_substring(column,'molecule:'),pc.match_substring(column,'scaffold:'))
  if not pc.any(mask).as_py():continue
  meta=pq.read_table(R/sh['path'],columns=['task_id','split']).to_pylist();splits={r['task_id']:r['split'] for r in meta};anchor_refs.append(gref)
  for gr in table.filter(mask).to_pylist():
   split=splits[gr['task_id']]
   for group in json.loads(gr['effective_groups_json']):
    if group.startswith(('molecule:','scaffold:')):
     assert group not in anchors or anchors[group]==split;anchors[group]=split
  if index%1000==0:print('NCI_OLD_GROUPS',index+1,'anchors',len(anchors),'seconds',round(time.perf_counter()-start,2),flush=True)
 groups={nsc:[v['identity']['molecule_group'],v['identity']['scaffold_group'],'NCI-source-NSC:'+nsc] for nsc,v in source_graphs.items()};components=connected_groups(groups.items());component_old=collections.defaultdict(set)
 for nsc,gs in groups.items():component_old[components[nsc]].update(anchors[g] for g in gs if g in anchors)
 allocation={};conflicts=[]
 for nsc,comp in components.items():
  prior=component_old[comp]
  if len(prior)>1:conflicts.append(dict(source_NSC=nsc,component=comp,prior_splits=sorted(prior)));continue
  allocation[nsc]=next(iter(prior)) if prior else group_split(comp,seed=m['seed'])
 allocation_rows=[dict(source_NSC=nsc,component_id=components[nsc],split=allocation.get(nsc),groups_json=json.dumps(groups[nsc]),prior_anchor_splits_json=json.dumps(sorted(component_old[components[nsc]])),source_study_identifier=None,source_collection='NCI_DTP_public_AIDS',group_scope='NSC+literalnonstereochemicalidentity+Murcko/fallbackscaffold;withincollectionchemicalholdout_notnewstudy') for nsc in source_graphs];p=O/'chemical-NSC-component-allocation.parquet';pq.write_table(pa.Table.from_pylist(allocation_rows),p,compression='zstd');assert pq.read_table(p).to_pylist()==allocation_rows;alloc_ref=ref(p,rows=len(allocation_rows))
 (O/'plan.json').open('x').write(json.dumps(dict(schema='mora3.NCIAIDSSourceAdmissionPlan/v1',current_master=master_ref,curated_source=curated_ref,seed=m['seed'],rights='OfficialNCIpublicnonconfidentialsource+NCIcopyrightreusepolicy;sourceconnectiontables/endpointfacts;noCorinaobservedgeometryorlicensepropagation',source_release_claims=c['source_release_dates'],provider_structure_count_discrepancy=297,exact_historical_checksum_equivalence=False,derived_cap_bytes=CAP,source_dataset_geometry_excluded=True,split_policy='oldmolecule/scaffoldallocationspreserved;newconnectedNSC/molecule/scaffoldcomponents80/10/10seededhash;publicwithincollectionchemicalholdout;notblind/newstudy',chemical_anchor_group_count=len(anchors),source_group_conflict_records=len(conflicts),fullsource_scope='allnativevalidclass+eligibleuncensoredphysicalpoint+strictbothpointprofile;missingcontexttaskmaskednotfabricated',mixture_weights=None),indent=2)+'\n')
 shards=[];lookup={};buffers=collections.defaultdict(list);task_counts=collections.Counter();records=collections.defaultdict(set);split_counts=collections.Counter();quarantines=[];ids=set();maskchecks=0;endpoint_cache={};endpoint_ref={};source_point_eligibility=collections.Counter()
 def reserve():assert sum(p.stat().st_size for p in B.rglob('*') if p.is_file() and 'raw' not in p.parts and 'research' not in p.parts)<CAP and shutil.disk_usage(R).free>200*2**30
 def flush(split):
  if not buffers[split]:return
  folder=O/'shards'/split;folder.mkdir(exist_ok=True);path=folder/f'{len(shards):05d}.parquet';rows=buffers[split];pq.write_table(pa.Table.from_pylist(rows),path,compression='zstd');back=pq.read_table(path).to_pylist();assert back==rows
  for row in back:assert hashlib.sha256(row['native_payload']).hexdigest()==row['native_payload_sha256'];assert CorpusTask.from_bytes(row['native_payload']).canonical_bytes()==row['native_payload']
  shards.append(ref(path,rows=len(rows),split=split,source_family=rows[0]['source_family'] if len({r['source_family'] for r in rows})==1 else None));g=O/'groups'/path.name;gr=[dict(task_id=r['task_id'],effective_groups_json=r['group_ids_json']) for r in rows];pq.write_table(pa.Table.from_pylist(gr),g,compression='zstd');assert pq.read_table(g).to_pylist()==gr;lookup[str(path.relative_to(R))]=ref(g,rows=len(gr));buffers[split]=[];reserve()
 def register(nsc,rid,family,task,direction,source_ref,target_dim):
  nonlocal maskchecks
  if nsc not in allocation:quarantines.append(dict(source_record_id=rid,source_NSC=nsc,cause='missingvalidsourcegraph_or_priorgroup_split_conflict',direction=direction));return
  graph=source_graphs[nsc];split=allocation[nsc];payload=task.canonical_bytes();digest=hashlib.sha256(payload).hexdigest();tid=hashlib.sha256(canonical([family,rid,task.canonical_bytes().decode()])).hexdigest();assert tid not in ids;ids.add(tid);task_counts[family+':'+direction]+=1;records[family].add(rid);split_counts[split]+=1;d=json.loads(payload);assert not any(key in d['conditions'] for key in ['NSC','CAS_RN','source_flag','StdDev','NumExp','source_conclusion','source_value','source_native_fields','source_geometry','target_length']);maskchecks+=1
  buffers[split].append(dict(task_id=tid,record_id=rid,source_family=family,source_revision='officialNCIcurrentpublicSHA_snapshot;provider1999/2004releaseclaims_distinct',source_sha256=curated_ref['sha256'],task_name=d['task'],native_schema='mora3.CorpusTask/v1',native_payload=payload,native_payload_sha256=digest,split=split,split_policy='NSC+molecule+scaffoldconnectedgroups;oldallocationsauthoritative;newseeded80/10/10;publicwithincollectionchemicalholdout_notblindstudy',group_ids_json=json.dumps(sorted(set(groups[nsc]+[components[nsc]]))),evidence_class='originalsource_assay_conclusion_or_computed_measurement_summary;referenceconnectivity;notclinicalvalidation',admitted_for_task=True,model_adapter_supported=False,admission_receipt_json=json.dumps(dict(source_ref=source_ref,rights='NCIpublicnonconfidentialdata;NCIcopyrightfreepolicy_withcredit',source_stereochemistry_known=False,source_geometry_notobserved=True,original_structure_count_discrepancy297=True,source_lot_study_time_cell_context_incomplete=True,withincollectionchemicalholdout_only=True,new_independent_experiment=False)),missing_context_json=json.dumps(d['missing']),full_axis_values=len(graph['smiles']) if direction=='generation' else target_dim,condition_context_json=json.dumps(dict(source_collection='NCI_DTP_public_AIDS',source_study_ID=None,unknown_stereochemistry_not_achirality=True,profile_is_not_image=True,source_cell_line=None,source_exposure_duration=None))));
  if len(buffers[split])>=1024:flush(split)
 for i,row in enumerate(classes):
  nsc=row['source_NSC'];graph=source_graphs.get(nsc)
  if graph is None:quarantines.append(dict(source_record_id=row['source_record_id'],source_NSC=nsc,cause='no_valid_groupable_original_source_graph',direction='both'));continue
  sr=dict(native_source=curated_ref,structure=graph['ref'],outcome=dict(path=c['class_file']['path'],sha256=c['class_file']['sha256'],row_index=i),source_NSC=nsc)
  for direction in ['prediction','generation']:register(nsc,row['source_record_id'],'NCI-AIDS-source-conclusion-prior',conclusion_task(graph['smiles'],row['source_conclusion'],direction,sr),direction,sr,1)
 for file in c['endpoint_files']:
  for i,row in enumerate(pq.read_table(check(file)).to_pylist()):
   if row['source_schema_status']!='native_schema_pass':continue
   point=json.loads(row['projection_json']);native=json.loads(row['source_native_fields_json']);assert point==parse_endpoint(native,row['endpoint_kind']);rid=row['source_record_id'];endpoint_cache[rid]=point;endpoint_ref[rid]=dict(path=file['path'],sha256=file['sha256'],row_index=i);source_point_eligibility[row['endpoint_kind']+'_source_point_mask']+=point['point_target_available']
   if not point['point_target_available']:continue
   nsc=row['source_NSC'];graph=source_graphs.get(nsc)
   if graph is None:quarantines.append(dict(source_record_id=rid,source_NSC=nsc,cause='pointmean_has_no_valid_groupable_original_source_graph',direction='both'));continue
   sr=dict(native_source=curated_ref,structure=graph['ref'],outcome=endpoint_ref[rid],source_NSC=nsc)
   for direction in ['prediction','generation']:register(nsc,rid,'NCI-AIDS-source-endpoint-prior',endpoint_task(graph['smiles'],point,direction,sr),direction,sr,1)
 for i,link in enumerate(pq.read_table(check(c['context_links'])).to_pylist()):
  if not link['exact_one_per_endpoint_and_both_source_point_masks']:continue
  nsc=link['source_NSC'];graph=source_graphs.get(nsc)
  if graph is None:continue
  er=json.loads(link['EC50_source_record_ids_json'])[0];ir=json.loads(link['IC50_source_record_ids_json'])[0];ec,ic=endpoint_cache[er],endpoint_cache[ir];assert ec['compound_highestdose_context_key']==ic['compound_highestdose_context_key'];rid='NCI-original-paired-context:'+hashlib.sha256(canonical(ec['compound_highestdose_context_key'])).hexdigest();sr=dict(native_source=curated_ref,structure=graph['ref'],outcomes=[endpoint_ref[er],endpoint_ref[ir]],context_link=dict(path=c['context_links']['path'],sha256=c['context_links']['sha256'],row_index=i),source_NSC=nsc)
  for direction in ['prediction','generation']:register(nsc,rid,'NCI-AIDS-paired-assay-profile-prior',joint_task(graph['smiles'],ec,ic,direction,sr),direction,sr,2)
 for split in list(buffers):flush(split)
 p=O/'task-specific-quarantines.json';p.open('x').write(json.dumps(dict(chemical_group_derivation_failures=graph_failures,connected_prior_split_conflicts=conflicts,task_exclusions=quarantines),indent=2)+'\n');quarantine_ref=ref(p);reserve();result=dict(schema='mora3.NCIAIDSOriginalSourceTaskAdmission/v1',status='scientificallyscoped_source_tasks_serialized;independentmasterunionpending',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_master=master_ref,curated_original=curated_ref,source_available_counts=dict(conclusion_rows=43850,structure_native_records=42687,EC50_summary_rows=39365,IC50_summary_rows=39350),source_integrity_pass_counts=dict(valid_native_derived_graphs=42076,valid_native_endpoint_rows=78715),source_point_masks=dict(source_point_eligibility),admitted_task_views=len(ids),task_views_by_family_direction=dict(task_counts),admitted_source_record_counts_by_family={k:len(v) for k,v in records.items()},split_counts=dict(split_counts),native_schema='mora3.CorpusTask/v1',model_adapter_supported=False,shards=shards,effective_group_lookup=lookup,task_shard_bytes=sum(s['bytes'] for s in shards),quarantines=quarantine_ref,chemical_group_allocation=alloc_ref,previous_chemical_anchor_sidecar_refs=anchor_refs,previous_chemical_anchor_groups=len(anchors),old_allocations_preserved=True,new_allocation_seed=m['seed'],provisional_within_collection_chemical_holdout_not_blind_or_newstudy=True,source_study_and_lot_groups_unavailable_not_fabricated=True,source_corina_geometry_not_observed_and_not_predictor=True,source_unknown_stereochemistry_never_achirality_label=True,all_task_canonical_and_Parquet_roundtrips=True,all_condition_allowlist_checks=maskchecks,source_to_MolNet_chemical_label_review=overlap_ref,source_to_MolNet_overlap_counts=dict(overlap_counts),MolNet_rows_not_separately_admitted=True,all_full_native_features_remain_in_raw_and_curated_refs=True,source_SD_NumExp_flags_not_predictor_features=True,only_source_equal_flag_knownunit_means_point_targets=True,source_aggregate_censor_flags_not_regression_bounds=True,IC50_EC50_ratio_not_created=True,paired_summary_views_not_additional_independent_measurements=True,new_independent_experiments=0,optimizer_steps=0,final_mixture_weights=None,seconds=time.perf_counter()-start)
 (O/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');print('COMPLETE',json.dumps({k:result[k] for k in ['admitted_task_views','task_views_by_family_direction','admitted_source_record_counts_by_family','split_counts','task_shard_bytes','source_to_MolNet_overlap_counts','seconds']}),flush=True)
if __name__=='__main__':main()
