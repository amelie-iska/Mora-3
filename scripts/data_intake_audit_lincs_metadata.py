"""Finite exact-ID/chemical/context audit of completed LINCS metadata only."""
from pathlib import Path
import collections,datetime,hashlib,json,math,os,shutil,sqlite3,sys,time
import pyarrow as pa
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_chemical_projection_scope import project_source_graph
from data_intake_lincs_reference_mapping import exact_perturbation_map
S=R/'outputs/data-sources/20261101-LINCS-Pilot1-source-v1'
H=S/'handoff-two-originals-full-native-verified-v1.json'
O=R/'outputs/data-intake/20261101-LINCS-corpus-v1/metadata-admission-audit-v1'
P=R/'outputs/data-intake/20261101-PubChem-native-corpus-v1/build-plan.json'
CAP=128*2**20

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p
 assert sha(p)==r['sha256'] and ('bytes' not in r or p.stat().st_size==r['bytes']);return p

def write(p,x):p.open('x').write(json.dumps(x,indent=2)+'\n')
def main():
 start=time.perf_counter();plan=json.loads(P.read_text());cores=plan.get('CPU_affinity') or json.loads((P.parent/'launch-receipt.json').read_text())['CPU_affinity'];os.sched_setaffinity(0,{cores[0]});pa.set_cpu_count(1);pa.set_io_thread_count(1);assert shutil.disk_usage(R).free>200*2**30
 assert sha(H)=='e237ccecb269b4561e00e291aaa26fc9870d13c3af805de2e77fb3622964ef17';h=json.loads(H.read_text());assert h['data_license']=='CC0-1.0' and h['source_agent_task_admitted_records']==0
 O.mkdir(parents=True);(O/'code').mkdir();(O/'code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes());(O/'code/data_intake_lincs_reference_mapping.py').write_bytes((R/'scripts/data_intake_lincs_reference_mapping.py').read_bytes())
 pre=json.loads(check(plan['preflight']).read_text());heldout=set(pre['old_heldout_chemical_guards']);frozen=set(pre['frozen_pilot_validation_chemical_guards']);qc={};native={};indexrefs=[]
 for component in h['components']:
  cp=check(component);d=json.loads(cp.read_text());asset=d['asset_id'];qc[asset]=d;check(d['original_payload']);assert d['independent_full_native_and_index_replay_rows']==d['original_payload']['rows'];axisfile=Path(d['full_axis_file']);assert sha(axisfile)==d['full_axis_sha256'];rows=[]
  for ix in d['row_indexes']:
   part=pq.read_table(check(ix)).to_pylist();assert len(part)==ix['rows'] and [r['source_row'] for r in part]==list(range(ix['source_row_start'],ix['source_row_stop']));rows+=part;indexrefs.append(ix)
  assert len(rows)==d['original_payload']['rows'];native[asset]=rows
 ge=native['DSE-LINCS-Pilot1-L1000'];cp=native['DSE-LINCS-Pilot1-CP'];projections={};errors={};graph_by_sha={};mapping_rows=[];direct_counts=collections.Counter()
 for row in ge:
  smi=row.get('Metadata_SMILES');graph=None
  if isinstance(smi,str) and smi:
   if smi not in projections and smi not in errors:
    try:projections[smi]=project_source_graph(smi)
    except Exception as exc:errors[smi]=repr(exc)
   value=projections.get(smi)
   if value is not None and value['exact_projection_eligible']:
    ident=next(ident for role,ident in value['graphs'] if role=='whole_source_graph');graph=hashlib.sha256(ident['canonical_isomeric_smiles'].encode()).hexdigest();graph_by_sha[graph]=dict(identity=ident,projection=value,original_SMILES=smi)
  direct_counts['strict_exact_source_graph' if graph else 'no_strict_exact_source_graph']+=1;mapping_rows.append(dict(pert_id=row['Metadata_pert_id'],source_row=row['source_row'],graph_sha=graph))
 bindings=exact_perturbation_map(mapping_rows);binding_rows=[dict(native_pert_id=k,eligible_reference_graph_sha=v['eligible_reference_graph_sha'],graph_candidates_json=json.dumps(v['graph_candidates']),source_GE_rows_json=json.dumps(v['source_rows']),unresolved_source_GE_rows_json=json.dumps(v['unresolved_source_rows']),reason=v['reason']) for k,v in bindings.items()];bp=O/'exact-native-perturbation-reference-bindings.parquet';pq.write_table(pa.Table.from_pylist(binding_rows),bp,compression='zstd');assert pq.read_table(bp).to_pylist()==binding_rows
 out=[];summary={};reasons=collections.Counter();crossassay_reference_ids=set();source_chemical_sets={};candidate_graphs=set()
 for asset,rows in native.items():
  mod='CellPainting' if asset.endswith('-CP') else 'L1000';expected_time=48 if mod=='CellPainting' else 24;stats=collections.Counter();chemical_ids=set();platewells=set();source_chemical_sets[mod]=chemical_ids
  for row in rows:
   assert row['Metadata_cell_id']=='A549' and row['Metadata_pert_timepoint']==expected_time
   platewell=(row['Metadata_Plate'],row['Metadata_Well']);assert platewell not in platewells;platewells.add(platewell);pid=row['Metadata_pert_id'];binding=bindings.get(pid);graph=binding['eligible_reference_graph_sha'] if binding else None;identity=graph_by_sha[graph]['identity'] if graph else None;groups=set()
   if identity:
    chemical_ids.add(pid);candidate_graphs.add(graph);groups.update(['chemical_exact_isomeric:'+graph,identity['molecule_group'],identity['scaffold_group']]);groups.update(ident[k] for kind,ident in graph_by_sha[graph]['projection']['graphs'] for k in ['molecule_group','scaffold_group'])
   dose=row.get('Metadata_pert_dose_micromolar');dose_known=type(dose) in (int,float) and math.isfinite(dose) and dose>=0;trt=row['Metadata_pert_type']=='trt';dose_conditioned=bool(dose_known and trt and dose>0);conflict=sorted(groups&heldout);freeze_conflict=sorted(groups&frozen)
   chemical_candidate=bool(graph and trt and dose_conditioned and not conflict and not freeze_conflict)
   status='candidate_for_new_task_specific_admission' if chemical_candidate else 'control_source_profile_only' if not trt else 'existing_heldout_chemical_guard' if conflict else 'frozen_pilot_validation_guard' if freeze_conflict else 'missing_or_ambiguous_exact_chemical_reference' if not graph else 'missing_physical_treatment_dose'
   stats['source_profiles']+=1;stats['finite_complete_profiles']+=int(row['finite_complete_profile']);stats['partial_feature_masked_profiles']+=int(not row['finite_complete_profile']);stats['source_treatment_rows']+=int(trt);stats['strict_unique_chemical_reference_rows']+=int(graph is not None);stats['known_nonnegative_source_nominal_dose_rows']+=int(dose_known);stats['positive_source_treatment_dose_rows']+=int(dose_conditioned);stats['candidate_chemical_dose_time_cell_profiles']+=int(chemical_candidate);stats['existing_heldout_chemical_guard_rows']+=int(bool(conflict));stats['frozen_pilot_validation_guard_rows']+=int(bool(freeze_conflict));stats['control_source_rows']+=int(not trt);stats['status:'+status]+=1;reasons[mod+':'+status]+=1
   out.append(dict(source_occurrence_key=row['source_occurrence_key'],source_asset_id=asset,modality=mod,source_native_row=row['source_row'],source_native_metadata_json=json.dumps(row,sort_keys=True),source_native_perturbation_id=pid,chemical_binding_scope='directGEsourceSMILES' if mod=='L1000' else 'same-study exact full pert_id reference-only; not assay pair',unique_exact_source_reference_graph_sha=graph,canonical_isomeric_source_reference_graph=identity['canonical_isomeric_smiles'] if identity else None,source_graph_binding_GE_rows_json=json.dumps(binding['source_rows']) if binding else None,chemical_groups_json=json.dumps(sorted(groups)),source_nominal_dose_micromolar=dose,source_nominal_dose_is_known=dose_known,positive_treatment_dose_condition_mask=dose_conditioned,source_time_hours=expected_time,source_cell_line='A549',source_feature_nulls=row['profile_null_count'],source_full_feature_axis=qc[asset]['full_feature_axis'],source_finite_complete_profile=row['finite_complete_profile'],existing_heldout_chemical_groups_json=json.dumps(conflict),frozen_pilot_validation_groups_json=json.dumps(freeze_conflict),chemical_dose_time_cell_task_candidate=chemical_candidate,candidate_status=status,task_admitted=False,serialized_task_views=0,original_source_feature_values_unmodified=True,raw_image_or_mask_available_for_task=False,crossassay_same_cell_or_matched_replicate_pair=False))
  summary[mod]=dict(stats,source_distinct_plates=len({r['Metadata_Plate'] for r in rows}),source_distinct_native_perturbation_IDs=len({r['Metadata_pert_id'] for r in rows}),strict_unique_reference_perturbation_IDs=len(chemical_ids),native_feature_axis=qc[asset]['full_feature_axis'])
 p=O/'source-metadata-chemical-context-admission-candidates.parquet';pq.write_table(pa.Table.from_pylist(out),p,compression='zstd');assert pq.read_table(p).to_pylist()==out
 write(O/'projection-errors.json',dict(parser_errors=errors,lossy_source_stereo={k:v['parser_messages'] for k,v in projections.items() if not v['exact_projection_eligible']}))
 actual=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());assert actual<CAP;manifest=dict(schema='mora3.LINCSExactNativeContextAdmissionAudit/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_handoff=ref(H),source_metadata_index_refs=indexrefs,source_rights='CC0-1.0',source_profiles=80060,all_native_metadata_rows_replayed=True,raw_source_whole_SHA256s_rechecked=True,exact_perturbation_bindings=ref(bp,rows=len(binding_rows)),bindings_by_disposition=dict(collections.Counter(v['reason'] for v in bindings.values())),strict_GE_direct_source_graph_counts=dict(direct_counts),candidate_rows=ref(p,rows=len(out)),counts=summary,source_distinct_exact_graph_candidates=len(candidate_graphs),exact_native_reference_perturbation_IDs_shared_across_assays=len(source_chemical_sets['CellPainting']&source_chemical_sets['L1000']),same_cell_or_matched_replicate_pairs=0,nominal_time_hours={'CellPainting':48,'L1000':24},source_cell_line='A549',dose_unit_provenance='author curated Metadata_pert_dose_micromolar definition, no calibration or new conversion',profile_null_policy='source partial feature rows preserved; exact feature validity mask required in any later task; no imputation or blanket source discard',all_missing_features_and_metadata_preserved=True,no_new_task_views_or_corpus_admissions=True,model_adapter_supported=False,current_authoritative_master='v20 until active PubChem full proof passes; source admission not optimizer use',source_split=None,split_policy='candidate train-only/exclusion audit, not heldout allocation; existing heldout/frozen guards retained; study/plate generalization unresolved',raw_image_traceability='source owner separate gap audit; no registered image/mask or scale from profile feature names',observed_feature_count_not_model_token_total=True,actual_model_token_totals=None,metadata_audit_cap_bytes=CAP,shared_source_and_corpus_derived_cap_bytes=2*2**30,source_derived_bytes=h['derived_bytes_observed'],constructor_bytes_before_manifest=actual,seconds=time.perf_counter()-start)
 write(O/'manifest.json',manifest);actual=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());assert actual<CAP and actual+h['derived_bytes_observed']<2*2**30;print('COMPLETE_LINCS_METADATA_ADMISSION_AUDIT',json.dumps(dict(counts=summary,bindings=manifest['bindings_by_disposition'],graph_candidates=len(candidate_graphs),constructor_bytes=actual,seconds=manifest['seconds'])),flush=True)

if __name__=='__main__':main()
