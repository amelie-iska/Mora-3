"""Exact source cell-ID lineage and nominal-coordinate evidence; no physical dose imputation."""
from pathlib import Path
import collections,csv,datetime,hashlib,json,math,re,sys,time
import pyarrow as pa
import pyarrow.parquet as pq
R=Path.cwd();sys.path.insert(0,str(R/'scripts'))
from data_intake_sciplex_native_metadata import r_pdata_records,headerless_tab
B=Path('outputs/data-intake/20261101-sciplex-lineage-resolution-v1');O=B/'cell-lineage-review-v1';O.mkdir(exist_ok=False);(O/'code').mkdir();start=time.perf_counter()
for name in ['data_intake_link_sciplex_original_metadata.py','data_intake_sciplex_native_metadata.py']:(O/'code'/name).write_bytes((R/'scripts'/name).read_bytes())
def ref(p,**kw):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),**kw)
def check(e):
 p=Path(e['path']);assert p.stat().st_size==e['bytes'] and ref(p)['sha256']==e['sha256'];return p
def save(name,rows):
 p=O/(name+'.parquet');pq.write_table(pa.Table.from_pylist(rows),p,compression='zstd');assert pq.read_table(p).to_pylist()==rows;return ref(p,rows=len(rows))
try:
 acq=json.loads((B/'manifest.json').read_text());sources={r['name']:r['decoded'] for r in acq['receipts'] if r['status']=='gzipCRC_UTF8_complete_metadata_verified'};base='GSM4150377_sciPlex2';pdref=sources[base+'_pData.txt.gz'];pd=r_pdata_records(check(pdref).read_text());cellref=sources[base+'_A549_Transcription_Modulators_cell.annotations.txt.gz'];cell=headerless_tab(check(cellref).read_text(),2);hashref=sources[base+'_A549_Transcription_Modulators_hashTable.out.txt.gz'];hashrows=headerless_tab(check(hashref).read_text(),5);sheetref=sources[base+'_hashSampleSheet.txt.gz'];sheet=headerless_tab(check(sheetref).read_text(),3);assert len(pd)==len(cell)==24262 and len(sheet)==192 and len(hashrows)==350181
 pd_by_id=collections.defaultdict(list)
 for i,r in enumerate(pd):pd_by_id[r['Cell']].append((i,r))
 assert len(pd_by_id)==24262;assert [(r['Cell'],r['sample']) for r in pd]==[(r[0],r[1]) for r in cell]
 samplelabels=collections.defaultdict(list)
 for i,r in enumerate(sheet):samplelabels[r[0]].append((i,r))
 hashes=collections.defaultdict(list)
 for i,r in enumerate(hashrows):hashes[(r[0],r[1],r[2])].append(i)
 pdtable=save('all-original-pData-records',[dict(source_row_index=i,source_cell_id=r['Cell'],source_library=r['sample'],source_native_fields_json=json.dumps(r,sort_keys=True)) for i,r in enumerate(pd)])
 hstable=save('all-original-hash-UMI-records',[dict(source_row_index=i,source_library=r[0],source_cell_id=r[1],source_sample_hash_label=r[2],native_fields_json=json.dumps(r)) for i,r in enumerate(hashrows)])
 shtable=save('all-original-sample-label-records',[dict(source_row_index=i,source_sample_hash_label=r[0],native_fields_json=json.dumps(r)) for i,r in enumerate(sheet)])
 gpinfo=Path('outputs/implementation/I6-gperturb-sciplex-access/SciPlex2_cell_info.csv');gppert=Path('outputs/implementation/I6-gperturb-sciplex-access/SciPlex2_perturbation.csv')
 with gpinfo.open() as f:gpcells=list(csv.DictReader(f))
 with gppert.open() as f:reader=csv.DictReader(f);pertcols=reader.fieldnames[1:];pert=list(reader)
 assert len(gpcells)==len(pert)==20643 and pertcols==['Dex','Nutlin','BMS','Vehicle','SAHA'];assert len({r[''] for r in gpcells})==20643
 links=[];quarantine=[];stats=collections.Counter();conditions=collections.defaultdict(lambda:dict(cells=0,vectors=collections.Counter(),source_hash_labels=set()));indices=[]
 for i,(gp,v) in enumerate(zip(gpcells,pert)):
  assert v['']==str(i);barcode=gp[''];matches=pd_by_id.get(barcode,[])
  if len(matches)!=1:quarantine.append(dict(processed_cell_index=i,source_cell_id=barcode,cause='original_cell_ID_missing_or_ambiguous',source_candidates=len(matches)));continue
  sourceindex,r=matches[0];indices.append(sourceindex);stats['exact_unique_cell_namespace_matches']+=1;ncount=float(gp['n_counts']);numi=float(r['n.umi']);same=ncount==numi;stats['processed_n_counts_equals_original_n_umi']+=same;label=r['top_oligo'];label_present=label in samplelabels;triples=hashes.get((r['sample'],barcode,label),[]);stats['source_assignment_label_present_in_sample_sheet']+=label_present;stats['source_cell_library_label_has_exact_hash_record']+=bool(triples);stats['source_hash_assignment_qval_one']+=r['qval']=='1';stats['source_hash_assignment_pval_zero']+=r['pval']=='0';nativevec=[float(v[k]) for k in pertcols];assert all(math.isfinite(x) and x>=0 for x in nativevec)
  m=re.fullmatch(r'(BMS|Dex|Nutlin|SAHA)_([0-9]+(?:[.][0-9]+)?)_([^_]+)',label);alias=m[1] if m else None;dose=float(m[2]) if m else None;welllabel=m[3] if m else None;expected=[0.0]*5
  if m:
   if dose==0:expected[pertcols.index('Vehicle')]=1.0
   else:expected[pertcols.index(alias)]=dose/100.0
  coordinate_match=m is not None and all(abs(a-b)<1e-12 for a,b in zip(expected,nativevec));stats['nominal_label_to_processed_vector_scale100_hypothesis_matches']+=coordinate_match
  key=(alias,str(dose));conditions[key]['cells']+=1;conditions[key]['vectors'][json.dumps(nativevec)]+=1;conditions[key]['source_hash_labels'].add(label)
  source_groups=['study:GSE139944','sample:GSM4150377','original_GEO_cell:GSM4150377:'+r['sample']+':'+barcode]
  if label_present:source_groups.append('original_GEO_hash_sample:GSM4150377:'+label)
  links.append(dict(processed_cell_index=i,processed_cell_info_id=barcode,processed_cell_info_payload=str(gpinfo),processed_perturbation_payload=str(gppert),original_source='GSE139944/GSM4150377',original_source_library=r['sample'],original_source_cell_ID=barcode,original_pData_row_index=sourceindex,original_cell_annotations_row_index=sourceindex,original_source_sample_hash_label=label,original_hash_source_row_indices_json=json.dumps(triples),source_sample_sheet_row_indices_json=json.dumps([x[0] for x in samplelabels.get(label,[])]),source_nominal_treatment_alias=alias,source_nominal_dose_label=dose,source_nominal_well_label=welllabel,source_physical_dose=None,source_physical_dose_unit=None,source_nominal_dose_is_physical=False,processed_native_perturbation_axis_json=json.dumps(pertcols),processed_native_perturbation_vector_json=json.dumps(nativevec),nominal_label_coordinate_scale100_match=coordinate_match,processed_n_counts=ncount,original_n_umi=numi,original_total_UMI_matches_processed_metadata=same,source_hash_assignment_evidence='source_reported_algorithmic_cell_to_scheduled_sample_hash_assignment;observedhashUMIsandexperimentalwellsarenotdirectpercelldosingmeasurements',source_hash_assignment_qval=r['qval'],source_hash_assignment_pval=r['pval'],source_hash_assignment_ratio=r['top_to_second_best_ratio'],source_quality_threshold_inferred=False,source_groups_json=json.dumps(source_groups),original_native_metadata_json=json.dumps(r,sort_keys=True),original_gene_expression_values_reconstructed=False,same_cell_CTG_or_image_pair=False,new_training_task_admitted=False))
 assert len(links)+len(quarantine)==20643
 lp=save('processed-to-original-cell-metadata-links',links);cp=save('nominal-condition-coordinate-evidence',[dict(source_treatment_alias=a,source_nominal_dose_label=d,processed_cells=v['cells'],processed_vector_histogram_json=json.dumps(dict(v['vectors']),sort_keys=True),source_hash_sample_labels_json=json.dumps(sorted(v['source_hash_labels'])),physical_unit=None,coordinate_evidence_not_physical_conversion=True) for (a,d),v in sorted(conditions.items())])
 qp=O/'quarantines.json';qp.open('x').write(json.dumps(quarantine,indent=2)+'\n');assert len(links)==20643 and not quarantine;assert stats['processed_n_counts_equals_original_n_umi']==20643;assert stats['source_assignment_label_present_in_sample_sheet']==20643 and stats['source_cell_library_label_has_exact_hash_record']==20643
 result=dict(schema='mora3.SciPlexOriginalCellLineageEvidence/v1',status='all20643processed_cells_exact_original_namespace_metadata_and_totalUMI_verified',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=20261031,acquisition=ref(B/'manifest.json'),source_refs=[pdref,cellref,hashref,sheetref],processed_refs=[ref(gpinfo),ref(gppert)],original_pData_records=len(pd),original_cell_annotation_records=len(cell),original_hash_UMI_records=len(hashrows),original_sample_hash_labels=len(sheet),processed_cells=20643,retained_metadata_links=len(links),quarantined_metadata_links=len(quarantine),counts=dict(stats),original_cell_indices_form_monotone_subsequence=indices==sorted(indices),original_source_full_metadata_roundtrips=True,original_tables=[pdtable,hstable,shtable],crosswalk=lp,nominal_condition_evidence=cp,quarantine=ref(qp),exact_original_cell_ID_lineage_supported=True,original_gene_count_values_5000axis_independently_reconstructed=False,rawbarcode_ID_crosswalk_not_same_as_fullrawcount_reconstruction=True,physical_dose_units_and_mapping_verified=False,source_aliases_are_not_verified_chemical_structures=True,source_assignment_not_independent_percell_exposure_or_viability_measurement=True,source_rights_review='OriginalpublicGEOmetadatareference;NCBIownnorestrictionpolicyretainspossiblethirdpartyrights;doesnotrelicenseupstreamCTGcsv. ExistingMITprocessedexpressionadmissionunchanged;newmetadataassignmentrequiresseparatetask/sourcepermissionreview.',source_images_masks_CTG_pairing=False,current_corpus_task_bytes_or_splits_changed=False,new_tasks=0,new_independent_experiments=0,model_adapter_supported_for_new_metadata=False,optimizer_steps=0,code_refs=[ref(p) for p in (O/'code').iterdir()],seconds=time.perf_counter()-start)
 (O/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');print('COMPLETE',json.dumps(dict(stats)),len(links))
except BaseException as exc:
 (O/'failure.json').open('x').write(json.dumps(dict(error=repr(exc),seconds=time.perf_counter()-start),indent=2)+'\n');raise
