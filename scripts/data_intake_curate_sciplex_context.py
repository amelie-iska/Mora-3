from pathlib import Path
import sys,json,csv,hashlib,urllib.request,collections,time,datetime,statistics
import pyarrow as pa
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_sciplex_context import hash_sample,ctg_record
B=R/'outputs/data-intake/20261031-sciplex-original-context-v1';raw=B/'research-original-repo';out=B/'curated-source-context-v2';out.mkdir();prior=B/'curated-source-context-v1'
start=time.perf_counter();tree=json.loads((raw/'tree.json').read_text());commit='079639c50811dd43a206a779ab2f0199a147c98f';files=[]
def receipt(path,**kw):
 return dict(path=str(path.relative_to(R)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**kw)
for name in ['bin/dose_response.R','bin/viability.R']:
 entry=next(row for row in tree['tree'] if row['path']==name);url=f'https://raw.githubusercontent.com/cole-trapnell-lab/sci-plex/{commit}/{name}'
 path=prior/(Path(name).name+'.source.txt');data=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert blob==entry['sha']
 files.append(receipt(path,url=url,git_blob=blob,role='read_only_primary_analysis_code_not_executed'))
ctgp=next(raw.glob('*CTG_results.csv'));hashp=next(raw.glob('*hashSampleSheet.txt'))
with ctgp.open(newline='') as f:source=list(csv.DictReader(f))
ctg=[ctg_record(row,i+2) for i,row in enumerate(source)]
lines=hashp.read_text().splitlines(keepends=True);samples=[hash_sample(line,i+1) for i,line in enumerate(lines) if line.strip()]
assert len(ctg)==192 and len(samples)==192
assert len({r['source_sample_label'] for r in samples})==192 and len({r['hash_oligo'] for r in samples})==192
assert {r['cell_type'] for r in ctg}=={'A549'}
cgroups=collections.defaultdict(list);hgroups=collections.defaultdict(list)
for r in ctg:cgroups[r['condition_group']].append(r)
for r in samples:hgroups[r['condition_group']].append(r)
assert set(cgroups)==set(hgroups) and len(cgroups)==32
assert all(len(v)==6 and {r['source_replicate_index'] for r in v}==set(range(1,7)) for v in cgroups.values())
assert all(len(v)==6 for v in hgroups.values())
conditions=[]
for group,rows in sorted(cgroups.items()):
 conditions.append(dict(condition_group=group,cell_type=rows[0]['cell_type'],source_treatment_label=rows[0]['source_treatment_label'],source_dose_coordinate=rows[0]['source_dose_coordinate'],source_real_dose_coordinates=sorted({r['source_real_dose_coordinate'] for r in rows}),CTG_source_lines=[r['source_line'] for r in rows],hash_sample_labels=[r['source_sample_label'] for r in hgroups[group]],CTG_replicates=len(rows),hash_sample_labels_count=len(hgroups[group]),join_granularity='source_condition_coordinate_separate_assays',same_cell=False,same_well=False,GPerturb_cell_mapping_verified=False,physical_dose_mapping_verified=False,source_unit=None,available_modality_masks_json=json.dumps(dict(source_CTG=True,sample_hash_context=True,raw_image=False,instance_mask=False,registered_expression_pair=False)),admitted_for_task=False))
shards=[]
for name,rows in [('CTG-source-records',ctg),('hash-sample-labels',samples),('condition-reference-links',conditions)]:
 path=out/(name+'.parquet');pq.write_table(pa.Table.from_pylist(rows),path,compression='zstd');roundtrip=pq.read_table(path).to_pylist();assert roundtrip==rows;shards.append(receipt(path,rows=len(rows)))
# Independent source-string readback of every row, rather than a probe.
back=pq.read_table(out/'CTG-source-records.parquet').to_pylist()
assert [json.loads(r['source_fields_json']) for r in back]==source
assert all(r['source_CTG']==float(s['CTG']) and r['source_real_dose_coordinate']==float(s['real_dose']) for r,s in zip(back,source))
assert [r['source_line_text'] for r in pq.read_table(out/'hash-sample-labels.parquet').to_pylist()]==[line.rstrip('\r\n') for line in lines if line.strip()]
manifest=dict(schema='mora3.SciPlex_original_separate_assay_context.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_repo='cole-trapnell-lab/sci-plex',source_commit=commit,source_files=[receipt(ctgp),receipt(hashp)],new_primary_analysis_code=files,shards=shards,source_CTG_records=192,source_sample_hash_labels=192,source_condition_coordinates=32,retained_source_records=384,quarantined_source_records=0,source_well_context_missing=sum(r['source_culture_plate'] is None for r in samples),source_well_context_missing_note='one source sample SAHA_100_E09 retained;plate/well not imputed',admitted_tasks=0,all_source_rows_exact_roundtrip=True,source_CTG_range=[min(r['source_CTG'] for r in ctg),max(r['source_CTG'] for r in ctg)],values_above_one_not_clamped=sum(r['source_CTG']>1 for r in ctg),condition_CTGreps=6,condition_hashlabels=6,source_dose_coordinates=sorted({r['source_dose_coordinate'] for r in ctg}),real_to_label_nonzero_ratios={t:sorted({r['source_real_dose_coordinate']/r['source_dose_coordinate'] for r in ctg if r['source_treatment_label']==t and r['source_dose_coordinate']}) for t in sorted({r['source_treatment_label'] for r in ctg})},raw_barcode_or_export_join_verified=False,physical_dose_mapping_verified=False,source_rights_status='repository has no explicit license;publication/data-specific grant unresolved;candidate review only',readout_evidence='source reported CTG assay readout;units/normalization not independently established;not clamped cell survival fraction',inferred_viability_warning='bin/viability.R derives per-cell viability from fitted recovered-cell-count dose curves;model-derived annotation,not direct per-cell measured viability',pair_scope='original notebook joins aggregated separate assays by treatment and source dose coordinate;not same cells or matched wells',segmentation_applicability='context reference only;no raw images, masks or same-cell links in these files',open_gates=['source rights','physical dose units and two-coordinate mapping','processed GPerturb-to-original hash/cell mapping','same-well assay linkage','direct vs inferred viability distinction'],scope='does not change admitted observational SciPlex expression tasks or their preserved native bytes',model_adapter_supported=False,optimizer_steps=0,seconds=time.perf_counter()-start)
with (out/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
print(json.dumps({k:manifest[k] for k in ['source_CTG_records','source_sample_hash_labels','source_condition_coordinates','admitted_tasks','all_source_rows_exact_roundtrip','source_CTG_range','values_above_one_not_clamped','seconds']}))
