#!/usr/bin/env python3
"""Full-feature CPG0003 profile intake with conservative exact intervention links."""
import argparse,csv,gzip,hashlib,json,pathlib,re
from decimal import Decimal,InvalidOperation
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from data_intake_curate import digest,fresh_output,write_json,provenance,molecular_identity

STUDIES=['TA-ORF-BBBC037-Rohban','LUAD-BBBC041-Caicedo','CDRPBIO-BBBC036-Bray']

def present(x):return isinstance(x,str) and x.strip() not in ('','-666','nan','NA','EMPTY','negcon')

def link_key(study,modality,row):
 if row.get('pert_type')=='control' or present(row.get('control_type')):return None
 if study=='CDRPBIO-BBBC036-Bray':
  raw=row.get('Metadata_Sample_Dose') if modality=='CellPainting' else row.get('pert_sample_dose')
  if not present(raw) or '_' not in raw:return None
  sample,dose=raw.rsplit('_',1)
  if not sample.startswith('BRD-'):return None
  try:
   value=Decimal(dose)
   if not value.is_finite() or value<0:return None
   normalized=str(value.normalize())
  except InvalidOperation:return None
  return (study,'exact_chemical_sample_and_source_dose_coordinate',sample,normalized)
 clone=row.get('Metadata_pert_mfc_id') if modality=='CellPainting' else row.get('pert_mfc_id')
 cell=row.get('Metadata_cell_id') if modality=='CellPainting' else row.get('cell_id')
 if not present(clone) or not clone.startswith('BRDN') or not present(cell):return None
 return (study,'exact_genetic_clone_and_cell_line',clone,cell)

def context(modality,row):
 if modality=='CellPainting':
  return dict(cell_line=row.get('Metadata_cell_id') or None,physical_dose_source=row.get('Metadata_mmoles_per_liter'),physical_dose_source_column='Metadata_mmoles_per_liter' if 'Metadata_mmoles_per_liter' in row else None,exposure_time=None,exposure_unit=None,plate=row.get('Metadata_Plate'),well=row.get('Metadata_Well'))
 return dict(cell_line=row.get('cell_id') if present(row.get('cell_id')) else None,dose_source=row.get('pert_dose') if present(row.get('pert_dose')) else None,dose_unit=row.get('pert_dose_unit') if present(row.get('pert_dose_unit')) else None,exposure_time=row.get('pert_time') if present(row.get('pert_time')) else None,exposure_unit=row.get('pert_time_unit') if present(row.get('pert_time_unit')) else None,plate=row.get('det_plate'),well=row.get('det_well'))

def curate(base):
 out=fresh_output(base/'curated/cpg0003-rosetta-profile-links-v1');summaries=[];global_links=[]
 for study in STUDIES:
  groups={};source_info=[];chemical_identities={};n_control=0
  for modality,filename in [('CellPainting','replicate_level_cp_augmented.csv.gz'),('L1000','replicate_level_l1k.csv.gz')]:
   relative=f'raw/rosetta-cpg0003/{study}/{modality}/{filename}';src=provenance(base,relative);p=base/relative
   with gzip.open(p,'rt') as f:headers=next(csv.reader(f))
   if len(headers)!=len(set(headers)):raise ValueError('Duplicate source columns')
   features=[k for k in headers if re.match(r'^(Cells|Cytoplasm|Nuclei|Image)_',k)] if modality=='CellPainting' else [k for k in headers if re.match(r'^\d.*_at$',k)]
   if not features:raise ValueError('No source feature columns')
   meta=[k for k in headers if k not in features];metadata=[];start=0;missing=infinite=0;writer=None
   feature_schema=pa.schema([('source_row',pa.int64())]+[(k,pa.float64()) for k in features])
   target=out/(study+'--'+modality+'--features.parquet')
   for chunk in pd.read_csv(p,chunksize=256,converters={k:str for k in meta}):
    numeric=chunk[features].astype(np.float64);values=numeric.to_numpy();missing+=int(np.isnan(values).sum());infinite+=int(np.isinf(values).sum())
    table=pa.Table.from_pandas(numeric,preserve_index=False);table=table.add_column(0,'source_row',pa.array(range(start,start+len(chunk)),type=pa.int64())).cast(feature_schema)
    if writer is None:writer=pq.ParquetWriter(target,feature_schema,compression='zstd')
    writer.write_table(table)
    for i,(_,raw) in enumerate(chunk[meta].iterrows(),start):
     row={k:str(v) for k,v in raw.to_dict().items()};key=link_key(study,modality,row);ctx=context(modality,row)
     control=row.get('pert_type')=='control' or present(row.get('control_type'));n_control+=int(control)
     metadata.append(dict(record_id=f'{study}:{modality}:{i}',source_row=i,modality=modality,study=study,source_metadata_json=json.dumps(row,sort_keys=True),source_context_json=json.dumps(ctx,sort_keys=True),link_key_json=json.dumps(key),control=control,feature_file=target.name,feature_finite_count=int(np.isfinite(values[i-start]).sum()),feature_missing_count=int(np.isnan(values[i-start]).sum()),feature_infinite_count=int(np.isinf(values[i-start]).sum()),split='unassigned_intervention_clone_molecule_study_groups',training_ready=False,predictor_policy='Only explicit pre-assay intervention/context allowlist; all profile features and post-assay/QC metadata are targets/audit-only.'))
     if key is not None:
      g=groups.setdefault(key,{'CellPainting':[],'L1000':[],'contexts':{'CellPainting':[],'L1000':[]}});g[modality].append(i);g['contexts'][modality].append(ctx)
      if modality=='L1000' and study=='CDRPBIO-BBBC036-Bray' and present(row.get('CPD_SMILES')):
       try:identity=molecular_identity(row['CPD_SMILES']);chemical_identities.setdefault(key[2],{})[identity['canonical_isomeric_smiles']]=identity
       except ValueError:pass
    start+=len(chunk)
   if writer:writer.close()
   pq.write_table(pa.Table.from_pylist(metadata),out/(study+'--'+modality+'--metadata.parquet'),compression='zstd')
   write_json(out/(study+'--'+modality+'--columns.json'),dict(source_feature_columns=features,source_metadata_columns=meta,source_header=headers,feature_units='source-defined; L1000 normalized/inferred profile values are not UMI counts, CP are source image-derived morphology features'))
   info=dict(study=study,modality=modality,source=src,source_rows=start,retained_rows=start,feature_columns=len(features),metadata_columns=len(meta),feature_values=start*len(features),missing_feature_values=missing,infinite_feature_values=infinite,full_feature_columns_retained=True,feature_artifact=target.name)
   source_info.append(info);print('profile',study,modality,start,len(features),flush=True)
  linked=[]
  for key,g in groups.items():
   if not g['CellPainting'] or not g['L1000']:continue
   ids=chemical_identities.get(key[2],{});identity=next(iter(ids.values())) if len(ids)==1 else None
   link=dict(link_id=hashlib.sha256(json.dumps(key).encode()).hexdigest(),study=study,alignment_key_json=json.dumps(key),cellpainting_source_rows=g['CellPainting'],l1000_source_rows=g['L1000'],cellpainting_contexts_json=json.dumps(g['contexts']['CellPainting'],sort_keys=True),l1000_contexts_json=json.dumps(g['contexts']['L1000'],sort_keys=True),chemical_identity_json=json.dumps(identity),chemical_identity_conflict=len(ids)>1,alignment_scope='exact chemical sample+source dose coordinate; cell/time/physical-unit equivalence unresolved' if study=='CDRPBIO-BBBC036-Bray' else 'exact genetic clone+cell line; separate assay replicates, exposure-time equivalence unresolved',same_cell_pairing=False,fully_matched_physical_context=False,causal_response_validated=False,training_ready=False,split='unassigned_connected_intervention_molecule_clone_study_groups')
   linked.append(link)
  if linked:pq.write_table(pa.Table.from_pylist(linked),out/(study+'--cross-modal-links.parquet'),compression='zstd')
  global_links.extend(linked)
  summaries.append(dict(study=study,sources=source_info,intervention_alignment_groups=len(linked),linked_cp_profile_rows=sum(len(x['cellpainting_source_rows']) for x in linked),linked_l1000_profile_rows=sum(len(x['l1000_source_rows']) for x in linked),chemical_structure_linked_groups=sum(json.loads(x['chemical_identity_json']) is not None for x in linked),chemical_structure_conflicts=sum(x['chemical_identity_conflict'] for x in linked),control_observations_preserved=n_control,unmatched_alignment_keys=sum(not v['CellPainting'] or not v['L1000'] for v in groups.values())))
 manifest=dict(schema_version=1,source_family='Cell Painting Gallery cpg0003-rosetta',license='CC0-1.0 (Cell Painting Gallery; primary repository license/citations retained)',source_paper='Haghighi et al.,High-Dimensional Gene Expression and Morphology Profiles of Cells across28000Genetic and Chemical Perturbations,Cell Systems2022',study_summaries=summaries,retained_profile_rows=sum(s['retained_rows'] for x in summaries for s in x['sources']),intervention_alignment_groups=len(global_links),same_cell_pairs=0,fully_matched_physical_context_pairs=0,training_ready=False,scientific_limits=['Links preserve independently measured assay replicates sharing exact source intervention identifiers; never same-cell observations','No aggregation or Cartesian expansion of replicates','All source feature columns, individual labels and metadata retained; no gene/feature selection or imputation','Chemical coordinate is an exact source key, not an inferred physical unit; missing cell/time contexts remain missing','Source chemical SMILES canonicalized independently; conflicts remain gated','No synthetic expression/morphology, causal validation, clinical claims, matched baseline or train/evaluation allocation'],artifacts={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir()})
 write_json(out/'manifest.json',manifest);print('rosetta',json.dumps({k:manifest[k] for k in ['retained_profile_rows','intervention_alignment_groups','same_cell_pairs','fully_matched_physical_context_pairs']}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',type=pathlib.Path,required=True);a=p.parse_args();curate(a.base)
