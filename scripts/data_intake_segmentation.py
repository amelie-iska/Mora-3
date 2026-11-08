#!/usr/bin/env python3
"""Decode released mask composites into target-excluded immutable sidecars.

No teacher or native model is executed. Source-algorithmic masks are not manual
annotations, full-field registered labels, or independent biological validation.
"""
import argparse,hashlib,io,json,pathlib,zipfile
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from data_intake_curate import digest,fresh_output,write_json


def split_composite(array):
 a=np.asarray(array)
 if a.ndim!=2 or a.dtype!=np.uint8 or a.shape[0]<=0 or a.shape[1]!=6*a.shape[0]:raise ValueError('Require grayscale uint8 six square tiles:five images plus one mask')
 size=a.shape[0];mask=a[:,5*size:]
 if not set(np.unique(mask)).issubset({0,255}):raise ValueError('Mask tile must be source-binary0/255; no thresholding')
 channels=np.stack([a[:,i*size:(i+1)*size] for i in range(5)],axis=0).copy()
 return channels,(mask==255).astype(np.uint8)


def image_condition_bytes(array):
 a=np.asarray(array)
 if a.ndim!=2 or a.dtype!=np.uint8 or a.shape[1]!=6*a.shape[0]:raise ValueError('Unexpected input composite shape/type')
 size=a.shape[0]
 # Never inspect the mask tile, even to derive input metadata/summary features.
 pixels=np.stack([a[:,i*size:(i+1)*size] for i in range(5)],axis=0)
 header=json.dumps(dict(shape=list(pixels.shape),dtype=str(pixels.dtype)),sort_keys=True).encode()
 return header+b'\n'+pixels.tobytes(order='C')


def image_condition_summary(array):
 a=np.asarray(array);n=a.shape[0];images=a[:,:5*n].reshape(n,5,n).transpose(1,0,2)
 return dict(shape=list(images.shape),means=images.mean(axis=(1,2)).tolist(),std=images.std(axis=(1,2)).tolist())


def encode_pre_assay_context(context):
 allow=('intervention_id','intervention_kind','cell_line','chemical_smiles','gene_symbol','source_dose_coordinate','physical_dose_value','physical_dose_unit','exposure_time_value','exposure_time_unit','organism_taxon_id')
 projected={k:context.get(k) for k in allow}
 if any(v is not None and not isinstance(v,(str,int,float,bool)) for v in projected.values()):raise ValueError('Condition fields must be explicit scalar context values')
 return json.dumps(projected,sort_keys=True,allow_nan=False).encode('utf8')

def scale_context(source_display_tags):
 return dict(pixel_size_um=None,pixel_units='px',source_display_tags=source_display_tags,rule='DPI/display resolution and magnification are not calibrated specimen pixel sizes.')


def curate(base):
 root=base.parents[3];out=fresh_output(base/'curated-v3');records=[];tasks=[];checks=[]
 rosetta=root/'outputs/data-intake/20261031-subagent/curated/cpg0003-rosetta-profile-links-v1'
 matrix=json.loads((root/'planning/capability_cooperation_matrix.json').read_text());modalities=[x['id'] for x in matrix['modalities']]
 for study,plate,well,prefix,observation in [('TA-ORF-BBBC037-Rohban','41744','b05','119@','selected-rosetta-observation.json'),('CDRPBIO-BBBC036-Bray','24277','a06','976@','selected-rosetta-chemical-observation.json')]:
  # The final selected TA row is saved here, superseding the earlier probe row.
  if study.startswith('TA-ORF'):
   meta=pq.read_table(rosetta/(study+'--CellPainting--metadata.parquet')).to_pylist();obs=next(x for x in meta if x['source_row']==28)
  else:obs=json.loads((base/'research'/observation).read_text())
  source_meta=json.loads(obs['source_metadata_json']);key=json.loads(obs['link_key_json'])
  links=pq.read_table(rosetta/(study+'--cross-modal-links.parquet')).to_pylist();link=next(x for x in links if json.loads(x['alignment_key_json'])==key)
  condition_context=dict(intervention_id=source_meta.get('Metadata_pert_mfc_id'),intervention_kind='chemical' if study.startswith('CDRPBIO') else 'genetic_clone',cell_line=source_meta.get('Metadata_cell_id'),chemical_smiles=(json.loads(link['chemical_identity_json']) or {}).get('canonical_isomeric_smiles'),gene_symbol=source_meta.get('Metadata_gene_name'),source_dose_coordinate=str(key[-1]) if study.startswith('CDRPBIO') else None,physical_dose_value=None,physical_dose_unit=None,exposure_time_value=None,exposure_time_unit=None,organism_taxon_id='9606')
  encoded_context=encode_pre_assay_context(condition_context)
  expression_rows=pq.read_table(rosetta/(study+'--L1000--metadata.parquet')).to_pylist();expr=[x for x in expression_rows if x['source_row'] in link['l1000_source_rows']]
  field_assets=[]
  for p in sorted((base/'raw').glob(f'BBBC0{37 if prefix=="119@" else 36}-{plate}-{well}-s1-*.tif')):
   im=Image.open(p);a=np.asarray(im)
   if a.ndim!=2 or a.dtype!=np.uint16:raise ValueError('Unexpected native field precision/dimensions')
   field_assets.append(dict(path=str(p.relative_to(root)),sha256=digest(p),shape=list(a.shape),dtype=str(a.dtype),source_channel=p.stem.rsplit('-',1)[-1],evidence_class='source_observed_image',scale=scale_context({'XResolution':str(im.tag_v2.get(282)),'YResolution':str(im.tag_v2.get(283)),'ResolutionUnit':str(im.tag_v2.get(296))})))
  if len(field_assets)!=5:raise ValueError('Require complete original five-channel field')
  write_json(out/(study+'--field-context.json'),dict(plate=plate,well=well,site='1',raw_field_assets=field_assets,source_profile=obs,rosetta_intervention_link=link,expression_observations=expr,same_cell_pairing=False,expression_granularity='separate assay replicates linked at intervention level',source_profile_assay_granularity='well-level aggregate,1783 source CP features',physical_context_fully_matched=False,missing=['calibrated_pixel_size','exact_historical_crop_export_config','registered_full_field_mask','nucleus_mask','compartment_masks','matched_baseline']))
  for i,p in enumerate(sorted((base/'raw').glob(prefix+'*.png'))):
   raw=np.asarray(Image.open(p));channels,mask=split_composite(raw);altered=raw.copy();altered[:,5*raw.shape[0]:]=255-altered[:,5*raw.shape[0]:]
   if image_condition_bytes(raw)!=image_condition_bytes(altered) or image_condition_summary(raw)!=image_condition_summary(altered):raise AssertionError('Target tile leaked into predictor')
   ident=study+':'+plate+':'+well+':s1:'+p.stem
   image_path=out/(p.stem+'--image-channels.npy');mask_path=out/(p.stem+'--source-mask.npy')
   with image_path.open('xb') as f:np.save(f,channels,allow_pickle=False)
   with mask_path.open('xb') as f:np.save(f,mask,allow_pickle=False)
   if not np.array_equal(np.load(image_path),channels) or not np.array_equal(np.load(mask_path),mask):raise AssertionError('Sidecar roundtrip changed pixels')
   availability={k:False for k in modalities}
   for k in ['baseline_intervention_context','expression','microscopy_image','instance_compartment_segmentation','cell_morphology_spatial_graph']:availability[k]=True
   availability['therapeutic_identity']=study.startswith('CDRPBIO') and json.loads(link['chemical_identity_json']) is not None
   record=dict(schema='mora3.cooperative_segmentation_intake.v1',record_id=ident,source_study=study,plate=plate,well=well,site='1',source_composite=dict(path=str(p.relative_to(root)),sha256=digest(p),shape=list(raw.shape),evidence_class='source_derived_image_and_algorithmic_mask_composite',predictor_input_allowed=False),image=dict(path=str(image_path.relative_to(root)),sha256=digest(image_path),shape=list(channels.shape),dtype='uint8',evidence_class='source_derived_image',channels=[dict(source_tile_index=j,biochemical_channel_identity=None) for j in range(5)],transform='lossless extraction of first five image tiles; historical normalization/resampling remains source-defined'),mask=dict(path=str(mask_path.relative_to(root)),sha256=digest(mask_path),shape=list(mask.shape),evidence_class='source_algorithmic_mask',object_kind='source_single_cell_object',mask_encoding='0background/1source_object,from exact source0/255 binary tile',source_method='CellProfiler/DeepProfiler released derivative; precise historical version/config unresolved',independent_ground_truth=False,full_field_registered=False),source_context_file=study+'--field-context.json',source_context_file_role='audit_only_not_a_predictor_payload',pre_assay_conditions=json.loads(encoded_context),source_profile_ref=dict(manifest=str((rosetta/'manifest.json').relative_to(root)),source_row=obs['source_row'],feature_file=obs['feature_file'],columns=1783,granularity='same source plate/well aggregate'),intervention_link=link,modality_availability=availability,missing_modality_mask={k:not v for k,v in availability.items()},physical_pixel_size_um=None,spatial_reference='crop-local pixel y/x; filename centroids retained as tokens,not promoted to verified full-field mapping',cell_entity=dict(source_kind='one upstream selected crop object',biological_identity=None,arbitrary_instance_label_is_identity=False),nucleus_mask_available=False,compartment_masks_available=False,training_ready=False,split='unassigned_parent_plate_well_site_intervention_groups',leakage_groups=[f'{study}:plate:{plate}:well:{well}',':'.join(str(x) for x in key)],applicability_gates=['source-native crop supervision candidate only','teacher/pseudo-label use policy and source method review','unknown biochemical order of five derivative image tiles','historical crop160 versus current exporter config128','no calibrated scale or registered full-field mask','no same-cell expression claim'])
   records.append(record)
   for direction in ['image_to_source_mask','observed_geometry_to_image_reconstruction']:
    image_input=direction=='image_to_source_mask'
    tasks.append(dict(task_id=ident+':'+direction,record_id=ident,direction=direction,conditions=['image.path','pre_assay_conditions'] if image_input else ['mask.path','pre_assay_conditions'],targets=['mask.path'] if image_input else ['image.path'],excluded_from_predictors=['source_composite','mask.path','mask-derived area/count/geometry','filename_class_label','target-derived topology','source_profile values','expression values','source_context_file audit payload','post-assay/QC metadata'] if image_input else ['image.path','source_composite','target image intensities'],mask_role='target_source_algorithmic' if image_input else 'observed_geometric_condition',desired_geometry_role='separate desired_user_condition at inference; no fabricated desired training observations',requires_native_head_channels=5,model_integration_ready=False,condition_context_encoding_sha256=hashlib.sha256(encoded_context).hexdigest(),audit_context_excluded=True,evidence_class='source_algorithmic_supervision_candidate',independent_validation=False,training_ready=False))
   checks.append(dict(record_id=ident,image_pixels_unchanged_when_mask_flipped=True,packed_predictor_bytes_unchanged=True,all_input_summary_features_unchanged=True,mask_area_px=int(mask.sum())))
 # Read-only reuse of source-provided masks with native1-channel images for next integration.
 references=[]
 with (root/'outputs/data/imaging-public-20261031-v2/records.jsonl').open() as f:
  candidates=[json.loads(l) for l in f if 'zenodo_5550933_DeepBacs_Data_Segmentation_Staph_Aureus_dataset:' in l]
 for split in ['train','test']:
  selected=[r for r in candidates if r['split']==split and r['width']<=512 and r['height']<=512][:2]
  if len(selected)!=2:raise ValueError('Require2native-compatible full fields per split')
  for r in selected:
   archive=root/r['source_archive']
   with zipfile.ZipFile(archive) as z:
    for kind in ['image','mask']:
     raw=z.read(r['source_members'][kind]);assert hashlib.sha256(raw).hexdigest()==r[kind+'_sha256']
   references.append(dict(source_record=r,evidence_class='source_instance_annotation_manual_status_unverified',read_only_reuse=True,physical_scale_known=False,intervention_available=False,expression_available=False,allowed_candidate_directions=['observed_image_to_instance_mask','observed_geometry_to_image_reconstruction'],blocked_directions=['image_or_mask_to_unobserved_expression_or_therapeutic_response'],keep_official_split=split,training_admission_not_promoted=True))
 write_json(out/'records.json',records);write_json(out/'tasks.json',tasks);write_json(out/'target-exclusion-checks.json',checks);write_json(out/'native-deepbacs-readonly-reuse.json',references)
 write_json(out/'desired-generation-routing-template.json',dict(role='inference_contract_only_no_generated_samples',desired_condition_evidence_class='desired_user_condition',generated_image_evidence_class='generated_image',mask_of_generated_image_evidence_class='predicted_mask_on_generated_image',prediction_evidence_class='model_predicted_response',checkpoint_seed_and_method_required=True,independent_experimental_validation=False,route='supported generated/predicted cellular image -> native segmentation -> px-unit object measurements/spatial graph -> constraint assessment',unsupported_output_policy='unevaluated; no fabricated image,mask,physical units or response',candidate_task='desired object geometry plus available source context -> image generation; requires native image decoder and cooperative adapter,not yet implemented'))
 manifest=dict(schema='mora3.cooperative_segmentation_slice.v3',supersedes='cooperative-segmentation-v1/curated/manifest.json;v1 broad audit-context pointers are not admitted as predictor inputs',governing_contract='planning/cooperative_segmentation_contract.md',seed=20261031,source_native_crop_records=len(records),raw_full_field_channel_files=10,intervention_wells=2,chemical_intervention_wells=1,genetic_intervention_wells=1,source_algorithmic_mask_crops=len(records),manual_masks_newly_acquired=0,native_existing_DeepBacs_references=len(references),task_candidates=len(tasks),same_cell_expression_pairs=0,fully_matched_assay_context_pairs=0,predictor_mask_tile_mutation_checks=len(checks),training_ready=False,source_image_ID_collision='Rosetta LUAD-BBBC041 is historical source label; compatible currentBBBC043/CPG0031 established from primary rename documentation. CurrentBBBC041 malaria is unrelated.',profile_only_rule='Profile tables are not segmented. These two exact plate/well observations have raw images and derivative crops; remaining Rosetta observations stay profile-only until lineage verified.',next_model_task=dict(id='CSEG-01/02-native-evidence-router',command='/home/iska/miniconda3/envs/tokengt/bin/python -m pytest -q tests/data_intake/test_segmentation.py tests/imaging',acceptance='Implement shared evidence/applicability routing and target-excluded array adapter; use four read-only DeepBacs source image/mask pairs for native1-channel image->mask->px-object graph roundtrip. Reject profile-only inputs; preserve source_algorithmic versus source annotations and generated image/mask classes. Keep5-channel crop tasks explicitly unsupported until a native compatible head exists.'),artifacts={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir()})
 write_json(out/'manifest.json',manifest);print(json.dumps({k:manifest[k] for k in ['source_native_crop_records','raw_full_field_channel_files','native_existing_DeepBacs_references','task_candidates','predictor_mask_tile_mutation_checks']}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',type=pathlib.Path,required=True);a=p.parse_args();curate(a.base.resolve())
