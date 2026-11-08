"""LINCS native profile completion with source units, null masks and exact types."""
import math
try:
    from data_intake_corpus import CorpusTask
    from data_intake_chemical_projection_scope import project_source_graph
except ModuleNotFoundError:
    from scripts.data_intake_corpus import CorpusTask
    from scripts.data_intake_chemical_projection_scope import project_source_graph


def source_native_profile_task(array,smiles,modality,direction,*,nominal_dose_micromolar,time_hours,cell_line,audit=None):
    if modality not in ('CellPainting','L1000') or direction not in ('molecule_to_profile','profile_to_molecule'):
        raise ValueError('explicit source modality and observational completion direction required')
    if cell_line!='A549' or time_hours!=(48 if modality=='CellPainting' else 24):
        raise ValueError('verified source time and A549 context required')
    if type(nominal_dose_micromolar) not in (float,int) or not math.isfinite(nominal_dose_micromolar) or nominal_dose_micromolar<=0:
        raise ValueError('positive source treatment dose required; controls/sentinels are not physical-dose tasks')
    expected=1783 if modality=='CellPainting' else 978
    axes=array['axis_ids'];values=array['values'];mask=array['observed_mask'];types=array['native_types']
    if any(len(x)!=expected for x in [axes,values,mask,types]) or len(set(axes))!=expected or any(not isinstance(k,str) or not k for k in axes):
        raise ValueError('full ordered native feature axis and labels required')
    for value,observed,native_type in zip(values,mask,types):
        if type(observed) is not bool or observed!=(value is not None):
            raise ValueError('exact native null validity required; no imputation')
        if native_type not in ('double','int64'):
            raise ValueError('source native int64/double dtype required')
        if value is None:continue
        if native_type=='int64' and (type(value) is not int or not -2**63<=value<2**63):
            raise ValueError('native int64 must remain an exact integer')
        if native_type=='double' and (type(value) is not float or not math.isfinite(value)):
            raise ValueError('native finite float64 required')
    if not any(mask):raise ValueError('at least one source observed feature required')
    projection=project_source_graph(smiles)
    if not projection['exact_projection_eligible']:raise ValueError('strict source reference graph required')
    iso=next(ident['canonical_isomeric_smiles'] for role,ident in projection['graphs'] if role=='whole_source_graph')
    common=dict(assay_modality=modality,cell_line=cell_line,
                source_nominal_dose=dict(value=nominal_dose_micromolar,unit='micromolar',evidence='author curated source field; no new conversion or concentration calibration'),
                source_exposure_time=dict(value=time_hours,unit='hours',evidence='author curation assignment retained in every native row'))
    if direction=='molecule_to_profile':
        conditions=dict(common,canonical_isomeric_smiles=iso,known_profile_axis=axes,known_native_feature_types=types)
        targets=dict(source_profile_native_array=array)
    else:
        conditions=dict(common,source_profile_native_array=array)
        targets=dict(canonical_isomeric_smiles=iso)
    evidence=dict(source_audit=audit or {},source_observed_or_model_generated='source_observation_with_computed_assay_features',
                  profile_evidence='source_image_computed_CellPainting_well_profile' if modality=='CellPainting' else 'source_preprocessed_L1000_probe_profile;measurement_vs_inference_perprobe_unresolved',
                  observed_mask_role='forward target/loss only; inverse explicitly observed input mask; no target-derived predictor mask',
                  native_int64_and_float64_preserved=True,all_feature_axes_preserved=True,
                  source_nominal_units_not_assay_calibration=True,
                  source_profile_not_single_cell_or_object_geometry=True,
                  same_cell_crossassay_pair=False,matched_replicate_crossassay_pair=False,
                  causal_or_independent_biological_validation=False,generated_data=False,
                  desired_profile_inference_requires_separate_desired_evidence_role=True,
                  source_target_loss='observed features only; missing is unknown, never zero/negative')
    missing=['raw_image_mask','registered_single_cell_or_spatial_entities','physical_pixel_scale_and_channels','matched_cell_state','same_cell_crossassay_pair','calibrated_assay_concentration','independent_biological_validation']
    return CorpusTask('source_observational_LINCS_'+direction,conditions,targets,evidence,missing)
