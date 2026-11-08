"""Source molecule/profile observational completion in either direction.

Exact source chemical identity may condition its own recorded assay profile even
when physical dose/time are unavailable. This does not pair separate assays/cells,
validate causality or turn normalized/computed profiles into raw measurements.
"""
import math
try:
    from scripts.data_intake_corpus import CorpusTask,chemical_identity
except ModuleNotFoundError:
    from data_intake_corpus import CorpusTask,chemical_identity

def molecule_profile_task(array,smiles,modality,direction,*,cell_line=None,audit=None):
    if modality not in ('CellPainting','L1000') or direction not in ('molecule_to_profile','profile_to_molecule'):
        raise ValueError('explicit source observational direction/modality required')
    expected=(1783,) if modality=='CellPainting' else (977,978)
    axis=array['axis_ids'];values=array['values']
    if len(axis) not in expected or len(values)!=len(axis) or len(set(axis))!=len(axis) or any(not isinstance(x,str) or not x for x in axis):
        raise ValueError('complete source feature identity/order required')
    if any(type(x) not in (float,int) or not math.isfinite(x) for x in values):raise ValueError('finite complete source profile required')
    if cell_line is not None and (not isinstance(cell_line,str) or not cell_line):raise ValueError('explicit source cell line or missing required')
    iso=chemical_identity(smiles)['canonical_isomeric_smiles'];common=dict(assay_modality=modality,cell_line=cell_line)
    if direction=='molecule_to_profile':
        conditions=dict(common,canonical_isomeric_smiles=iso,known_profile_axis=axis);targets=dict(source_profile_array=array)
    else:
        conditions=dict(common,source_profile_array=array);targets=dict(canonical_isomeric_smiles=iso)
    missing=['physical_dose','exposure_time','matched_cell_state','same_cell_crossassay_pair','raw_registered_image_mask_pair','independent_biological_validation']
    if cell_line is None:missing.append('cell_line')
    evidence=dict(source_audit=audit or {},observational_scope='same_source_intervention_to_its_own_assay_profile;notCPtoL1000pairedassay',profile_evidence='source_image_computed_normalized_morphology' if modality=='CellPainting' else 'source_normalized_L1000_expression;per_feature_measurement_vs_inference_unverified;notUMIcounts',dose_and_time='notprovidedaspredictors;sourceambiguouscoordinatesremaininaudit',generated_data=False,segmentation_applicability='profile_only;novectorsegmentationorimage/maskfabrication')
    return CorpusTask('source_observational_'+direction,conditions,targets,evidence,missing)
