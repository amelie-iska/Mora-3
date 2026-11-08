"""Source-reported assay tasks; explicit desired priors are not biological validation."""
from pathlib import Path
import sys,math
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_intake_corpus import CorpusTask
PANEL='NCI_public_AIDS_screen_source_protocol'
READOUT={'EC50':'protection_of_infected_cells','IC50':'growth_inhibition_of_uninfected_cells'}
MISSING=['verified_stereochemical_identity_and_lot','exact_cell_line_strain_duration_and_replicate_context','same_cell_image_expression_or_segmentation_pair','clinical_outcome_or_mechanism_validation','independent_study_blind_evaluation']

def direction_ok(direction):
    if direction not in ['prediction','generation']:raise ValueError('explicit task direction required')

def conclusion_task(smiles,label,direction,source_ref):
    direction_ok(direction)
    if label not in ['CA','CM','CI']:raise ValueError('unknown source class')
    if not isinstance(smiles,str) or not smiles:raise ValueError('source-derived connectivity required')
    base=dict(assay_panel=PANEL,source_stereochemistry_known=False)
    if direction=='prediction':conditions=dict(base,molecule_source_connectivity_smiles=smiles);targets=dict(source_conclusion=label)
    else:conditions=dict(base,desired_source_conclusion=label);targets=dict(molecule_source_connectivity_smiles=smiles)
    return CorpusTask('source_NCI_assay_conclusion_'+direction+'_prior',conditions,targets,dict(source_ref=source_ref,evidence='source_manual_overall_judgment_of_experimental_dose_response_curves;not_direct_binding_or_clinical_efficacy',future_generated_outputs_require_model_produced_provenance=True,source_geometry='Corina_computed_not_input_or_observed_target',source_missing_stereochemistry_is_not_achirality=True),MISSING)

def point_check(point):
    if not point.get('point_target_available'):raise ValueError('source aggregate censoring is not an available point target')
    if point['source_ConcUnit'] not in ['M','u']:raise ValueError('unknown physical source unit')
    if point['endpoint'] not in READOUT or not math.isfinite(point['source_value']) or not math.isfinite(point['source_log10_highest_concentration']):raise ValueError('finite source endpoint and design required')

def endpoint_task(smiles,point,direction,source_ref):
    direction_ok(direction);point_check(point)
    base=dict(assay_panel=PANEL,assay_readout=READOUT[point['endpoint']],concentration_unit=point['source_ConcUnit'],source_log10_highest_tested_concentration=point['source_log10_highest_concentration'],source_stereochemistry_known=False)
    if direction=='prediction':conditions=dict(base,molecule_source_connectivity_smiles=smiles);targets=dict(source_log10_endpoint_mean=point['source_value'])
    else:conditions=dict(base,desired_source_log10_endpoint_mean=point['source_value']);targets=dict(molecule_source_connectivity_smiles=smiles)
    return CorpusTask('source_NCI_assay_endpoint_'+direction+'_prior',conditions,targets,dict(source_ref=source_ref,source_endpoint=point['endpoint'],evidence='source_computed_summary_of_experimental_assay_measurements;allreportedexperimentsreachedendpoint',source_flag='=',native_unit_role='M_molar_or_u_micrograms_per_mL;no_mass_to_molar_conversion',source_mean_scale='log10concentration_nativeheader',reported_SD_and_NumExp='retained_source_refs;notpredictors;zeroSDnotperfectprecision;noSEM_inference',future_generated_outputs_require_model_produced_provenance=True,source_geometry='computed_notobserved'),MISSING)

def joint_task(smiles,ec,ic,direction,source_ref):
    direction_ok(direction);point_check(ec);point_check(ic)
    if ec['endpoint']!='EC50' or ic['endpoint']!='IC50' or ec['compound_highestdose_context_key']!=ic['compound_highestdose_context_key']:raise ValueError('literal compound/unit/highest-dose context must match')
    base=dict(assay_panel=PANEL,endpoint_axis=['infected_cell_protection_EC50','uninfected_cell_growth_IC50'],concentration_unit=ec['source_ConcUnit'],source_log10_highest_tested_concentration=ec['source_log10_highest_concentration'],source_stereochemistry_known=False)
    values=[ec['source_value'],ic['source_value']]
    if direction=='prediction':conditions=dict(base,molecule_source_connectivity_smiles=smiles);targets=dict(source_log10_endpoint_means=values)
    else:conditions=dict(base,desired_source_log10_endpoint_means=values);targets=dict(molecule_source_connectivity_smiles=smiles)
    return CorpusTask('source_NCI_paired_assay_profile_'+direction+'_prior',conditions,targets,dict(source_ref=source_ref,evidence='two_source_assay_summary_means;compound_and_dose_range_context_link',same_cell_or_replicate_pair=False,source_lot_or_full_protocol_identity_verified=False,selectivity_ratio_or_therapeutic_index_derived=False,future_generated_outputs_require_model_produced_provenance=True,source_censor_flags=['=','=']),MISSING)
