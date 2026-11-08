"""Source-computed Rfam family priors, separate from experimental RNA structure."""
import re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_intake_corpus import CorpusTask
from data_intake_nucleotide_prior import sequence_groups

def family_task(sequence,families,clans,audit,direction):
    sequence_groups(sequence)
    families=sorted(set(families));clans=sorted(set(clans))
    if not families or any(not isinstance(x,str) or re.fullmatch(r'RF[0-9]{5}',x) is None for x in families):
        raise ValueError('explicit source Rfam accession required')
    if any(not isinstance(x,str) or re.fullmatch(r'CL[0-9]{5}',x) is None for x in clans):
        raise ValueError('explicit source clan accession required')
    if direction=='prediction':
        conditions=dict(source_native_sequence=sequence)
        targets=dict(source_Rfam_family_ids=families,source_clan_ids=clans,family_label_policy='positive_source_assignments_only;all_unlisted_families_unknown')
    elif direction=='generation':
        conditions=dict(desired_source_Rfam_family_ids=families,desired_source_clan_ids=clans)
        targets=dict(source_native_sequence=sequence)
    else:raise ValueError('explicit source-prior prediction/generation direction required')
    return CorpusTask('Rfam_source_family_'+direction+'_prior',conditions,targets,dict(source_evidence='source_curated_and_covariance_model_assigned_family_prior;not_independent_experiment',source_release='Rfam15.1 January2026',source_context=audit,negative_labels_available=False,excluded_predictors=['source_headers_coordinates_IDs','bitscore_Evalue_truncation_CM_positions','family_catalog_names_citations_model_derived_statistics','target_length_and_hash']),['independent_experimental_function_or_3D_structure','unlisted_family_negative_truth','cell_perturbation_or_expression_pair','homology_taxonomy_parent_heldout','native_model_family_prior_reader'])
