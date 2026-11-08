"""Canonical-sequence conditions with local evidence and explicit product scope."""
import math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_intake_uniprot_annotations import validate_entry,safe_generation_conditions,ec_assertions,direct_experimental
from data_intake_uniprot_supplement import physical_records
from data_intake_corpus import CorpusTask

def canonical_scope(e,pointer):
    parts=pointer.split('/')
    if len(parts)>2 and parts[1]=='comments':
        return not e['comments'][int(parts[2])].get('molecule')
    if parts[1:2]==['proteinDescription']:
        return parts[2:3] in [['recommendedName'],['alternativeNames']]
    return False

def generation_task(e,*,source_ref=None):
    identity=validate_entry(e);c=safe_generation_conditions(e)
    if c is None:return None
    c['source_experimentally_supported_EC_ids']=sorted({x['normalized']['identifier'] for x in ec_assertions(e) if canonical_scope(e,x['source_pointer']) and direct_experimental(x['source_value'])})
    kinetics=[]
    for row in physical_records(e):
        if row['parameter_type'] not in ['MichaelisConstant','maximumVelocity'] or row['molecule_scope'] is not None:continue
        import json
        original=json.loads(row['source_value_json'])
        if direct_experimental(original) and type(row['value']) in [int,float] and math.isfinite(row['value']) and isinstance(row['unit'],str) and row['unit']:
            kinetics.append(dict(parameter_type=row['parameter_type'],source_value=row['value'],source_unit=row['unit'],substrate_label=row.get('substrate'),assay_pH=None,assay_temperature=None))
    c['source_kinetic_conditions']=kinetics
    if not any(c[k] for k in c if k!='organism_taxon_id'):return None
    return CorpusTask('source_UniProt_annotation_conditioned_canonical_sequence',c,dict(canonical_sequence=e['sequence']['value']),dict(identity=identity,source_ref=source_ref,condition_role='source_observed_annotations;notmodelgeneratedvalidation',scope_policy='canonicalconditionsonly;parentcommentmoleculeandprocessedproductECexcluded;noisoformlabelinheritance',excluded_condition_fields=['sequence','sequence_length_mass_checksum_pI','protein_gene_names','accessions_and_crossref_IDs_otherthanGO_EC','citations_evidence_referenceIDs','residue_features_variants_isoformsequences','inferred_only_conditions','unverified_product_specific_conditions'],evidence_policy='directECO0000269forfunction_EC_kinetics;experimentalGOcodes;localbindingnotparentbroadcast',model_adapter_supported=False),['per_kinetic_assay_pH_temperature_protocol','validated_homology_heldout','all_unverified_isoform_label_transfer','native_model_annotation_conditioned_decoder'])
