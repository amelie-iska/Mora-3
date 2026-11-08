"""Task-specific molecular reference/codec gates, retaining complete source axes."""
import hashlib,json
from rdkit import Chem,rdBase
import selfies as sf
from data_intake_corpus import CorpusTask,chemical_identity
MISSING=['experimental_property_activity_or_response_labels','physical_assay_dose_time_lot_and_cell_context','3D_geometry_images_expression_or_segmentation','independent_blind_chemical_evaluation']

def validate_native(row):
    if row.get('coordinate_geometry_present') is not False:raise ValueError('source geometry must be absent')
    smiles=row.get('native_smiles')
    if row.get('native_current_RDKit_parse_pass') is not True or not isinstance(smiles,str) or not smiles:raise ValueError('valid native source graph required')
    with rdBase.BlockLogs():mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('invalid native source graph')
    iso=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
    if iso!=row['source_canonical_isomeric_smiles'] or hashlib.sha256(iso.encode()).hexdigest()!=row['source_current_canonical_graph_sha256']:raise ValueError('source canonical graph identity mismatch')
    present=row.get('standard_selfies_target_present');passed=row.get('graph_roundtrip_pass')
    if type(present) is not bool or type(passed) is not bool or present!=passed:raise ValueError('incoherent SELFIES availability mask')
    if present:
        value=row.get('standard_selfies')
        if not isinstance(value,str) or not value:raise ValueError('missing SELFIES target')
        decoded=sf.decoder(value)
        with rdBase.BlockLogs():back=Chem.MolFromSmiles(decoded)
        if back is None or Chem.MolToSmiles(back,canonical=True,isomericSmiles=True)!=iso:raise ValueError('SELFIES exact isomeric graph roundtrip mismatch')
    elif row.get('standard_selfies') is not None:raise ValueError('unavailable SELFIES must not be an admitted target')
    return dict(smiles_eligible=True,selfies_eligible=present,identity=chemical_identity(smiles),source_canonical_isomeric_smiles=iso)

def smiles_prior(row,source_ref,*,validated=None):
    v=validated if validated is not None else validate_native(row)
    if not v['smiles_eligible']:raise ValueError('valid native source graph required')
    return CorpusTask('source_native_molecular_SMILES_reference_prior',{'prior_domain':'molecular_reference_native_SMILES'},{'native_smiles':row['native_smiles']},dict(source_ref=source_ref,evidence_class='source_reference_connectivity_not_experimental_response',source_native_split_audit_only=row['source_native_split'],standard_SELFIES_available=v['selfies_eligible'],source_geometry_present=False,source_dataset='sagawa/pubchem-10m-canonicalized@f83219601635a0a80fc99c13a9ca37f99ef34f0a',future_generated_outputs_require_model_produced_provenance=True),MISSING)

def codec_task(row,direction,source_ref,*,validated=None):
    if direction not in ['smiles_to_selfies','selfies_to_smiles']:raise ValueError('explicit codec direction required')
    v=validated if validated is not None else validate_native(row)
    if not v['selfies_eligible']:raise ValueError('source SELFIES graph-roundtrip target unavailable')
    base={'representation_task':'standard_SELFIES_codec'}
    if direction=='smiles_to_selfies':conditions=dict(base,native_smiles=row['native_smiles']);targets={'standard_selfies':row['standard_selfies']}
    else:conditions=dict(base,standard_selfies=row['standard_selfies']);targets={'source_canonical_isomeric_smiles':row['source_canonical_isomeric_smiles']}
    return CorpusTask('computed_standard_SELFIES_'+direction,conditions,targets,dict(source_ref=source_ref,evidence_class='computed_representation_of_source_reference_graph_not_experiment',standard_SELFIES_not_original_GroupSELFIES=True,atomic_charge_isotope_and_isomeric_graph_roundtrip_verified=True,source_geometry_present=False,source_native_split_audit_only=row['source_native_split']),MISSING)
