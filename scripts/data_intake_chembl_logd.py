"""One original ChEMBL experimental logD source objective; not CX computed LogD.

The source relation/units fields remain missing. Interior nominal readouts may
supervise a source-value task; boundary values stay gated for censoring review.
All original activity fields remain audit-only and never become predictors.
"""
import math
try:
    from scripts.data_intake_corpus import CorpusTask,chemical_identity
except ModuleNotFoundError:
    from data_intake_corpus import CorpusTask,chemical_identity

def source_logd_task(row):
    if row['assay_chembl_id']!='CHEMBL3301363' or row['assay_type']!='P' or row['standard_type']!='LogD7.4' or row['type']!='LogD7.4':
        raise ValueError('original experimental assay required')
    if row['standard_relation'] is not None or row['relation'] is not None:
        raise ValueError('source relation requires separate censored objective')
    if row['standard_units'] is not None or row['units'] is not None or row['standard_flag']!=1 or row['data_validity_comment'] is not None:
        raise ValueError('unexpected original source schema/validity gate')
    value=float(row['standard_value'])
    if not math.isfinite(value) or value!=float(row['value']):raise ValueError('finite matching original and standard value required')
    if not -1.5<value<4.5:raise ValueError('assay boundary with unspecified censoring retained for separate review')
    identity=chemical_identity(row['canonical_smiles'])
    return CorpusTask('source_reported_experimental_logD7_4',dict(canonical_isomeric_smiles=identity['canonical_isomeric_smiles'],pH=7.4,solvent_pair=['octan-1-ol','water'],method='shake_flask'),dict(source_reported_logD7_4=value),dict(source_activity=row,source_relation=row['standard_relation'],source_units=row['standard_units'],evidence='source_experimental_numeric_readout;relation_unspecified;not_computed_CX_LogD;not_protein_affinity'),['assay_temperature','assay_duration','measurement_uncertainty','source_relation_unspecified','paired_cellular_modalities'])
