"""Native original RNAcentral reference priors; source status is never activity truth."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_intake_corpus import CorpusTask
ALPHABET=frozenset('ACGTURYSWKMBDHVNacgturyswkmbdhvn')

def base_known_mask(sequence):return [x.upper() in 'ACGTU' for x in sequence]

def prior_task(sequence,*,source_ref):
    if not isinstance(sequence,str) or not sequence or set(sequence)-ALPHABET:
        raise ValueError('nonempty original source IUPAC nucleotide sequence required; no repair')
    return CorpusTask('source_native_nucleotide_sequence_prior',
        dict(prior_domain='RNA_reference_native_symbols',source_organism=None),
        dict(source_native_sequence=sequence),
        dict(source_ref=source_ref,source_role='releasedRNAreference_sequence_prior;notindependentexperimentalmeasurement',source_record_status_is_not_activity_label=True,source_symbols='nativecase_T_U_IUPACretainedwithoutcroporrepair',source_symbol_loss_mask='allnativevalidcharactersobserved;biologicalbase-resolutionlossonlyupperACGTU;ambiguitynotimputed',excluded_predictors=['sourceidentifiers_headers_status_classifications','targetsequence_length_hash','sourceauditmetadata'],native_reader_supported=False),
        ['native_neural_RNAprior_reader','nativeheader_taxonomy_function_scope_verification','homology_and_genuine_heldout_protocol','cellular_response_or_imagepair'])
