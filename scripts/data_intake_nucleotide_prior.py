"""Data-only full native-symbol DNA/RNA-reference priors; no biological base imputation."""
import hashlib,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_intake_corpus import CorpusTask,canonical
ALPHABET=set('ACGTURYSWKMBDHVNXacgturyswkmbdhvnx')
_COMPLEMENT=str.maketrans('ACGTRYSWKMBDHVNX','TGCAYRSWMKVHDBNX')

def base_known_mask(sequence):return [x.upper() in 'ACGTU' for x in sequence]

def sequence_groups(sequence):
    if not isinstance(sequence,str) or not sequence or set(sequence)-ALPHABET:raise ValueError('nonempty original native nucleotide alphabet required')
    upper=sequence.upper();baseorder=upper.replace('U','T');rc=baseorder.translate(_COMPLEMENT)[::-1];order=min(baseorder,rc)
    return dict(native_sha256=hashlib.sha256(sequence.encode()).hexdigest(),uppercase_sha256=hashlib.sha256(upper.encode()).hexdigest(),reverse_complement_base_order_group=hashlib.sha256(order.encode()).hexdigest(),derived_group_is_molecular_identity=False)

def prior_identity(sequence,domain,organism):return hashlib.sha256(canonical([domain,organism,hashlib.sha256(sequence.encode()).hexdigest()])).hexdigest()

def rfam_parent(identifier):
    match=re.fullmatch(r'([^\s/]+)/([0-9]+)-([0-9]+)',identifier)
    ref=match[1] if match else None;interval=[int(match[2]),int(match[3])] if match else None
    groups=[]
    if ref:
        groups=['source_nucleotide_reference:'+ref]
        if re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*[.][0-9]+',ref):groups+=['genbank:'+ref,'genomic_accession_unversioned:'+re.sub(r'[.][0-9]+$','',ref)]
    else:groups=['unparsed_nucleotide_header:'+hashlib.sha256(identifier.encode()).hexdigest()]
    return dict(reference_id=ref,source_interval_1based_closed=interval,source_coordinate_order='increasing' if interval and interval[1]>=interval[0] else 'decreasing' if interval else None,observed_strand=None,Rfam_family_id=None,groups=groups,parent_scope='source_header_reference_notindependentlyrefetched' if ref else 'unparsed_source_header')

def prior_task(sequence,*,domain,organism,source_ref):
    sequence_groups(sequence)
    if domain not in ['source_DNA_sequence','RNA_family_reference_native_symbols']:raise ValueError('unreviewed prior domain')
    if organism is not None and (not isinstance(organism,str) or not organism):raise ValueError('literal source organism or explicit None required')
    return CorpusTask('source_native_nucleotide_sequence_prior',dict(prior_domain=domain,source_organism=organism),dict(source_native_sequence=sequence),dict(source_ref=source_ref,source_role='releasedreference_sequence_prior;notindependentexperimentalmeasurement',source_symbols='case_T_U_andIUPACambiguitypreserved;no sourcecrop',source_symbol_loss_mask='allsourcevalidcharactersobserved;biologicalbase-resolutionobjectivesrequireuppercharinACGTU;noambiguityimputation',excluded_predictors=['sourceIDs_headers_coordinates_features_proteins','targetsequence_length_and_hash','auditandfullnative_sourceobjects'],native_reader_supported=False,source_proteins_admitted=False),['verified_translation_CDS_strand_join_codonstart_geneticcode','sourceprotein_annotation_qualification','Rfam_family_confidence_alignmentjoins','measuredcellular_response_or_imagepair','parent_homology_taxon_heldout_proof','native_model_nucleotideprior_reader']+(['source_organism'] if organism is None else []))
