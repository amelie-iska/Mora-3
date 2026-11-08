"""Native Bernett benchmark roles; sampled negatives are not measured absences."""
import hashlib
import re
from data_sources_goa import ACCESSION

SPLITS={"0":"validation","1":"train","2":"test"}
AMINO_ACIDS=frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")

def source_pair(line, filename, line_index):
    match=re.fullmatch(r"Intra([012])_(pos|neg)_rr\.txt",filename)
    fields=line.split()
    if match is None or len(fields)!=2 or any(not ACCESSION.fullmatch(x) for x in fields):
        raise ValueError("expected exact source filename and two canonical accession tokens")
    if not isinstance(line_index,int) or line_index<1:raise ValueError("native line index must be positive")
    block,kind=match.groups();a,b=fields
    return dict(source_record_id="figshare21591618v3:"+filename+":"+str(line_index),source_file=filename,source_line_index=line_index,source_block="Intra"+block,source_native_split=SPLITS[block],protein1_accession=a,protein2_accession=b,undirected_accession_pair_key="uniprot-accession-pair:"+"|".join(sorted(fields)),benchmark_label=int(kind=="pos"),label_role="source_benchmark_target",evidence_class="curated_HIPPIEv2.3_positive_pair" if kind=="pos" else "algorithmically_sampled_negative_pair",experimental_noninteraction=False,direct_binding_proven=False,task_admitted=False,model_adapter_supported=False)

def protein_source_record(header,sequence):
    if not isinstance(header,str) or not header or not isinstance(sequence,str):raise ValueError("native FASTA header/sequence required")
    source_id=header.split()[0]
    return dict(native_source_id=source_id,native_header=header,native_sequence=sequence,native_length=len(sequence),native_sequence_sha256=hashlib.sha256(sequence.encode("ascii")).hexdigest(),canonical_accession_syntax=bool(ACCESSION.fullmatch(source_id)),native_alphabet_pass=bool(sequence) and set(sequence)<=AMINO_ACIDS,source_SwissProt_release=None,task_admitted=False,model_adapter_supported=False)

def qualified_sequence_disposition(accession,sequence,baseline):
    entry=baseline.get(accession);reason=None
    if not ACCESSION.fullmatch(accession):reason="source_form_or_namespace_needs_explicit_binding"
    elif entry is None:reason="not_in_exact_qualified_current_entry_baseline"
    elif entry.get("reviewed") is not True:reason="current_entry_not_verified_reviewed"
    else:
        try:score=float(entry["annotation_score"])
        except (KeyError,ValueError,TypeError):score=float("nan")
        if not 4<=score<=5:reason="current_annotation_score_missing_or_below4"
        elif entry.get("sequence")!=sequence:reason="exact_current_entry_sequence_mismatch"
    return dict(exact_qualified_current_entry_sequence_candidate=reason is None,disposition="candidate_current_exact_entry_sequence" if reason is None else "quarantine_current_sequence_binding",reason=reason,historical_SwissProt_release_equivalence_asserted=False,task_admitted=False)


def unique_source_sequence(source_id,source_by_id):
    """Resolve only a unique native FASTA occurrence without creating absent IDs."""
    records=source_by_id.get(source_id,[])
    return records[0] if len(records)==1 else None
