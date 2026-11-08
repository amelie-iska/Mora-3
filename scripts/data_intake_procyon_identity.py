"""Exact current whole-protein qualification and explicitly unproven domain links."""
import hashlib
from decimal import Decimal,InvalidOperation


def qualify_whole_protein(accession,sequence,baseline):
    digest=hashlib.sha256(sequence.encode()).hexdigest();entry=baseline.get(accession)
    result=dict(native_ID=accession,native_sequence_SHA256=digest,native_length=len(sequence),
        exact_accession_and_full_sequence_qualified=False,task_admitted=False,
        labels_or_annotation_score_transferred_by_sequence_similarity=False)
    if entry is None:return dict(result,status='exact_native_accession_absent_from_qualified_current_baseline')
    try:score=Decimal(entry['annotation_score'])
    except (InvalidOperation,TypeError):score=Decimal('-1')
    eligible=entry['reviewed']=='reviewed' and score.is_finite() and score>=4
    result.update(baseline_accession=accession,baseline_sequence_SHA256=entry['sequence_SHA256'],baseline_annotation_score=entry['annotation_score'],baseline_reviewed=entry['reviewed'],baseline_native_row=entry['native_row'])
    if not eligible:return dict(result,status='current_entry_reviewed_annotation_threshold_not_satisfied')
    if digest!=entry['sequence_SHA256']:return dict(result,status='exact_accession_present_but_full_sequence_mismatch')
    result.update(exact_accession_and_full_sequence_qualified=True,status='exact_accession_AND_current_canonical_sequence_and_annotation4plus')
    return result


def domain_prefix_span_candidate(native_ID,sequence,source_proteins,qualified):
    """Native suffix syntax is not source-proven parent/coordinate semantics."""
    prefix=native_ID.split('_',1)[0];parent=source_proteins.get(prefix);spans=[]
    if parent is not None and sequence:
        start=parent.find(sequence)
        while start!=-1:
            spans.append([start+1,start+len(sequence)])
            start=parent.find(sequence,start+1)
    return dict(native_domain_ID=native_ID,unverified_ID_prefix_parent_candidate=prefix,
        prefix_candidate_exists_in_source_whole_protein_FASTA=parent is not None,
        prefix_candidate_whole_protein_current_qualified=bool(qualified.get(prefix,{}).get('exact_accession_and_full_sequence_qualified')),
        derived_literal_substring_spans_1based_closed=spans,
        unique_exact_substring_candidate=len(spans)==1,
        native_parent_or_coordinate_metadata_verified=False,
        annotation_score_or_function_labels_inherited=False,whole_canonical_protein_qualification=False,
        domain_task_admitted=False,
        derived_span_scope='literal substring in source whole-protein candidate; no source-bound native domain parent, Pfam coordinate semantics or experimental residue scope asserted')
