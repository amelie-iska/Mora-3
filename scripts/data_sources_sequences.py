"""Source-native biological sequence QC; annotation status and task admission separate."""
import hashlib

DNA_ALPHABET=set("ACGTRYSWKMBDHVN")
PROTEIN_ALPHABET=set("ACDEFGHIKLMNPQRSTVWYBXZJUO*")
_COMPLEMENT=str.maketrans("ACGTRYSWKMBDHVN","TGCAYRSWMKVHDBN")


def feature_alignment(parent,feature):
    start,end=feature.get("start"),feature.get("end")
    result=dict(source_start=start,source_end=end,coordinate_status="invalid",sequence_alignment="unavailable",orientation_evidence="source_derived_exact_alignment_not_observed_strand")
    if not isinstance(start,int) or not isinstance(end,int) or not 0<=start<end<=len(parent):return result
    result["coordinate_status"]="valid_zero_based_half_open"
    target=feature.get("sequence")
    if not isinstance(target,str) or not target:return result
    segment=parent[start:end].upper();value=target.upper();forward=value==segment;reverse=value==segment.translate(_COMPLEMENT)[::-1]
    result["sequence_alignment"]="forward_and_reverse_palindrome" if forward and reverse else "forward_exact" if forward else "reverse_complement_exact" if reverse else "mismatch"
    result["source_feature_length"]=len(target);result["coordinate_span_length"]=end-start
    return result


def dna_record_qc(row):
    sequence=row.get("sequence");valid_string=isinstance(sequence,str) and bool(sequence)
    seq=sequence.upper() if valid_string else "";invalid=sorted(set(seq)-DNA_ALPHABET)
    qc=dict(accession=row.get("accession"),source_parent_group="genbank:"+str(row.get("accession")),
        source_sequence_length=len(sequence) if valid_string else 0,
        source_sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest() if valid_string else None,
        uppercase_sequence_sha256=hashlib.sha256(seq.encode()).hexdigest() if valid_string else None,
        canonicalization="uppercase only for case-insensitive DNA identity; original source string immutable",
        dna_sequence_quality_pass=valid_string and not invalid,invalid_source_bases=invalid,
        observed_strand_available=False,protein_annotation_count=len(row.get("proteins") or []),
        protein_target_available=bool(row.get("proteins")),translation_task_admitted=False,
        translation_gates="CDS strand, compound/exon join, codon_start, genetic code and exact accession-source release verification are absent; do not infer contiguous translated target from bounding span",
        new_uniprot_sequence_admitted=False,task_admitted=False,feature_qc=[])
    for kind in ("introns","exons"):
        for i,feature in enumerate(row.get(kind) or []):
            qc["feature_qc"].append(dict(feature_type=kind,source_feature_index=i,**feature_alignment(sequence if valid_string else "",feature)))
    qc["protein_invalid_annotation_count"]=sum(bool(set(str(p.get("sequence","")).upper())-PROTEIN_ALPHABET) or not p.get("sequence") for p in row.get("proteins") or [])
    return qc
