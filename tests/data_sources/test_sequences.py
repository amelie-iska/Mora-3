from pathlib import Path
import sys
import pytest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/"AGENTS.md").exists())
sys.path.insert(0,str(ROOT/"scripts"))
from data_sources_sequences import dna_record_qc, feature_alignment

def test_source_parent_coordinates_and_reverse_orientation_are_preserved():
    x=feature_alignment("AACGTTGC",{"start":2,"end":5,"sequence":"ACG"})
    assert x["coordinate_status"]=="valid_zero_based_half_open"
    assert x["sequence_alignment"]=="reverse_complement_exact"
    assert x["orientation_evidence"]=="source_derived_exact_alignment_not_observed_strand"

def test_wrong_coordinate_convention_is_not_silently_shifted():
    x=feature_alignment("AACGTTGC",{"start":3,"end":5,"sequence":"CGT"})
    assert x["sequence_alignment"]=="mismatch"
    assert x["source_start"]==3 and x["source_end"]==5

@pytest.mark.parametrize("start,end",[(-1,2),(1,99),(3,2)])
def test_out_of_bounds_annotation_is_flagged(start,end):
    assert feature_alignment("ACGT",dict(start=start,end=end,sequence="AC"))["coordinate_status"]=="invalid"

def test_missing_protein_does_not_invent_a_negative_or_translation_target():
    x=dna_record_qc(dict(accession="ABC.1",organism="unknown",sequence="ACGTN",introns=[],exons=[],proteins=[]))
    assert x["dna_sequence_quality_pass"] is True
    assert x["protein_annotation_count"]==0
    assert x["protein_target_available"] is False
    assert x["translation_task_admitted"] is False

def test_invalid_base_is_not_replaced_with_zero_or_unknown():
    x=dna_record_qc(dict(accession="ABC.1",sequence="AC?T",introns=[],exons=[],proteins=[]))
    assert x["dna_sequence_quality_pass"] is False
    assert x["invalid_source_bases"]==["?"]
