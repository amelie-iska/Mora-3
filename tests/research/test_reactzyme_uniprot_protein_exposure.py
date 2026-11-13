"""Original ReactZyme/UniProt biological-exposure regression; fixtures only in this test."""
from pathlib import Path
import pytest

from mora3.research.reactzyme_uniprot_protein_exposure import (
    protein_exposure, audit_original_reactzyme_uniprot,
)

SEQ_A = "sequence:" + "a"*64
SEQ_B = "sequence:" + "b"*64
GUARD = "detected_protein_guard:8567fe0c29bb39a0:qualified_MMseqs_linclust30cov80:" + "a"*64

def test_distinct_homology_and_exact_identity_types_not_merged_with_reactions():
    a = [
        ("a1", ("accession:P12345", SEQ_A, GUARD, "rhea:RHEA:100")),
        ("a2", ("accession:P56789", SEQ_B)),
    ]
    b = [
        ("b1", ("accession:P12345", SEQ_A, GUARD)),
        ("b2", ("accession:Q99999", "sequence:" + "c"*64, GUARD)),
        ("b3", ("rhea:RHEA:100", "molecule:" + "f"*64)),
    ]
    r=protein_exposure(a,b)
    assert r["task_rows"]=={"ReactZyme":2,"UniProt":3}
    assert r["shared_group_counts"]=={"accession":1,"sequence":1,"qualified_homology_guard":1}
    assert r["cross_source_components"]==1
    assert r["shared_accessions_with_equal_sequence_groups"]==1
    assert r["shared_accessions_without_equal_sequence_groups"]==0
    assert r["largest_cross_component"]==3
    assert r["cross_source_member_rows"]=={"ReactZyme":1,"UniProt":2}
    assert r["global_evaluation_split_admitted"] is False

@pytest.mark.parametrize("x", ["accession:-", "accession:unknown", "rhea:RHEA:1", "molecule:"+"a"*64])
def test_missing_and_nonprotein_keys_cannot_form_false_cross_source_identity(x):
    r=protein_exposure([("a", (x,))], [("b", (x,))])
    assert r["cross_source_components"]==0
    assert r["shared_group_counts"]=={"accession":0,"sequence":0,"qualified_homology_guard":0}

def test_same_accession_without_equal_sequence_is_group_exposure_not_exact_identity():
    r=protein_exposure([("a", ("accession:P12345", SEQ_A))],
                       [("b", ("accession:P12345", SEQ_B))])
    assert r["shared_group_counts"]["accession"] == 1
    assert r["shared_group_counts"]["sequence"] == 0
    assert r["shared_accessions_with_equal_sequence_groups"] == 0
    assert r["shared_accessions_without_equal_sequence_groups"] == 1
    assert r["cross_source_components"] == 1


def test_duplicate_native_ids_rejected():
    with pytest.raises(ValueError,match="Duplicate"):
        protein_exposure([("a", (SEQ_A,)),("a", (SEQ_B,))],[("b",(SEQ_A,))])

def test_original_source_exposure_without_any_target_or_optimizer():
    result=audit_original_reactzyme_uniprot(Path.cwd())
    assert result["original_shards"]=={"ReactZyme":161,"UniProt":407}
    assert result["original_rows"]=={"ReactZyme":41156,"UniProt":104176}
    assert result["retained_rows"]=={"ReactZyme":40464,"UniProt":103171}
    assert result["exposure"]["task_rows"]==result["retained_rows"]
    assert result["heldout_targets_opened"] is False
    assert result["optimizer_updates"]==result["actual_proxy_trials"]==result["actual_target_runs"]==0
    assert result["training_admission"] is False
