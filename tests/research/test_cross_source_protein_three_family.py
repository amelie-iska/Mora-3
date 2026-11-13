"""Three-family original protein grouping: synthetic regressions, no training."""
import pytest

from mora3.research.cross_source_protein_three_family import (
    combined_protein_closure,
)

def item(n, *groups):
    return (f"{n:064x}", tuple(groups))

def test_three_way_transitive_bridge_without_false_direct_pair():
    r = (item(1, "accession:A1"),)
    u = (item(2, "accession:A1", "sequence:" + "a" * 64),)
    k = (item(3, "sequence:" + "a" * 64),)
    out = combined_protein_closure(r, u, k)
    assert out["triple_family_components"] == 1
    assert out["largest_triple_component"] == 3
    assert out["indirect_reactzyme_kinetics_components"] == 1
    assert out["cross_family_component_rows"] == {"ReactZyme": 1, "UniProt": 1, "UniProt-kinetics": 1}
    assert out["direct_reactzyme_kinetics_group_counts"] == {"accession": 0, "sequence": 0, "qualified_homology_guard": 0}

def test_pubmed_study_exposure_does_not_assert_same_protein():
    out = combined_protein_closure(
        (item(11, "pubmed:12345"),),
        (item(12, "pubmed:12345"),),
        (item(13, "pubmed:12345"),),
    )
    assert out["triple_family_components"] == 0
    assert out["components"] == 3
    assert out["shared_study_tokens_all_three"] == 1
    assert out["cross_family_component_rows"] == {"ReactZyme": 0, "UniProt": 0, "UniProt-kinetics": 0}

@pytest.mark.parametrize("bad", [
    (item(22, "accession:A"), item(22, "sequence:" + "b" * 64)),
    (item(23, "accession:A", "accession:A"),),
    (item(24, "detected_protein_guard:bogus"),),
    (("not-a-hex-task-id", ("accession:A",)),),
])
def test_rejects_nonunique_and_unqualified_grouping(bad):
    if len(bad) == 2:
        args = ((bad[0],), (bad[1],), (item(31, "accession:C"),))
    else:
        args = (tuple(bad), (item(32, "accession:B"),), (item(33, "accession:C"),))
    with pytest.raises(ValueError):
        combined_protein_closure(*args)

def test_canonical_component_commitment_order_invariant():
    r = (item(40, "accession:X"), item(41, "accession:Y"))
    u = (item(42, "accession:Y"), item(43, "accession:X"))
    k = (item(44, "accession:X"), item(45, "sequence:" + "f" * 64))
    a = combined_protein_closure(r, u, k)
    b = combined_protein_closure(r[::-1], u[::-1], k[::-1])
    assert a["component_membership_sha256"] == b["component_membership_sha256"]
