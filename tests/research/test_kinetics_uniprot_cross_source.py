"""Conservative original UniProt kinetics / sequence-prior exposure regressions."""
from mora3.research.cross_source_uniprot_kinetics import protein_group_closure
import pytest


def _r(family, task, groups):
    return dict(source=family, task_id=task*64, groups=tuple(groups))


def test_direct_sequence_and_homology_links_are_classed_not_experiments():
    kinetics=(_r("kinetics", "a", ("accession:A", "sequence:S1", "pubmed:P1")),
              _r("kinetics", "b", ("accession:B", "detected_protein_guard:H1")))
    sequence=(_r("sequence", "c", ("accession:A", "sequence:S1")),
              _r("sequence", "d", ("accession:D", "detected_protein_guard:H1")),
              _r("sequence", "e", ("accession:E",)))
    d=protein_group_closure(kinetics, sequence)
    assert d["cross_source_components"]==2
    assert d["kinetics_rows_cross_exposed"]==2
    assert d["sequence_rows_cross_exposed"]==2
    assert d["direct_cross_source_group_classes"]=={"accession":1,"sequence":1,"detected_protein_guard":1}
    assert d["global_split_admitted"] is False
    assert d["same_protein_measurement_claimed"] is False


def test_study_only_link_is_exposure_not_sequence_identity():
    kinetics=(_r("kinetics","a",("pubmed:P1",)),)
    sequence=(_r("sequence","b",("pubmed:P1",)),)
    d=protein_group_closure(kinetics,sequence)
    assert d["cross_source_components"]==1
    assert d["direct_cross_source_group_classes"]=={"pubmed":1}
    assert d["cross_source_identity_only_components"]==0


def test_rejects_unqualified_and_duplicate_sources():
    a=_r("kinetics","a",("accession:A",))
    b=_r("sequence","b",("accession:B",))
    with pytest.raises(ValueError):protein_group_closure((a,a),(b,))
    with pytest.raises(ValueError):protein_group_closure((dict(a,groups=("accession:A","accession:A")),),(b,))
    with pytest.raises(ValueError):protein_group_closure((dict(a,groups=("unscoped",)),),(b,))
    with pytest.raises(ValueError):protein_group_closure((dict(a,source="sequence"),),(b,))
