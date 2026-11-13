"""TEST-ONLY sciplex study/group exposure; never treats a study as a same-cell pair."""
from hashlib import sha256
import pytest

from mora3.research.sciplex_original_group_exposure import (
    study_group_exposure, audit_original_sciplex_study_groups,
)


def tid(s):
    return sha256(s.encode()).hexdigest()


def test_study_exposure_is_not_cell_identity_or_a_paired_biological_effect():
    group = "study:GSE139944:GSM4150377"
    left = ((tid("a"), (group,)), (tid("b"), (group,)))
    right = ((tid("c"), (group,)), (tid("d"), ("study:GSE139944:GSMother",)))
    r = study_group_exposure(left, right)
    assert r["original_retained_task_rows"] == {"sciplex_native_admitted": 2, "sciplex_native_expanded": 2}
    assert r["cross_population_study_components"] == 1
    assert r["cross_population_task_rows"] == {"sciplex_native_admitted": 2, "sciplex_native_expanded": 1}
    assert r["largest_cross_component"] == 3
    assert r["component_count"] == 2
    assert r["same_cell_pairings_proven"] is False
    assert r["heldout_split_independence_proven"] is False
    assert r["training_admission"] is False


def test_non_study_tokens_do_not_join_same_cell_or_response():
    left = ((tid("a"), ("chemical_identifier:ABC",)),)
    right = ((tid("b"), ("chemical_identifier:ABC",)),)
    r = study_group_exposure(left, right)
    assert r["cross_population_study_components"] == 0
    assert r["component_count"] == 2


@pytest.mark.parametrize("bad", ["study:", "study:unknown", "study:-", "study:None"])
def test_ambiguous_study_group_refuses(bad):
    with pytest.raises(ValueError):
        study_group_exposure(((tid("a"), (bad,)),), ((tid("b"), (bad,)),))


def test_repeated_task_ids_or_duplicate_tokens_refuse():
    key = "study:GSE1:GSM1"
    with pytest.raises(ValueError):
        study_group_exposure(((tid("a"), (key, key)),), ((tid("b"), (key,)),))
    with pytest.raises(ValueError):
        study_group_exposure(((tid("a"), (key,)),), ((tid("a"), (key,)),))


def test_actual_original_train_group_sidecars_without_native_targets():
    from pathlib import Path
    r = audit_original_sciplex_study_groups(Path.cwd())
    assert r["source_shards"] == {"sciplex_native_admitted": 4, "sciplex_native_expanded": 158}
    assert r["source_rows"] == r["retained_rows"] or all(
        r["source_rows"][k] >= r["retained_rows"][k] for k in r["source_rows"]
    )
    assert r["study_exposure"]["original_retained_task_rows"] == r["retained_rows"]
    assert r["retained_rows"] == {"sciplex_native_admitted": 1024, "sciplex_native_expanded": 40262}
    assert r["study_exposure"]["component_count"] == 1
    assert r["study_exposure"]["shared_study_tokens"] == 1
    assert r["study_exposure"]["largest_cross_component"] == 41286
    assert r["heldout_sources_opened"] is False
    assert r["source_native_values_opened"] is False
    assert r["optimizer_updates"] == r["actual_proxy_trials"] == r["actual_target_runs"] == 0
    assert r["training_admission"] is False
