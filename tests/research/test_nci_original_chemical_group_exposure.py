"""NCI original-sidecar expansion of scoped seven-family chemical split exposure."""
from pathlib import Path
import pytest

from mora3.research.nci_original_chemical_group_exposure import (
    summarize_group_exposure, audit_original_nci_chemical_groups,
)


def row(task, molecule, scaffold):
    return {"task_id": task, "molecule": "molecule:" + molecule * 64,
            "scaffold": "scaffold:" + scaffold * 64}


def test_cross_source_heldout_train_and_heldout_heldout_detected():
    roles = {
        "AZ": {"train": (row("a", "a", "1"),)},
        "NCI-conclusion": {
            "train": (row("n", "b", "2"),),
            "validation": (row("v", "a", "3"),),
            "test": (row("t", "c", "3"),),
        },
    }
    result = summarize_group_exposure(roles)
    assert result["train_heldout_components"] == 1
    assert result["cross_source_train_heldout_components"] == 1
    assert result["heldout_heldout_shared_components"] == 1
    assert result["heldout_training_pairs"]["NCI-conclusion.validation|AZ.train"]["molecule"] == 1
    assert result["heldout_training_pairs"]["NCI-conclusion.validation|NCI-conclusion.train"]["molecule"] == 0
    assert result["global_split_admitted"] is False


def test_source_specific_nsc_labels_are_not_sufficient_to_link_chemical_records():
    roles = {
        "NCI-conclusion": {"train": (row("a", "a", "1"),)},
        "FreeSolv": {"validation": (row("b", "b", "2"),)},
    }
    result = summarize_group_exposure(roles)
    assert result["train_heldout_components"] == 0
    assert result["global_split_admitted"] is False


def test_duplicate_ids_and_invalid_axes_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        summarize_group_exposure({"X": {"train": (row("a", "a", "1"),)},
                                  "Y": {"test": (row("a", "b", "2"),)}})
    with pytest.raises(ValueError, match="group"):
        summarize_group_exposure({"X": {"train": (
            {"task_id": "bad", "molecule": "scaffold:" + "a" * 64,
             "scaffold": "scaffold:" + "b" * 64},)}})


def test_original_nci_groups_cross_source_without_reading_heldout_targets():
    result = audit_original_nci_chemical_groups(Path.cwd())
    assert result["nci_original_shards"] == {
        "NCI-conclusion": {"train": 69, "validation": 7, "test": 7},
        "NCI-endpoint": {"train": 31, "validation": 4, "test": 4},
        "NCI-profile": {"train": 2, "validation": 1, "test": 1},
    }
    assert result["nci_group_readback"]["original_shards"] == 120
    assert result["nci_group_readback"]["mixed_metadata_shards"] == 6
    assert result["nci_group_readback"]["original_rows"] == 120774
    assert result["nci_group_readback"]["original_rows"] == result["nci_group_readback"]["retained_rows"]
    assert result["source_numeric_shards_read"] is False
    assert result["heldout_targets_decoded"] is False
    assert result["source_group_hash_domains_independently_equated"] is False
    assert result["global_split_admitted"] is False
    assert result["optimizer_updates"] == result["actual_campaign_runs"] == 0
    assert result["training_admission"] is False


def test_validation_test_is_separated_from_same_split_heldout_overlap():
    roles = {
        "NCI-conclusion": {
            "validation": (row("v1", "a", "1"),),
            "test": (row("t1", "b", "1"),),
        },
        "NCI-endpoint": {"validation": (row("v2", "c", "2"),)},
    }
    result = summarize_group_exposure(roles)
    assert result["validation_test_mixed_components"] == 1
    assert result["cross_source_validation_test_mixed_components"] == 0
    assert result["heldout_heldout_shared_components"] == 1
