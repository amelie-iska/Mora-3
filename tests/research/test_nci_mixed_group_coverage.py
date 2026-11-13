"""Test-only mixed-shard metadata/group admission, never a campaign trial."""
from copy import deepcopy
import json
import pytest

from mora3.research.nci_original_chemical_group_exposure import _read_mixed_nci_rows


FAM = {
    "conclusion": "NCI-AIDS-source-conclusion-prior",
    "endpoint": "NCI-AIDS-source-endpoint-prior",
}
TASK = {
    "conclusion": "source_NCI_assay_conclusion_prediction_prior",
    "endpoint": "source_NCI_assay_endpoint_generation_prior",
}


def sample(role="train"):
    path = f"outputs/data-intake/20261101-NCI-AIDS-original-v1/admitted-source-tasks-v1/shards/{role}/mixed.parquet"
    entries = [("conclusion", "a"), ("endpoint", "b"), ("conclusion", "c")]
    descriptor = {
        "source_shard": {"path": path, "source_family": None, "split": role, "rows": 3},
        "disposition": {"kind": "homogeneous-original-row-disposition", "rows": 3,
                        "code": {"train": 0, "validation": 1, "test": 2}[role]},
        "primary_retained_rows": 3,
        "source_task_schema_counts": [
            {"source_family": FAM["conclusion"], "task_name": TASK["conclusion"],
             "split": role, "native_schema": "mora3.CorpusTask/v1",
             "eligible_view_disposition": {"train": 0, "validation": 1, "test": 2}[role],
             "admitted_for_task": True, "rows": 2},
            {"source_family": FAM["endpoint"], "task_name": TASK["endpoint"],
             "split": role, "native_schema": "mora3.CorpusTask/v1",
             "eligible_view_disposition": {"train": 0, "validation": 1, "test": 2}[role],
             "admitted_for_task": True, "rows": 1},
        ],
    }
    original = [{
        "source_family": FAM[family], "task_name": TASK[family],
        "split": role, "admitted_for_task": True,
        "native_schema": "mora3.CorpusTask/v1", "task_id": label * 64,
    } for family, label in entries]
    sidecars = [{
        "task_id": label * 64,
        "effective_groups_json": json.dumps([
            "molecule:" + label * 64, "scaffold:" + label * 64
        ]),
    } for _, label in entries]
    return descriptor, original, sidecars


@pytest.mark.parametrize("role", ["train", "validation", "test"])
def test_mixed_shard_uses_original_row_family_and_role_not_null_shard(role):
    descriptor, original, sidecars = sample(role)
    result = _read_mixed_nci_rows(descriptor, original, sidecars, set())
    assert len(result[("NCI-conclusion", role)]) == 2
    assert len(result[("NCI-endpoint", role)]) == 1
    assert sum(map(len, result.values())) == 3


@pytest.mark.parametrize("failure", ["sidecar", "family", "role", "duplicate", "task"])
def test_mixed_shard_refuses_mismatched_source_and_group_metadata(failure):
    descriptor, original, sidecars = sample()
    if failure == "sidecar":
        sidecars[0]["task_id"] = "f" * 64
    elif failure == "family":
        original[0]["source_family"] = "OTHER"
    elif failure == "role":
        original[1]["split"] = "test"
    elif failure == "duplicate":
        original[2]["task_id"] = original[0]["task_id"]
        sidecars[2]["task_id"] = original[0]["task_id"]
    elif failure == "task":
        original[0]["task_name"] = TASK["endpoint"]
    with pytest.raises(ValueError):
        _read_mixed_nci_rows(descriptor, original, sidecars, set())


def test_mixed_shard_rejects_undisclosed_ordinal_exclusion():
    descriptor, original, sidecars = sample()
    with pytest.raises(ValueError):
        _read_mixed_nci_rows(descriptor, original, sidecars,
                             {(descriptor["source_shard"]["path"], 1)})
