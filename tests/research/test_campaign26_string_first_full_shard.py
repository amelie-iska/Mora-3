"""Original full-first-shard STRING: fixed native route, batched equivalence, independent bytes."""
from pathlib import Path

from mora3.research.campaign26_string_first_full_shard import audit_string_first_shard

ACTIVE = Path("/home/iska/Documents/amelie/bio/Mora-3")


def test_complete_first_original_string_shard_with_native_cursor_and_independent_parquet():
    evidence = audit_string_first_shard(ACTIVE)
    assert evidence["schema"] == "Mora3.Campaign26StringFirstFullShard/v1"
    assert evidence["native_first_shard_exhausted"] is True
    assert evidence["original_train_rows"] > 2
    assert evidence["original_train_rows"] == evidence["independent_original_train_rows"]
    assert evidence["native_batch1_equal_batch64_equal_parquet"] is True
    assert evidence["frozen_candidate_common_first_two_equal"] is True
    assert evidence["native_batch1_midshard_cursor_recovered"] is True
    assert evidence["original_native_score_target_positions"] == 4 * evidence["original_train_rows"]
    assert evidence["computed_association_not_experimental_binding"] is True
    assert evidence["original_validation_rows"] == evidence["original_test_rows"] == 0
    assert evidence["optimizer_updates"] == evidence["counted_proxy_trials"] == 0
    assert evidence["training_admission"] is False
