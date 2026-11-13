"""Real train-only native first-window conservation, not mock provenance.

The native source reference is immutable, but source eligibility is still open.
This diagnostic never enters the optimizer or a campaign run budget.
"""
import hashlib
from pathlib import Path
import pytest

from mora3.research.common_source_native_replay import (
    audit_common_source_first_prior_window,
)

ROOT = Path('/home/iska/Documents/amelie/bio/Mora-3')


def test_actual_native_first_window_cross_snapshot_conservation():
    receipt = audit_common_source_first_prior_window(ROOT)
    assert receipt['schema'] == 'Mora3.CommonSourceFirstNativeWindow/v1'
    assert receipt['task_key'] == 'native-prior:dna'
    assert receipt['role'] == 'train'
    assert receipt['windows_replayed'] == 1
    assert receipt['actual_native_rows'] == 1
    assert receipt['native_payload_bytes_read'] > 0
    assert receipt['observed_train_target_tokens'] > 0
    assert receipt['original_rows_and_payloads_identical'] is True
    assert receipt['original_native_source_digest'] == receipt['common_native_source_digest']
    assert receipt['candidate_adapter_sha256'] != receipt['common_adapter_sha256']
    assert receipt['candidate_initial_cursor_sha256'] != receipt['common_initial_cursor_sha256']
    assert receipt['training_admission'] is False
    assert receipt['full_replay_complete'] is False
    assert receipt['source_global_split_accepted'] is False
    assert receipt['optimizer_updates'] == 0
    assert receipt['actual_campaign_runs'] == 0


@pytest.mark.parametrize('task_key,max_windows',[
    ('native-prior:protein',1),('native-prior:dna',0),('native-prior:dna',2),
    ('source_role_list_reconstruction_on_current_reference',1),
])
def test_refuses_other_route_or_changed_replay_budget_before_source_io(task_key,max_windows):
    with pytest.raises(ValueError):
        audit_common_source_first_prior_window(ROOT,task_key=task_key,max_windows=max_windows)


def test_independently_hashed_native_payload_and_original_role_refuse_modified_bytes():
    from types import SimpleNamespace
    import pyarrow as pa
    from mora3.research.common_source_native_replay import _source_fingerprint

    def window(data,declared):
        return SimpleNamespace(batch=SimpleNamespace(original_split='train',
            source_shard=dict(path='immutable-original-train-shard',sha256='f'*64,rows=1),
            source_row_ordinals=(0,),native_rows=pa.table(dict(split=['train'],
                task_id=['original-task'],native_payload=[data],native_payload_sha256=[declared]))))
    old=window(b'source-byte-A',hashlib.sha256(b'source-byte-A').hexdigest())
    before=_source_fingerprint(old)
    with pytest.raises(ValueError,match='digest'):
        _source_fingerprint(window(b'source-byte-B',hashlib.sha256(b'source-byte-A').hexdigest()))
    new=window(b'source-byte-B',hashlib.sha256(b'source-byte-B').hexdigest())
    assert before[0]!=_source_fingerprint(new)[0]
    changed=window(b'source-byte-A',hashlib.sha256(b'source-byte-A').hexdigest())
    changed.batch.original_split='validation'
    with pytest.raises(ValueError,match='train'):_source_fingerprint(changed)
