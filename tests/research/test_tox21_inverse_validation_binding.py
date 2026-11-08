import json
from pathlib import Path
import pytest
from mora3.data.admet import canonical
from mora3.research.tox21_inverse_validation_binding import bind_pending_inverse_validation,_scope_check

@pytest.mark.parametrize('role',['train','test'])
def test_cursor_role_substitution_refused(role,real_binding):
    cursor=json.loads(real_binding.initial_cursor);cursor['role']=role
    with pytest.raises(ValueError,match='cursor'):tuple(real_binding.iter_from(canonical(cursor)))

@pytest.fixture(scope='module')
def real_binding():return bind_pending_inverse_validation(Path.cwd(),read_batch_size=32,max_selected_arrow_bytes=1000000)

def test_exact_real_validation_only_source_selection(real_binding):
    b=real_binding
    assert len(b.coordinates)==114 and b.observed_targets==5225
    rows=tuple(b.iter_from(b.initial_cursor))
    assert sum(len(batch.source_row_ordinals) for batch,_ in rows)==114
    assert not tuple(b.iter_from(rows[-1][1]))
    assert not b.complete_cohort and not b.training_authorized
    assert 'initial_reference_loss' in b.unresolved
    assert all(batch.original_split=='validation' for batch,_ in rows)

def test_no_promoted_complete_proof():
    with pytest.raises(ValueError):_scope_check({'complete_owner_snapshot_sha256':'a'*64})

def test_missing_budget_refused_before_read():
    with pytest.raises(ValueError):bind_pending_inverse_validation('/missing',read_batch_size=True,max_selected_arrow_bytes=1000)
