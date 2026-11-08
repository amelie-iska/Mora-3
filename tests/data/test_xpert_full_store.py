"""Actual pinned store tests; numeric transport is not task admission."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from mora3.data import xpert_mmap
from mora3.data import xpert_full_store

ROOT = Path(__file__).resolve().parents[2]


def source():
    assert hasattr(xpert_full_store, 'XPertFullNumericStore'), 'explicit full-store binding missing'
    return xpert_full_store.XPertFullNumericStore(root=ROOT)


@pytest.mark.parametrize('key,start,stop', [
    ('32221302-X', 590*978, 591*978),
    ('32221275-X-data', 0, 128),
    ('32221275-X-indices', 0, 128),
    ('32221275-X-indptr', 0, 33695),
])
def test_actual_native_bits_validity_offsets_and_closed_model_gate(key,start,stop):
    store = source()
    result = store.read_storage(key,start,stop)
    manifest = json.loads((ROOT/store.MANIFEST['path']).read_bytes())
    record = manifest['array_transports'][key]['transport']
    oracle = np.load(ROOT/record['path'], mmap_mode='r', allow_pickle=False).reshape(-1)[start:stop]
    assert result.values_bytes == oracle.tobytes()
    assert result.finite_mask == np.isfinite(oracle).tobytes()
    audit = result.audit()
    assert audit['full_store_manifest_sha256'] == store.MANIFEST['sha256']
    assert audit['source_identity_verified'] is False
    assert audit['context_bound'] is False
    assert audit['task_admitted'] is False
    with pytest.raises(ValueError): result.require_model_use('training')


def test_fixture_cannot_silently_accept_full_store():
    store=source()
    with pytest.raises(ValueError):
        xpert_mmap.XPertMmapSource(root=ROOT, manifest=store.MANIFEST['path'],
            expected_manifest_sha256=store.MANIFEST['sha256'])


@pytest.mark.parametrize('key,start,stop', [('missing',0,1),('32221302-X',True,2),('32221302-X',0,2_000_001)])
def test_invalid_or_unbounded_storage_refused(key,start,stop):
    with pytest.raises(ValueError): source().read_storage(key,start,stop)


def test_no_alternative_release_binding():
    cls=type(source())
    with pytest.raises(TypeError): cls(root=ROOT, expected_manifest_sha256='0'*64)


def test_selection_revalidates_handoff_before_numeric_access(tmp_path):
    cls = type(source())
    for ref in (cls.MANIFEST, cls.HANDOFF):
        target = tmp_path/ref['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/ref['path']).read_bytes())
    store = cls(root=tmp_path)
    handoff = tmp_path/cls.HANDOFF['path']
    handoff.write_bytes(handoff.read_bytes()+b' ')
    with pytest.raises(ValueError, match='immutable proof mismatch'):
        store.read_storage('32221302-X',0,978)


def test_full_dense_rows_keep_complete_native_axis_and_caller_order():
    store=source()
    rows=store.read_dense_rows((590,0))
    m=json.loads((ROOT/store.MANIFEST['path']).read_bytes())
    oracle=np.load(ROOT/m['array_transports']['32221302-X']['transport']['path'],mmap_mode='r',allow_pickle=False)
    assert tuple(r.native_row_index for r in rows)==(590,0)
    for r in rows:
        assert r.axis_length==978
        assert r.values_bytes==oracle[r.native_row_index].tobytes()
        assert r.profile().source_positions()==tuple(range(r.native_row_index*978,(r.native_row_index+1)*978))
        assert r.values.audit()['context_bound'] is False
        with pytest.raises(ValueError): r.require_model_use('training')


@pytest.mark.parametrize('column',[0,33693])
def test_actual_csc_columns_native_bits_order_offsets_and_implicit_zero(column):
    store=source(); result,=store.read_csc_columns((column,))
    m=json.loads((ROOT/store.MANIFEST['path']).read_bytes())
    def raw(key):
        return np.load(ROOT/m['array_transports'][key]['transport']['path'],mmap_mode='r',allow_pickle=False)
    ptr=raw('32221275-X-indptr'); start,stop=map(int,ptr[column:column+2])
    values=raw('32221275-X-data')[start:stop]; indices=raw('32221275-X-indices')[start:stop]
    assert result.values_bytes==values.tobytes()
    assert result.values.finite_mask==np.isfinite(values).tobytes()
    assert result.indices.values_bytes==indices.tobytes()
    assert result.row_indices()==tuple(map(int,indices))
    assert result.storage_positions()==tuple(range(start,stop))
    assert result.logical_rows==111445
    assert result.coordinate_unique and result.strictly_ascending
    assert result.implicit_observed_zero_count==111445-len(indices)
    assert result.implicit_zero_is_missing is False
    assert result.audit()['coordinate_proof_sha256']=='97a72b164af4627f516e1088751f78c8706c7b1e4858a0a514432ff70aa25260'
    assert result.audit()['context_bound'] is False
    for purpose in ('training','prediction','generation','DE_statistics'):
        with pytest.raises(ValueError): result.require_model_use(purpose)


@pytest.mark.parametrize('method,selectors',[
    ('read_dense_rows',()),('read_dense_rows',(True,)),('read_dense_rows',(678401,)),
    ('read_dense_rows',(0,0)),('read_dense_rows',[0]),
    ('read_csc_columns',()),('read_csc_columns',(True,)),('read_csc_columns',(33694,)),
    ('read_csc_columns',(0,0)),('read_csc_columns',tuple(range(17)))])
def test_full_native_selector_bounds(method,selectors):
    with pytest.raises(ValueError): getattr(source(),method)(selectors)


def test_csc_coordinate_proof_revalidated_before_selection(tmp_path):
    cls=type(source())
    m=json.loads((ROOT/cls.MANIFEST['path']).read_bytes())
    d=next(d for d in m['datasets'] if d['matrix_encoding']=='csc_matrix')
    proof=d['csc_full_original_coordinate_proof']
    # Full manifest uses original absolute project paths; use the same relative
    # proof under the isolated root, never mutate source evidence.
    for ref in (cls.MANIFEST,cls.HANDOFF):
        p=tmp_path/ref['path']; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((ROOT/ref['path']).read_bytes())
    rel=Path(proof['path']).relative_to(ROOT)
    p=tmp_path/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(proof['path']).read_bytes()+b' ')
    with pytest.raises(ValueError,match='immutable proof mismatch'):
        cls(root=tmp_path).read_csc_columns((0,))
