"""Unsigned access metadata and original CSC coordinate references, not training data."""
import json
import numpy as np
import pytest
from mora3.data import xpert_mmap as mmap_api
from test_xpert_mmap import save_pair, canonical

@pytest.mark.parametrize('dtype', ('<u4', '<u8'))
def test_unsigned_metadata_preserves_native_bits_and_chunk_proofs(tmp_path, dtype):
    values = np.array([0, 1, 2**32-1, 7], dtype=dtype)
    descriptor = save_pair(tmp_path, values, chunk_rows=3)
    array = mmap_api.NativeMmapArray(root=tmp_path, descriptor_bytes=canonical(descriptor))
    result = array.read_elements(1, 4)
    assert result.values_bytes == values[1:4].tobytes()
    assert result.finite_mask == b'\1'*3
    assert result.native_dtype == dtype
    assert not result.native_values().flags.writeable


def test_coordinate_reference_keeps_repeated_empty_columns_and_original_ordinals():
    from mora3.data.xpert_ordinal_index import original_gene_coordinates
    ptr = np.array([0, 0, 2, 2, 4, 5], dtype='<i4')
    offsets = np.array([0, 2, 4], dtype='<u4')
    assert original_gene_coordinates(ptr, offsets, 5).tolist() == [1, 3, 4]

@pytest.mark.parametrize('ptr, offsets, width', [
    ([1, 2, 3], [1], 2), ([0, 3, 2], [1], 2),
    ([0, 2, 4], [4], 2), ([0, 2, 4], [1, 1], 2),
    ([0, 2, 4], [3, 1], 2), ([0, 2, 4], [0, 1], 2),
    ([0, 2, 4], [1], True),
])
def test_invalid_offset_order_extent_or_duplicate_gene_refused(ptr, offsets, width):
    from mora3.data.xpert_ordinal_index import original_gene_coordinates
    with pytest.raises(ValueError):
        original_gene_coordinates(np.array(ptr, dtype='<i4'), np.array(offsets, dtype='<u4'), width)


def test_index_binding_is_explicit_and_candidate_only():
    from mora3.data.xpert_ordinal_index import XPertNormanOrdinalIndex
    assert XPertNormanOrdinalIndex.HANDOFF['sha256'] == 'f274acd5c95ac2de380473e4307437147e21151d2f7308911df26774be27b7f5'
    assert XPertNormanOrdinalIndex.MANIFEST['sha256'] == '4315b32d793d62a3f8003f4fddcde86552963bba5e2f130e1a233d348fd2b55e'
    with pytest.raises(ValueError, match='admission'):
        XPertNormanOrdinalIndex.require_model_use(None, 'training')

@pytest.mark.parametrize('rows', ((), (True,), (-1,), (111445,), (0,0), (0,1,2,3), [0]))
def test_actual_reader_refuses_invalid_or_unbounded_rows(rows):
    from pathlib import Path
    from mora3.data.xpert_ordinal_index import XPertNormanOrdinalIndex
    reader = XPertNormanOrdinalIndex(root=Path(__file__).resolve().parents[2])
    with pytest.raises(ValueError): reader.read_row_offsets(rows)


def test_actual_three_rows_match_independent_offsets_and_all_gene_intervals():
    import bisect
    from pathlib import Path
    from mora3.data.xpert_ordinal_index import XPertNormanOrdinalIndex
    root = Path(__file__).resolve().parents[2]
    reader = XPertNormanOrdinalIndex(root=root)
    rows = reader.read_row_offsets((590,0,1))
    m = json.loads((root/reader.MANIFEST['path']).read_bytes())
    rp = np.load(m['array_transports']['row_indptr']['transport']['path'], mmap_mode='r', allow_pickle=False)
    ro = np.load(m['array_transports']['row_source_element_ordinal']['transport']['path'], mmap_mode='r', allow_pickle=False)
    cp = np.load(m['source_spec']['arrays']['indptr']['transport']['path'], mmap_mode='r', allow_pickle=False)
    for value in rows:
        r=value.native_row_index
        expected=ro[int(rp[r]):int(rp[r+1])]
        assert value.original_offsets_bytes == expected.tobytes()
        assert value.gene_coordinates().tolist() == [bisect.bisect_right(cp,int(o))-1 for o in expected]
        assert not value.offsets().flags.writeable
        assert value.audit()['complete_gene_axis_length'] == 33694
        assert value.audit()['expression_values_read'] is False
        assert value.audit()['selected_original_row_indices_rechecked'] is False
        for purpose in ('training','prediction','generation','DE_statistics'):
            with pytest.raises(ValueError, match='admission'): value.require_model_use(purpose)


def test_new_index_handoff_revalidated_each_selection(monkeypatch):
    from pathlib import Path
    from mora3.data.xpert_ordinal_index import XPertNormanOrdinalIndex
    reader = XPertNormanOrdinalIndex(root=Path(__file__).resolve().parents[2])
    monkeypatch.setattr(reader, 'HANDOFF', dict(reader.HANDOFF,sha256='0'*64))
    with pytest.raises(ValueError, match='proof mismatch'): reader.read_row_offsets((0,))


def test_unsigned_chunk_tamper_refused(tmp_path):
    values = np.array([0,1,2**32-1,7],dtype='<u4')
    d=save_pair(tmp_path,values)
    d['chunks'][0]['numeric_sha256']='0'*64
    array=mmap_api.NativeMmapArray(root=tmp_path,descriptor_bytes=canonical(d))
    with pytest.raises(ValueError,match='integrity'):array.read_elements(0,1)
