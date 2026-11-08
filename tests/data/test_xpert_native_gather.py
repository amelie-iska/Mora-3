"""Original ordinal gathers retain bits and verify each touched chunk once."""
import numpy as np
import pytest
from test_xpert_mmap import save_pair, canonical
from mora3.data import xpert_mmap as api


def make(tmp_path):
    a=np.array([0.,-0.,np.nan,np.inf,-np.inf,1.25,3.,4.],dtype='<f4')
    d=save_pair(tmp_path,a,chunk_rows=3)
    return a,d,api.NativeMmapArray(root=tmp_path,descriptor_bytes=canonical(d))


def test_batched_gather_preserves_native_bits_masks_order_and_single_chunk_io(tmp_path):
    from mora3.data.xpert_native_gather import gather_original_ordinals
    a,d,array=make(tmp_path)
    rows=(np.array([0,2,5,7],dtype='<u4'),np.array([1,2,6],dtype='<u4'),np.array([],dtype='<u4'))
    payload=sum(c['numeric_bytes']+c['validity_bytes'] for c in d['chunks'])
    results,audit=gather_original_ordinals(array,rows,max_payload_bytes=payload)
    for (raw,valid),row in zip(results,rows):
        assert raw==a[row].tobytes()
        assert valid==np.isfinite(a[row]).tobytes()
    assert audit['selected_chunk_indices']==[0,1,2]
    assert audit['numeric_payload_bytes']==a.nbytes
    assert audit['validity_payload_bytes']==3
    assert audit['output_value_bytes']==28
    assert audit['whole_file_hash_recomputed'] is False

@pytest.mark.parametrize('selection', [(),(np.array([2,1],dtype='<u4'),),(np.array([1,1],dtype='<u4'),),(np.array([8],dtype='<u4'),),(np.array([0],dtype='<i4'),),(np.array([0],dtype='<u4'),)*4])
def test_bad_or_unbounded_selectors_refused(tmp_path,selection):
    from mora3.data.xpert_native_gather import gather_original_ordinals
    _,_,a=make(tmp_path)
    with pytest.raises(ValueError):gather_original_ordinals(a,selection,max_payload_bytes=100)

@pytest.mark.parametrize('budget',[True,0,-1,3*1024**3+1,1])
def test_explicit_io_budget_refused_before_mapping(tmp_path,budget,monkeypatch):
    from mora3.data.xpert_native_gather import gather_original_ordinals
    _,_,a=make(tmp_path)
    def forbidden(*args):raise AssertionError('mapped before budget check')
    monkeypatch.setattr(api,'_mapped_bytes',forbidden)
    with pytest.raises(ValueError):gather_original_ordinals(a,(np.array([0],dtype='<u4'),),max_payload_bytes=budget)

@pytest.mark.parametrize('target',['numeric_sha256','validity_sha256'])
def test_chunk_integrity_not_weakened(tmp_path,target):
    from mora3.data.xpert_native_gather import gather_original_ordinals
    _,d,_=make(tmp_path);d['chunks'][0][target]='0'*64
    a=api.NativeMmapArray(root=tmp_path,descriptor_bytes=canonical(d))
    with pytest.raises(ValueError,match='integrity'):gather_original_ordinals(a,(np.array([0],dtype='<u4'),),max_payload_bytes=100)


def test_bound_index_exposes_new_explicit_row_gather_only():
    from mora3.data.xpert_ordinal_index import XPertNormanOrdinalIndex
    assert callable(getattr(XPertNormanOrdinalIndex,'read_stored_rows',None))

@pytest.mark.parametrize('wrong_indices,budget,expected',[(False,1000,None),(True,1000,'row indices'),(False,1,'combined')])
def test_row_binding_checks_original_indices_and_combined_budget(tmp_path,monkeypatch,wrong_indices,budget,expected):
    from mora3.data import xpert_ordinal_index as mod
    from mora3.data.xpert_mmap import _canonical
    data=np.array([-0.,np.nan,2.,np.inf],dtype='<f4')
    # Separate subdirectories because save_pair uses fixed filenames.
    v=tmp_path/'values';v.mkdir();ix=tmp_path/'indices';ix.mkdir()
    vd=save_pair(v,data,chunk_rows=2)
    rowindices=np.array([0,1,0,1],dtype='<i4')
    if wrong_indices:rowindices[2]=1
    id=save_pair(ix,rowindices,chunk_rows=2)
    # Use actual checked file descriptors rooted in each fixture directory.
    original_constructor=mod.NativeMmapArray
    monkeypatch.setattr(mod,'NativeMmapArray',lambda *,root,descriptor_bytes:original_constructor(
        root=v if b'"<f4"' in descriptor_bytes else ix,descriptor_bytes=descriptor_bytes))
    monkeypatch.setattr(mod,'transport_descriptor',_canonical)
    reader=object.__new__(mod.XPertNormanOrdinalIndex);reader.root=tmp_path
    from types import SimpleNamespace
    reader.store=SimpleNamespace(_array=lambda manifest,key: mod.NativeMmapArray(root=tmp_path,descriptor_bytes=canonical(vd if key=='values' else id)))
    ordinals=tuple(mod.NormanRowOrdinals(r,np.array(os,dtype='<u4').tobytes(),
        np.array([0,1],dtype='<i4').tobytes(),_canonical({'context_bound':False}))
        for r,os in [(1,[1,3]),(0,[0,2])])
    monkeypatch.setattr(reader,'read_row_offsets',lambda rows:ordinals)
    monkeypatch.setattr(reader,'_snapshot',lambda:({}, {}, {'native_stored_matrix_values':4,'arrays':{'data':'values','indices':'indices'}}))
    if expected:
        with pytest.raises(ValueError,match=expected):reader.read_stored_rows((1,0),max_payload_bytes=budget)
    else:
        result=reader.read_stored_rows((1,0),max_payload_bytes=budget)
        assert [r.native_row_index for r in result]==[1,0]
        for r,o in zip(result,([1,3],[0,2])):
            assert r.values_bytes==data[o].tobytes()
            assert r.finite_mask==np.isfinite(data[o]).tobytes()
            assert not r.native_values().flags.writeable
            assert r.audit()['selected_original_row_indices_rechecked'] is True
            with pytest.raises(ValueError,match='admission'):r.require_model_use('training')
