"""Synthetic format witnesses; actual source replay is recorded separately."""
import hashlib
import io
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import pytest
from mora3.data import monuseg_native as native


def png(a):
    s=io.BytesIO(); Image.fromarray(a).save(s,format='PNG'); return s.getvalue()


@pytest.fixture
def source(tmp_path,monkeypatch):
    image=png(np.arange(36,dtype=np.uint8).reshape(3,4,3))
    masks=[png(np.array([[0,1,1,0],[0,1,0,0],[0,0,0,0]],dtype=np.uint8)),
           png(np.array([[0,255,0,0],[0,255,255,0],[0,0,0,0]],dtype=np.uint8))]
    row=dict(image={'bytes':image,'path':None},instances=[{'bytes':b,'path':None} for b in masks],patient='TCGA-test-slide',tissue=2)
    path=tmp_path/'test.parquet'; pq.write_table(pa.Table.from_pylist([row,row]),path)
    monkeypatch.setitem(native._SOURCES,'test',hashlib.sha256(path.read_bytes()).hexdigest())
    return path,image,masks


def test_order_overlap_context_and_admission(source):
    path,image,masks=source
    pair,=native.load_monuseg_pairs(path,split='test',row_indices=(1,))
    assert pair.image_encoded==image and pair.masks_encoded==tuple(masks)
    assert pair.encoding=='ordered_binary_instances' and pair.split=='test'
    arrays=pair.target_arrays()
    assert int((arrays[0]&arrays[1]).sum())==2
    assert [int(a.sum()) for a in arrays]==[3,3]
    assert dict(pair.image.context)['patient']=='TCGA-test-slide'
    assert pair.image.parent_id.endswith(':patient:TCGA-test-slide')
    assert pair.image.channels==('red','green','blue')
    with pytest.raises(ValueError,match='admission'): pair.require_training()


@pytest.mark.parametrize('indices',[(),(0,0),(True,),(-1,),(0,1,2),'0'])
def test_bounded_unique_integer_rows(source,indices):
    with pytest.raises(ValueError): native.load_monuseg_pairs(source[0],split='test',row_indices=indices)


def test_source_hash_and_split_refuse(source):
    path,_,_=source
    with pytest.raises(ValueError): native.load_monuseg_pairs(path,split='validation',row_indices=(0,))
    with path.open('ab') as s:s.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):native.load_monuseg_pairs(path,split='test',row_indices=(0,))


def test_missing_row_refuses(source):
    with pytest.raises(ValueError,match='row'):native.load_monuseg_pairs(source[0],split='test',row_indices=(2,))


def test_mask_targets_do_not_change_image_graph(source):
    from dataclasses import replace
    from mora3.model.color_semantic_inference import image_condition_graph
    pair,=native.load_monuseg_pairs(source[0],split='test',row_indices=(0,))
    changed=replace(pair,masks_encoded=tuple(reversed(pair.masks_encoded)),mask_sha256=tuple(reversed(pair.mask_sha256)))
    assert changed.binding_sha256!=pair.binding_sha256
    assert image_condition_graph(pair.image,pixel_sha256=pair.image_sha256)==image_condition_graph(changed.image,pixel_sha256=changed.image_sha256)
