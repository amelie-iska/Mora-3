import hashlib
import io
import zipfile
import numpy as np
import pytest
from PIL import Image
from mora3.data import deepbacs


def archive(tmp_path, *, mask=None):
    path=tmp_path/'fixture.zip'
    with zipfile.ZipFile(path,'w') as z:
        for split,folder in [('train','train/full_images'),('test','test')]:
            for kind,array in [('brightfield',np.arange(12,dtype=np.uint16).reshape(3,4)),
                               ('masks',np.array([[0,7,7,0],[0,0,42,0],[0,0,42,0]],dtype=np.uint16) if mask is None else mask)]:
                b=io.BytesIO(); Image.fromarray(array).save(b,format='TIFF')
                z.writestr(f'brightfield_dataset/{folder}/{kind}/parent.tif',b.getvalue())
    return path,hashlib.sha256(path.read_bytes()).hexdigest()


def test_preserves_labels_alignment_and_target_exclusion(tmp_path):
    path,sha=archive(tmp_path)
    rows=deepbacs.read_full_images(path,expected_sha256=sha)
    assert [r.split for r in rows]==['train','test']
    a=rows[0]; p=a.project()
    assert set(p['inputs'])=={'image'}
    assert p['inputs']['image'].shape==(3,4)
    assert p['targets']['labels'].tolist()==[[0,7,7,0],[0,0,42,0],[0,0,42,0]]
    assert a.instance_ids==(7,42)
    assert a.parent_group==rows[1].parent_group
    assert a.pixel_size_um==.08
    from dataclasses import replace
    changed=replace(a,mask_bytes=np.zeros((3,4),dtype=np.uint16).tobytes(),mask_dtype='<u2')
    assert np.array_equal(a.project()['inputs']['image'],changed.project()['inputs']['image'])
    with pytest.raises(ValueError,match='split'): rows[1].project(for_training=True)


def test_digest_and_alignment_fail_closed(tmp_path):
    path,sha=archive(tmp_path)
    with pytest.raises(ValueError,match='checksum'): deepbacs.read_full_images(path,expected_sha256='0'*64)
    path,sha=archive(tmp_path,mask=np.zeros((2,4),dtype=np.uint16))
    with pytest.raises(ValueError,match='aligned'): deepbacs.read_full_images(path,expected_sha256=sha)


def test_missing_pair_fails(tmp_path):
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('brightfield_dataset/test/brightfield/a.tif',b'bad')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='pair'): deepbacs.read_full_images(path,expected_sha256=sha)
