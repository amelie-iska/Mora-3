"""Catch invented instance labels, cross-slide pairing and silent source changes."""
import hashlib
import io
import zipfile
import numpy as np
import pytest
from PIL import Image
from mora3.data import tnbc_native


def png(a):
    stream=io.BytesIO(); Image.fromarray(a).save(stream,format='PNG'); return stream.getvalue()


IMAGE='TNBC_NucleiSegmentation/Slide_01/01_1.png'
MASK='TNBC_NucleiSegmentation/GT_01/01_1.png'


def archive(tmp_path, *, mask=None, image=None, missing=False, extra=()):
    p=tmp_path/'source.zip'
    image=np.arange(36,dtype=np.uint8).reshape(3,4,3) if image is None else image
    mask=np.array([[0,255,255,0],[0,255,0,0],[0,0,0,255]],dtype=np.uint8) if mask is None else mask
    with zipfile.ZipFile(p,'w') as z:
        z.writestr(IMAGE,png(image))
        if not missing: z.writestr(MASK,png(mask))
        for name,raw in extra: z.writestr(name,raw)
    return p,hashlib.sha256(p.read_bytes()).hexdigest()


def load(p,sha,members=(IMAGE,)):
    return tnbc_native.load_tnbc_pairs(p, image_members=members, archive_sha256=sha)


def test_binary_nuclear_supervision_preserves_pixels_and_refuses_training(tmp_path):
    p,sha=archive(tmp_path); before=p.read_bytes(); x,=load(p,sha)
    assert x.encoding=='semantic_binary' and x.split=='evaluation'
    np.testing.assert_array_equal(x.image_array(),np.arange(36,dtype=np.uint8).reshape(3,4,3))
    np.testing.assert_array_equal(x.target_arrays()[0],[[False,True,True,False],[False,True,False,False],[False,False,False,True]])
    assert x.image.parent_id.endswith('slide-group:01')
    assert x.image.channels==('red','green','blue') and x.image.pixel_size_um is None
    context=dict(x.image.context)
    assert context['perturbation'] is None and context['expression'] is None
    assert context['object_type']=='nucleus' and context['source_split'] is None
    with pytest.raises(ValueError,match='admission'): x.require_training()
    with pytest.raises(ValueError,match='channel'): x.native_input()
    assert p.read_bytes()==before


def test_changed_target_cannot_change_visible_image(tmp_path):
    p,sha=archive(tmp_path); x,=load(p,sha)
    other=tmp_path/'other'; other.mkdir()
    p,sha=archive(other,mask=np.zeros((3,4),dtype=np.uint8)); y,=load(p,sha)
    assert x.image_array().tobytes()==y.image_array().tobytes()
    assert x.image==y.image  # target/archive hashes must not become predictor context
    assert x.binding_sha256!=y.binding_sha256


@pytest.mark.parametrize('members', [(),(IMAGE,IMAGE),('TNBC_NucleiSegmentation/Slide_02/01_1.png',),('../Slide_01/01_1.png',),(MASK,),tuple([IMAGE]*51)])
def test_ambiguous_or_unbounded_requests_fail(tmp_path,members):
    p,sha=archive(tmp_path)
    with pytest.raises(ValueError): load(p,sha,members)


def test_archive_checksum_is_required_and_checked(tmp_path):
    p,sha=archive(tmp_path)
    for bad in ('0'*64,'bad',None):
        with pytest.raises(ValueError): load(p,bad)


@pytest.mark.parametrize('kw',[{'missing':True},{'mask':np.full((3,4),2,dtype=np.uint8)}, {'mask':np.zeros((2,4),dtype=np.uint8)}, {'image':np.zeros((3,4),dtype=np.uint8)}, {'extra':(('../escape',b'x'),)}])
def test_invalid_source_pairs_fail_closed(tmp_path,kw):
    p,sha=archive(tmp_path,**kw)
    with pytest.raises(ValueError): load(p,sha)


def test_rgba_original_retains_alpha_as_explicit_channel(tmp_path):
    pixels=np.arange(48,dtype=np.uint8).reshape(3,4,4)
    p,sha=archive(tmp_path,image=pixels)
    x,=load(p,sha)
    np.testing.assert_array_equal(x.image_array(),pixels)
    assert x.image.channels==('red','green','blue','alpha')
    with pytest.raises(ValueError,match='channel'): x.native_input()
