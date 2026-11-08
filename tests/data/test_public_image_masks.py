import hashlib
import io
from dataclasses import replace
import numpy as np
import pytest
from PIL import Image
from mora3.data import public_image_masks as public
from mora3.data.cooperative_segmentation import SegmentationArtifact


def png(a):
    b=io.BytesIO(); Image.fromarray(a).save(b,format='PNG'); return b.getvalue()


def fixture(kind='ordered_binary_instances'):
    image=png(np.arange(12,dtype=np.uint8).reshape(3,4))
    masks=(png(np.array([[0,255,255,0],[0,0,0,0],[0,0,0,0]],dtype=np.uint8)),
           png(np.array([[0,0,255,0],[0,0,255,0],[0,0,0,0]],dtype=np.uint8)))
    if kind!='ordered_binary_instances': masks=masks[:1]
    a=SegmentationArtifact('image','image','source_observed','condition','fixture:v1','parent',
        (('perturbation',None),('expression',None)),'source#image','source-yx-no-transform-v1',
        channels=('brightfield',),spatial_shape=(3,4))
    return a,image,masks,kind


def build(parts):
    a,image,masks,kind=parts
    return public.PublicImageMasks.from_encoded(a,image,masks,encoding=kind,split='test',
        image_sha256=hashlib.sha256(image).hexdigest(),
        mask_sha256=tuple(hashlib.sha256(m).hexdigest() for m in masks))


def test_ordered_overlap_and_target_exclusion():
    parts=fixture(); x=build(parts)
    targets=x.target_arrays()
    assert len(targets)==2 and targets[0][0,2] and targets[1][0,2]
    assert targets[0].dtype==np.bool_
    changed=list(parts); changed[2]=tuple(reversed(parts[2])); y=build(changed)
    assert x.native_input()==y.native_input()
    assert x.binding_sha256!=y.binding_sha256
    targets[0][:]=False
    assert x.target_arrays()[0].sum()==2
    assert x.native_input().artifact.context==parts[0].context
    with pytest.raises(ValueError,match='admission'): x.require_training()


@pytest.mark.parametrize('kind',['semantic_binary','instance_id_map'])
def test_semantic_is_not_instance_labels(kind):
    x=build(fixture(kind))
    a=x.target_arrays()[0]
    assert (a.dtype==np.bool_)==(kind=='semantic_binary')
    if kind=='instance_id_map': assert a[0,1]==255
    assert x.encoding==kind


@pytest.mark.parametrize('change',['image_hash','mask_hash','alignment','binary','count','split','origin','channels'])
def test_invalid_source_contract_fails_closed(change):
    a,image,masks,kind=fixture()
    kw=dict(encoding=kind,split='test',image_sha256=hashlib.sha256(image).hexdigest(),
        mask_sha256=tuple(hashlib.sha256(m).hexdigest() for m in masks))
    if change=='image_hash': kw['image_sha256']='0'*64
    if change=='mask_hash': kw['mask_sha256']=('0'*64,)*2
    if change=='alignment': masks=(png(np.zeros((2,4),dtype=np.uint8)),); kw['mask_sha256']=(hashlib.sha256(masks[0]).hexdigest(),)
    if change=='binary': masks=(png(np.full((3,4),2,dtype=np.uint8)),); kw['mask_sha256']=(hashlib.sha256(masks[0]).hexdigest(),)
    if change=='count': kw['mask_sha256']=()
    if change=='split': kw['split']='unknown'
    if change=='origin': a=replace(a,evidence='model_generated',checkpoint='g')
    if change=='channels': a=replace(a,channels=('red','green','blue'))
    with pytest.raises(ValueError): public.PublicImageMasks.from_encoded(a,image,masks,**kw)


def test_rgb_preserved_but_native_channel_conversion_refused():
    a,image,masks,kind=fixture()
    a=replace(a,channels=('red','green','blue'))
    pixels=np.arange(36,dtype=np.uint8).reshape(3,4,3)
    image=png(pixels)
    x=build((a,image,masks,kind))
    assert np.array_equal(x.image_array(),pixels)
    with pytest.raises(ValueError,match='channel'): x.native_input()


def test_streamed_targets_preserve_order_without_retaining_full_stack():
    x=build(fixture())
    iterator=x.iter_target_arrays()
    assert iter(iterator) is iterator
    assert next(iterator)[0,1]
    assert not next(iterator)[0,1]
    with pytest.raises(StopIteration): next(iterator)


def test_pillow_mode1_source_binary_masks_are_preserved():
    a,image,masks,kind=fixture()
    binary=np.zeros((3,4),dtype=bool); binary[1,2]=True
    x=build((a,image,(png(binary),),kind))
    assert np.array_equal(next(x.iter_target_arrays()),binary)
