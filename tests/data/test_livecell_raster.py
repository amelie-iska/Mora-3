from dataclasses import replace
import numpy as np
import pytest
import torch
from mora3.data.livecell import LiveCellExample
from mora3.data.livecell_graph import LiveCellGraphTask
from mora3.data.livecell_raster import rasterize_instances, raster_task


def example(polygons=(((0,0,2,0,2,2,0,2),),)):
    return LiveCellExample.from_coco(image=np.arange(20,dtype=np.uint8).reshape(4,5),
        image_info=dict(id=1,file_name='A172_Phase_C7_1_00d00h00m_1.tif',height=4,width=5),
        annotations=[dict(id=i,image_id=1,segmentation=list(p)) for i,p in enumerate(polygons)],
        split='train',source_sha256='a'*64)


def test_pixel_center_rectangle_and_components_preserve_instance_overlap():
    x=example((((0,0,2,0,2,2,0,2),(3,2,5,2,5,4,3,4)),((1,1,3,1,3,3,1,3),)))
    masks=rasterize_instances(x)
    expected=torch.zeros(2,4,5,dtype=torch.bool)
    expected[0,:2,:2]=True; expected[0,2:,3:]=True; expected[1,1:3,1:3]=True
    assert torch.equal(masks,expected)
    assert masks[:,1,1].sum()==2


def test_triangle_has_independent_analytic_center_oracle_and_winding_invariance():
    p=(0,0,4,0,0,4)
    a=rasterize_instances(example(((p,),)))[0]
    # Boundary centers on x+y=4 are included explicitly.
    y,x=torch.meshgrid(torch.arange(4)+.5,torch.arange(5)+.5,indexing='ij')
    assert torch.equal(a,x+y<=4)
    rev=tuple(v for pt in reversed(list(zip(p[::2],p[1::2]))) for v in pt)
    assert torch.equal(a,rasterize_instances(example(((rev,),)))[0])


def test_empty_degenerate_and_no_resize():
    assert rasterize_instances(example(())).shape==(0,4,5)
    assert not rasterize_instances(example((((0,0,1,1,2,2),),))).any()


def test_forward_target_mutation_never_changes_input_and_inverse_hides_pixels():
    x=example(); y=replace(x,polygons=())
    a=raster_task(LiveCellGraphTask(x,'image_to_mask'),for_training=True)
    b=raster_task(LiveCellGraphTask(y,'image_to_mask'),for_training=True)
    assert torch.equal(a['inputs']['image'],b['inputs']['image'])
    assert a['input_graph'].nodes==b['input_graph'].nodes
    assert set(a['inputs'])=={'image'} and a['targets']['instances'].dtype==torch.bool
    c=raster_task(LiveCellGraphTask(x,'mask_to_image'))
    d=raster_task(LiveCellGraphTask(replace(x,pixels_hex='ff'*20),'mask_to_image'))
    assert torch.equal(c['inputs']['instances'],d['inputs']['instances'])
    assert set(c['inputs'])=={'instances'}
    with pytest.raises(ValueError,match='split'):
        raster_task(LiveCellGraphTask(replace(x,split='test'),'image_to_mask'),for_training=True)


def test_cuda_matches_cpu():
    if not torch.cuda.is_available(): pytest.skip('no CUDA')
    x=example((((.5,.5,4,.5,.5,3.5),),))
    assert torch.equal(rasterize_instances(x),rasterize_instances(x,device='cuda').cpu())
