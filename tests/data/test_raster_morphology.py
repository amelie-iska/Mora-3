from dataclasses import replace
import numpy as np
import pytest
import torch
from test_deepbacs import archive
from mora3.data.deepbacs import read_full_images
from mora3.data.raster_morphology import RasterMorphologyTask
from mora3.model.morphology import MorphologyForeground, foreground_objective


def task(tmp_path, split='train', direction='image_to_mask'):
    p,sha=archive(tmp_path)
    row=next(r for r in read_full_images(p,expected_sha256=sha) if r.split==split)
    return RasterMorphologyTask.from_deepbacs(row,direction)


def test_graph_geometry_masks_and_target_exclusion(tmp_path):
    a=task(tmp_path)
    b=replace(a,labels_hex='0000'*12)
    assert a.input_graph().canonical_bytes()==b.input_graph().canonical_bytes()
    assert torch.equal(a.input_image(),b.input_image())
    assert len(a.input_graph().nodes)==1
    g=a.target_graph()
    assert len(g.nodes)==3 and len(g.edges)==2
    values={m.name:m.value for m in g.nodes[1].measurements}
    assert values['area']==2 and values['x_min']==1 and values['x_max']==2
    assert 'instance_id' not in values
    p=a.project()
    assert set(p['inputs'])=={'image'}
    assert p['targets']['instances'].shape==(2,3,4)
    assert p['targets']['instances'].any(0).sum()==4
    reverse=replace(a,direction='mask_to_image').project()
    assert set(reverse['inputs'])=={'instances'} and set(reverse['targets'])=={'image'}
    assert a.input_graph().provenance[0].license=='CC-BY-4.0'


def test_resume_split_and_snapshot_validation(tmp_path):
    a=task(tmp_path)
    restored=RasterMorphologyTask.from_bytes(a.canonical_bytes())
    a.assert_resume_matches(restored)
    with pytest.raises(ValueError,match='resume'): a.assert_resume_matches(replace(a,direction='mask_to_image'))
    with pytest.raises(ValueError,match='split'): task(tmp_path,'test').project(for_training=True)
    with pytest.raises(ValueError): replace(a,image_hex='00')
    with pytest.raises(ValueError): replace(a,direction='unknown')
    with pytest.raises(ValueError): RasterMorphologyTask.from_bytes(a.canonical_bytes().replace(b'"version":1',b'"version":2'))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_neural_connection_target_exclusion_and_gradients(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    torch.manual_seed(37)
    a=task(tmp_path); model=MorphologyForeground(width=8).to(device)
    z=model(a,for_training=True)
    assert z.shape==(3,4)
    assert torch.equal(z,model(replace(a,labels_hex='0000'*12)))
    foreground_objective(z,a.project(device=device)['targets']['instances']).backward()
    for module in (model.pixels,model.backbone,model.head):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert sum(p.grad.abs().sum() for p in module.parameters())>0
    with pytest.raises(ValueError,match='split'): model(task(tmp_path,'test'),for_training=True)


def test_snapshot_requires_explicit_version_and_direction(tmp_path):
    import json
    a=task(tmp_path)
    for field in ('version','direction'):
        d=json.loads(a.canonical_bytes()); del d[field]
        with pytest.raises(ValueError): RasterMorphologyTask.from_bytes(json.dumps(d))
