"""Bound continuation must preserve Adam/RNG state and reject changed requests."""
import copy
from dataclasses import replace
import hashlib
import pytest
import torch
from test_segmentation_inference import setup
from mora3.model.cooperative_training import CooperativeModel
from mora3.model.geometry_pixel_fusion import GeometryPixelFusion
from mora3.model import geometry_pixel_fusion as module


def equal(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

@pytest.fixture(autouse=True)
def deterministic():
    previous=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_exact_snapshot_continuation_and_no_snapshot_alias(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    t,n,r=setup(tmp_path,device)
    f=GeometryPixelFusion(CooperativeModel(('A','B'),width=8).image,id_width=16).to(device)
    trainer=module.GeometryPixelTrainer(f,t,r,n,admitted_task_sha256=hashlib.sha256(t.canonical_bytes()).hexdigest(),prediction_asset='prediction.pt')
    trainer.step();state=trainer.snapshot();untouched=copy.deepcopy(state)
    loss=trainer.step();expected=trainer.snapshot()
    trainer.restore(state);assert trainer.step()==loss
    equal(expected,trainer.snapshot());equal(state,untouched)
    assert trainer.steps==2
    for changed in [dict(binding={}),dict(steps=-1),dict(steps=True)]:
        bad=copy.deepcopy(state);bad.update(changed)
        before=trainer.snapshot()
        with pytest.raises(ValueError):trainer.restore(bad)
        equal(before,trainer.snapshot())
    other=module.GeometryPixelTrainer(copy.deepcopy(f),t,replace(r,supported_objects=('whole_cell','nucleus')),n,admitted_task_sha256=hashlib.sha256(t.canonical_bytes()).hexdigest(),prediction_asset='prediction.pt')
    with pytest.raises(ValueError):other.restore(state)
