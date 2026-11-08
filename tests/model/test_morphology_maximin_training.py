import copy
import pytest
import torch
from test_morphology import task
from mora3.model.morphology_training import MorphologyTrainer, targets
from mora3.model.morphology import foreground_objective
from mora3.model.maximin import maximin_objective

@pytest.fixture(autouse=True)
def deterministic():
    previous=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_maximin_trainer_equation_and_exact_resume(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(67)
    trainer=MorphologyTrainer([task()],width=8,device=device,affinity_mode='maximin-square')
    z=trainer.model(trainer.tasks[0],for_training=True)
    m=targets(trainer.tasks[0],device); _,h,w=m.shape
    labels=torch.full((h,w),-1,device=device,dtype=torch.long)
    for i,plane in enumerate(m): labels[plane]=i
    labels[m.sum(0)!=1]=-1
    ids=torch.arange(h*w,device=device).reshape(h,w)
    edges=torch.cat([torch.stack([ids[:,:-1].flatten(),ids[:,1:].flatten()],1),torch.stack([ids[:-1,:].flatten(),ids[1:,:].flatten()],1)])
    logits=torch.cat([z[1,:,:-1].flatten(),z[2,:-1,:].flatten()])
    expected=foreground_objective(z[0],m)+maximin_objective(logits,edges,labels.flatten())
    assert trainer.step()==float(expected.detach())
    state=copy.deepcopy(trainer.snapshot()); loss=trainer.step()
    resumed=MorphologyTrainer([task()],width=8,device=device,affinity_mode='maximin-square')
    resumed.restore(state)
    assert resumed.step()==loss
    for k,v in trainer.model.state_dict().items():
        if isinstance(v,torch.Tensor): assert torch.equal(v,resumed.model.state_dict()[k])
    with pytest.raises(ValueError,match='config'):
        MorphologyTrainer([task()],width=8,device=device).restore(state)
    with pytest.raises(ValueError,match='configuration'):
        MorphologyTrainer([task()],variant='foreground',affinity_mode='maximin-square')


def test_maximin_variant_keeps_hidden_masks_out_of_conditioning():
    from dataclasses import replace
    from mora3.data.livecell_graph import LiveCellGraphTask
    a=task(); b=LiveCellGraphTask(replace(a.example,polygons=()),a.direction)
    trainer=MorphologyTrainer([a],width=8,affinity_mode='maximin-square')
    assert torch.equal(trainer.model(a),trainer.model(b))
    with pytest.raises(ValueError,match='split'):
        MorphologyTrainer([task('test')],width=8,affinity_mode='maximin-square')


def test_raster_maximin_overlap_background_and_border_semantics():
    from mora3.model.maximin import raster_maximin_objective
    z=torch.zeros((2,1,3),requires_grad=True)
    m=torch.tensor([[[True,True,False]],[[False,True,True]]])
    # Only endpoints 0 and 2 supervised, distinct objects; overlap is transit.
    torch.testing.assert_close(raster_maximin_objective(z,m),torch.tensor(.125))
    assert raster_maximin_objective(z,m[:0]).item()==0
    with pytest.raises(ValueError): raster_maximin_objective(z[:1],m)
