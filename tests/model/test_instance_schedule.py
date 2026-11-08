import copy
import pytest
import torch
from test_lincs_cooperative import build, cases
from test_color_supervised_task import source
from mora3.model.color_instance_task import ColorInstanceTask
from mora3.model.color_cooperative import ColorCooperativeModel


def make(tmp_path,cases,device='cpu',**kwargs):
    from mora3.model.instance_schedule import InstanceSchedule
    from mora3.model.instance_cooperative import InstanceCooperativeModel
    model=InstanceCooperativeModel(ColorCooperativeModel(build(tmp_path,cases,device).model,color_id_width=2), slots=3,max_output_bytes=1000000)
    s=source('ordered_binary_instances',split='train');task=ColorInstanceTask(s,s.binding_sha256)
    return InstanceSchedule(model,(task,),weight=2.5,batch_size=1,**kwargs)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_weighted_reference_and_all_active_gradients(tmp_path,cases,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    schedule=make(tmp_path,cases,device)
    loss,n=schedule.evaluate(3)
    oracle=schedule.model.instance_objective(schedule.tasks[0])['loss']*2.5
    assert n==1
    torch.testing.assert_close(loss,oracle)
    params=tuple(schedule.model.instance.parameters())
    a=torch.autograd.grad(loss,params);b=torch.autograd.grad(oracle,params)
    for x,y in zip(a,b):torch.testing.assert_close(x,y)
    assert all(torch.isfinite(x).all() for x in a)
    before=copy.deepcopy(schedule.model.state_dict())
    with pytest.raises(ValueError,match='admission|held-out'):schedule.objective(0)
    for key,value in before.items():
        if isinstance(value,torch.Tensor):assert torch.equal(value,schedule.model.state_dict()[key])
    state=schedule.state_dict();schedule.validate_state(state)
    state['weight']=True
    with pytest.raises(ValueError):schedule.validate_state(state)


@pytest.mark.parametrize('change',['weight','size','duplicate','empty','step','policy','context'])
def test_reject_invalid_policy_and_drift(tmp_path,cases,change):
    from mora3.model.instance_schedule import InstanceSchedule
    s=make(tmp_path,cases)
    if change in ('weight','size','duplicate','empty','context'):
        kw=dict(weight=1.,batch_size=1)
        tasks=s.tasks
        if change=='weight':kw['weight']=float('nan')
        if change=='size':kw['batch_size']=True
        if change=='duplicate':tasks=tasks*2
        if change=='empty':tasks=()
        if change=='context':kw['context_fields']=('mask',)
        with pytest.raises(ValueError):InstanceSchedule(s.model,tasks,**kw)
    elif change=='step':
        with pytest.raises(ValueError):s.evaluate(True)
    else:
        s.model.configuration['version']=99
        with pytest.raises(ValueError,match='binding'):s.evaluate(0)


def test_cyclic_batches_preserve_order_and_source_bound_context(tmp_path,cases):
    from dataclasses import replace
    from mora3.model.instance_schedule import InstanceSchedule
    s=make(tmp_path,cases)
    records=[]
    for i in range(3):
        original=source('ordered_binary_instances')
        record=replace(original,image=replace(original.image,artifact_id='image-'+str(i)))
        records.append(ColorInstanceTask(record,record.binding_sha256))
    schedule=InstanceSchedule(s.model,records,weight=1.,batch_size=2,context_fields=('tissue',))
    assert schedule.select(0)==tuple(records[:2])
    assert schedule.select(1)==(records[2],records[0])
    assert schedule.select(2)==(records[1],records[2])
    loss,n=schedule.evaluate(1)
    from mora3.model.color_semantic_inference import image_artifact_digest
    ref=torch.stack([s.model.instance_objective(t,context_fields=('tissue',),
        expected_image_sha256=image_artifact_digest(t.source.image))['loss'] for t in (records[2],records[0])]).mean()
    torch.testing.assert_close(loss,ref)
    assert n==2


@pytest.mark.parametrize('block',[None,3])
def test_block_binding_and_pre_neural_admission(tmp_path,cases,monkeypatch,block):
    s=make(tmp_path,cases,pixel_block_size=block)
    loss,n=s.evaluate(0)
    expected=s.model.instance_objective(s.tasks[0],pixel_block_size=block)['loss']*2.5
    torch.testing.assert_close(loss,expected)
    assert n==1 and s.state_dict()['pixel_block_size']==block
    monkeypatch.setattr(s.model,'instance_objective',lambda *a,**k:pytest.fail('neural call before admission'))
    with pytest.raises(ValueError,match='admission'):s.objective(0)
    s.pixel_block_size=5
    with pytest.raises(ValueError,match='binding'):s.evaluate(0)


@pytest.mark.parametrize('block',[True,0,-1,1.5])
def test_invalid_block(tmp_path,cases,block):
    with pytest.raises(ValueError,match='block'):make(tmp_path,cases,pixel_block_size=block)
