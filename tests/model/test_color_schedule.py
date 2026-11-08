import copy
import pytest
import torch
from test_lincs_cooperative import build, cases
from test_color_supervised_task import source
from mora3.model.color_supervised_task import ColorSemanticTask
from mora3.model.color_cooperative import ColorCooperativeModel


def make(tmp_path,cases,device='cpu',**kwargs):
    from mora3.model.color_schedule import ColorSemanticSchedule
    model=ColorCooperativeModel(build(tmp_path,cases,device).model,color_id_width=2)
    s=source();task=ColorSemanticTask(s,s.binding_sha256)
    return ColorSemanticSchedule(model,(task,),weight=2.5,batch_size=1,**kwargs)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_weighted_reference_and_all_active_gradients(tmp_path,cases,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    schedule=make(tmp_path,cases,device)
    loss,n=schedule.evaluate(3)
    oracle=schedule.model.color_objective(schedule.tasks[0])['loss']*2.5
    assert n==1
    torch.testing.assert_close(loss,oracle)
    params=tuple(schedule.model.color.parameters())
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
    from mora3.model.color_schedule import ColorSemanticSchedule
    s=make(tmp_path,cases)
    if change in ('weight','size','duplicate','empty','context'):
        kw=dict(weight=1.,batch_size=1)
        tasks=s.tasks
        if change=='weight':kw['weight']=float('nan')
        if change=='size':kw['batch_size']=True
        if change=='duplicate':tasks=tasks*2
        if change=='empty':tasks=()
        if change=='context':kw['context_fields']=('mask',)
        with pytest.raises(ValueError):ColorSemanticSchedule(s.model,tasks,**kw)
    elif change=='step':
        with pytest.raises(ValueError):s.evaluate(True)
    else:
        s.model.configuration['version']=99
        with pytest.raises(ValueError,match='binding'):s.evaluate(0)


def test_cyclic_batches_preserve_order_and_source_bound_context(tmp_path,cases):
    from dataclasses import replace
    from mora3.model.color_schedule import ColorSemanticSchedule
    s=make(tmp_path,cases)
    records=[]
    for i in range(3):
        original=source()
        record=replace(original,image=replace(original.image,artifact_id='image-'+str(i)))
        records.append(ColorSemanticTask(record,record.binding_sha256))
    schedule=ColorSemanticSchedule(s.model,records,weight=1.,batch_size=2,context_fields=('tissue',))
    assert schedule.select(0)==tuple(records[:2])
    assert schedule.select(1)==(records[2],records[0])
    assert schedule.select(2)==(records[1],records[2])
    loss,n=schedule.evaluate(1)
    from mora3.model.color_semantic_inference import image_artifact_digest
    ref=torch.stack([s.model.color_objective(t,context_fields=('tissue',),
        expected_image_sha256=image_artifact_digest(t.source.image))['loss'] for t in (records[2],records[0])]).mean()
    torch.testing.assert_close(loss,ref)
    assert n==2
