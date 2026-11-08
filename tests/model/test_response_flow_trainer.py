import copy
import pytest
import torch
from test_sciplex_shared import task
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.response_flow import ResponseFlowHead
from mora3.model.flow_matching import gaussian_path, velocity_loss
from mora3.model.response_flow_trainer import ResponseFlowBatch, ResponseFlowTrainer


def make(device='cpu', **kwargs):
    model=ResponseFlowHead(SciPlexSharedModel(('A','B'),width=8)).to(device).double()
    return ResponseFlowTrainer(model,ResponseFlowBatch((task(),task(counts=(3,7),cell='b'))),batch_size=1,**kwargs)


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_stochastic_step_matches_reference_and_full_state_resume(device,tmp_path):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    torch.manual_seed(5501);tr=make(device);m=copy.deepcopy(tr.model)
    rng=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all() if device=='cuda' else None
    y=torch.log1p(task().project(device=device,dtype=torch.float64)['targets']['expression'])[None]
    noise=torch.randn_like(y);time=torch.rand(1,device=device,dtype=y.dtype)
    x,u=gaussian_path(noise,y,time,sigma_min=.01)
    loss=velocity_loss(m([task()],x,time),u,torch.ones_like(y,dtype=torch.bool))
    opt=torch.optim.Adam(m.parameters(),lr=.001);loss.backward();opt.step()
    torch.set_rng_state(rng)
    if gpu is not None:torch.cuda.set_rng_state_all(gpu)
    result=tr.step();assert result['loss']==float(loss.detach());assert result['indices']==(0,)
    equal(tr.model.state_dict(),m.state_dict());equal(tr.optimizer.state_dict(),opt.state_dict())
    assert tr.model.native.response_head.weight.grad is None
    assert tr.model.native.intervention_head.weight.grad is None
    assert tr.model.native.backbone.feature.weight.grad.abs().sum()>0
    path=tmp_path/'state.pt';torch.save(tr.state_dict(),path)
    result=tr.step();assert result['indices']==(1,)
    expected=tr.state_dict();fresh=make(device)
    fresh.load_state_dict(torch.load(path,weights_only=False));assert fresh.step()==result
    equal(fresh.state_dict(),expected)


def test_source_target_axis_schedule_and_path_binding_fail_before_mutation():
    tr=make();tr.step();state=tr.state_dict()
    for other in [make(sigma_min=.1),make(lr=.002),
        ResponseFlowTrainer(copy.deepcopy(tr.model),ResponseFlowBatch((task(counts=(99,2)),)),batch_size=1),
        ResponseFlowTrainer(copy.deepcopy(tr.model),tr.batch,batch_size=2)]:
        before=copy.deepcopy(other.model.state_dict());rng=torch.get_rng_state()
        with pytest.raises(ValueError,match='binding'):other.load_state_dict(state)
        equal(other.model.state_dict(),before);assert torch.equal(rng,torch.get_rng_state())
    for tasks in [(),(task('intervention'),),(task().task,)]:
        with pytest.raises(ValueError):ResponseFlowBatch(tasks)
    with pytest.raises(ValueError):ResponseFlowTrainer(ResponseFlowHead(SciPlexSharedModel(('B','A'),width=8)),tr.batch,batch_size=1)
    for kwargs in [dict(sigma_min=float('nan')),dict(sigma_min=1),dict(batch_size=0),dict(lr=-1)]:
        options=dict(batch_size=1);options.update(kwargs)
        with pytest.raises(ValueError):ResponseFlowTrainer(copy.deepcopy(tr.model),tr.batch,**options)


def test_target_changes_binding_but_not_context_and_input_batch_is_immutable():
    values=[task()];batch=ResponseFlowBatch(values);values.clear();assert len(batch.tasks)==1
    changed=ResponseFlowBatch((task(counts=(90,80),cell='hidden'),))
    assert batch.binding()!=changed.binding()
    tr=ResponseFlowTrainer(ResponseFlowHead(SciPlexSharedModel(('A','B'),width=8)),batch,batch_size=1)
    x=torch.ones(1,2);t=torch.tensor([.3])
    assert torch.equal(tr.model(batch.tasks,x,t),tr.model(changed.tasks,x,t))
