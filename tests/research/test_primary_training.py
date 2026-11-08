"""Synthetic primitive updates; interrupted versus uninterrupted full recovery."""
import copy
import random

import numpy as np
import pytest
import torch
from test_mu_optimizer import target,optimizer


def engine():
    from mora3.research.primary_training import NativeMuUpdateEngine
    model,roles=target();opt=optimizer(model,roles)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=2,gamma=.8)
    return NativeMuUpdateEngine(model,opt,binding_sha256='a'*64,
        task_cycle=('one','two','one'),source_cursors={'one':b'0','two':b'0'},scheduler=scheduler)


def seed():
    random.seed(20261106);np.random.seed(20261106);torch.manual_seed(20261106)


def advance(e):
    # Distinct stochastic sources are all necessary to reproduce the update.
    scale=random.random()+float(np.random.random())+torch.rand((),dtype=torch.float64)
    loss=sum((p*scale).square().sum() for p in e.model.parameters() if p.requires_grad)
    name=e.next_task;before=e.source_cursors[name];after=str(int(before)+1).encode()
    return e.update(name,loss,cursor_before=before,cursor_after=after)


def equal(a,b):
    if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and torch.equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert type(a)is type(b) and len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert type(a)is type(b) and a==b


def test_safe_checkpoint_full_rng_optimizer_scheduler_cursor_continuation(tmp_path):
    seed();reference=engine();reference_events=[advance(reference) for _ in range(7)]
    reference_state=reference.checkpoint_state()
    seed();first=engine();events=[advance(first) for _ in range(3)]
    path=tmp_path/'checkpoint.pt';first.save_checkpoint(path)
    before=path.read_bytes()
    with pytest.raises(FileExistsError):first.save_checkpoint(path)
    assert path.read_bytes()==before
    # Construction/noise before restore must not alter continuation.
    resumed=engine();random.random();np.random.random();torch.rand(7)
    resumed.load_checkpoint(path)
    events.extend(advance(resumed) for _ in range(4))
    assert events==reference_events
    equal(resumed.checkpoint_state(),reference_state)
    assert resumed.model.output is resumed.model.tied
    assert resumed.scheduler.optimizer is resumed.optimizer
    assert all(id(p) in {id(v) for v in resumed.model.parameters()}
        for group in resumed.optimizer.param_groups for p in group['params'])


@pytest.mark.parametrize('drift',['binding','rng','optimizer','scheduler','alias','nonfinite'])
def test_failed_restore_never_mutates_live_parameters_optimizer_rng_or_cursors(drift):
    seed();e=engine();advance(e);before=e.checkpoint_state();state=copy.deepcopy(before)
    if drift=='binding':state['binding_sha256']='b'*64
    elif drift=='rng':state['rng']['numpy']=('MT19937',[0],0,0,0.)
    elif drift=='optimizer':state['optimizer']['param_groups'][-1]['lr']=float('nan')
    elif drift=='scheduler':state['scheduler']['gamma']=.1
    elif drift=='alias':state['model']['tied']=state['model']['tied']+1
    else:state['model']['input'][0,0]=float('nan')
    with pytest.raises((ValueError,RuntimeError)):e.restore_state(state)
    equal(e.checkpoint_state(),before)


@pytest.mark.parametrize('failure',['task','cursor','nonfinite-loss','nonfinite-gradient'])
def test_invalid_update_does_not_advance_optimizer_parameters_or_source_cursor(failure):
    seed();e=engine();before=e.checkpoint_state();name=e.next_task
    loss=e.model.input.sum()
    cursor=e.source_cursors[name]
    if failure=='task':name='two'
    elif failure=='cursor':cursor=b'wrong'
    elif failure=='nonfinite-loss':loss=loss*float('nan')
    else:
        # Finite forward, infinite derivative at exactly zero.
        loss=(e.model.input-e.model.input.detach()).sqrt().sum()
    with pytest.raises(ValueError):e.update(name,loss,cursor_before=cursor,cursor_after=b'1')
    assert e.steps==0 and e.source_cursors==before['source_cursors']
    equal(e.model.state_dict(),before['model']);equal(e.optimizer.state_dict(),before['optimizer'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='existing CUDA device unavailable')
def test_initialized_cuda_generator_state_is_staged_and_restored_without_updates():
    seed();torch.cuda.init();e=engine();state=e.checkpoint_state()
    generator=torch.Generator(device='cuda:0').set_state(state['rng']['torch_cuda'][0])
    expected=torch.rand(4,device='cuda:0',generator=generator)
    torch.rand(7,device='cuda:0')
    e.restore_state(state)
    assert torch.equal(torch.rand(4,device='cuda:0'),expected)
    assert e.steps==0 and not e.optimizer.state
