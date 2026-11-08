"""Cache removal would fail hit/eviction and exact learned-path reuse checks."""
import copy
import pytest
import torch
from test_sciplex_shared import task
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.sciplex_preparation import SciPlexPreparationCache


def test_bounded_content_precision_admission_and_mutation_isolation():
    c=SciPlexPreparationCache(max_bytes=8192)
    t=task()
    a=c.project(t,axis_ids=('A','B'),dtype=torch.float32)
    a['targets']['expression'].fill_(999)
    b=c.project(t,axis_ids=('A','B'),dtype=torch.float32)
    assert b['targets']['expression'].tolist()==[1,2]
    assert c.hits==1 and c.bytes_used<=8192
    c.project(task(counts=(3,4)),axis_ids=('A','B'),dtype=torch.float32)
    c.project(t,axis_ids=('A','B'),dtype=torch.float64)
    assert c.misses==3 and c.bytes_used<=8192
    with pytest.raises(ValueError):c.project(t,axis_ids=('B','A'))
    with pytest.raises(ValueError):c.project(t.task,axis_ids=('A','B'))
    tiny=SciPlexPreparationCache(max_bytes=1)
    tiny.project(t,axis_ids=('A','B'))
    assert tiny.bytes_used==0
    c.clear();assert c.bytes_used==0


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('direction',['response','intervention'])
def test_cached_logits_losses_gradients_and_updates_exact(device,direction):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(1601)
    a=SciPlexSharedModel(('A','B'),width=8).to(device)
    b=copy.deepcopy(a);b.preparation_cache=SciPlexPreparationCache(max_bytes=4096)
    ts=[task(direction)]
    oa=torch.optim.Adam(a.parameters());ob=torch.optim.Adam(b.parameters())
    for _ in range(3):
        oa.zero_grad();ob.zero_grad()
        assert torch.equal(a(ts),b(ts))
        la,lb=a.loss(ts),b.loss(ts);assert torch.equal(la,lb)
        la.backward();lb.backward()
        for p,q in zip(a.parameters(),b.parameters()):
            if p.grad is not None:assert torch.equal(p.grad,q.grad)
        oa.step();ob.step()
    assert b.preparation_cache.hits>0
    # Cold cache after loading the same weights must not change predictions.
    b.preparation_cache.clear();b.load_state_dict(a.state_dict())
    assert torch.equal(a(ts),b(ts))
    other=task(direction,counts=(99,88)) if direction=='response' else task(direction,identity='SAHA')
    for key,v in b.graph(ts[0]).items():assert torch.equal(v,b.graph(other)[key])


def test_exact_bytes_budget_eviction_and_receipt_invalidation():
    import json
    from mora3.data.sciplex_admission import ObservationalSciPlexTask
    t=task();c=SciPlexPreparationCache(max_bytes=100000)
    c.project(t,axis_ids=('A','B'));size=c.bytes_used
    c.max_bytes=size
    c.project(task(counts=(3,4)),axis_ids=('A','B'))
    c.project(t,axis_ids=('A','B'))
    assert c.misses==3 and c.hits==0 and c.bytes_used<=size
    r=json.loads(t.receipt_bytes);r['audit_sha256']='c'*64
    other=ObservationalSciPlexTask.from_receipt(t.task,r)
    c.project(other,axis_ids=('A','B'))
    assert c.misses==4


def test_cold_cache_full_optimizer_resume_exact(tmp_path):
    from mora3.model.training import save_checkpoint,load_checkpoint
    torch.manual_seed(1602)
    m=SciPlexSharedModel(('A','B'),width=8)
    m.preparation_cache=SciPlexPreparationCache(max_bytes=100000)
    opt=torch.optim.Adam(m.parameters(),lr=.001);ts=[task('intervention')]
    def step():
        opt.zero_grad();v=m.loss(ts);v.backward();opt.step();return v.detach()
    step();p=tmp_path/'state.pt'
    save_checkpoint(p,m,opt,step=1,binding=m.binding(ts))
    expected_loss=step();expected=copy.deepcopy(m.state_dict())
    m.preparation_cache.clear()
    assert load_checkpoint(p,m,opt,binding=m.binding(ts))==1
    assert torch.equal(step(),expected_loss)
    for k,v in expected.items():
        actual=m.state_dict()[k]
        assert torch.equal(v,actual) if isinstance(v,torch.Tensor) else v==actual
