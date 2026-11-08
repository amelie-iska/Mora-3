"""Shared native objectives must update shared weights and only selected heads."""
import copy
from dataclasses import replace
import pytest
import torch
from test_morphology import task as image_task
from test_sciplex_shared import task as response_task
from mora3.model.cooperative_training import CooperativeModel, objective_schedule
from mora3.model.morphology import foreground_objective
from mora3.model.instance_affinity import affinity_objective
from mora3.model.morphology_training import targets

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_selected_native_equations_and_gradient_ownership(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(67)
    m=CooperativeModel(('A','B'),width=8).to(device).double()
    assert m.image.backbone is m.response.backbone
    assert len(list(m.parameters()))==len({id(p) for p in m.parameters()})
    for kind,t in [('image',image_task()),('response',response_task()),('intervention',response_task('intervention'))]:
        m.zero_grad(set_to_none=True)
        actual=m.objective(kind,[t])
        if kind=='image':
            z=m.image(t,for_training=True); labels=targets(t,device)
            expected=foreground_objective(z[0],labels)+affinity_objective(z[1:],labels)
        else: expected=m.response.loss([t])
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        actual.backward()
        assert m.response.backbone.feature.weight.grad.abs().sum()>0
        assert (m.image.head.weight.grad is not None)==(kind=='image')
        assert (m.response.response_head.weight.grad is not None)==(kind=='response')
        assert (m.response.intervention_head.weight.grad is not None)==(kind=='intervention')
        assert (m.image.pixels.enc0[0].weight.grad is not None)==(kind=='image')


def test_schedule_skips_absent_preserves_each_source_and_is_reproducible():
    pools={'image':tuple(range(3)),'response':tuple(range(5)),'intervention':()}
    a=objective_schedule(pools,step=0,batch_size=2)
    assert a==('image',(0,1))
    assert objective_schedule(pools,step=1,batch_size=2)==('response',(0,1))
    assert objective_schedule(pools,step=2,batch_size=2)==('image',(2,0))
    assert objective_schedule(pools,step=3,batch_size=2)==('response',(2,3))
    assert objective_schedule({},step=0,batch_size=2) is None
    for bad in [-1,1.5,True]:
        with pytest.raises(ValueError):objective_schedule(pools,step=bad,batch_size=2)
    with pytest.raises(ValueError):objective_schedule({'invented':(1,)},step=0,batch_size=1)


def test_task_mask_admission_and_hidden_target_boundary():
    m=CooperativeModel(('A','B'),width=8)
    a=image_task(); b=replace(a,example=replace(a.example,polygons=()))
    assert torch.equal(m.image(a),m.image(b))
    for kind,ts in [('response',[response_task('intervention')]),('image',[image_task('test')]),('image',[]),('absent',[a])]:
        with pytest.raises(ValueError):m.objective(kind,ts)
    assert torch.equal(m.response([response_task()]),m.response([response_task(counts=(99,88))]))


def test_image_batch_mean_and_permutation():
    m=CooperativeModel(('A','B'),width=8).double()
    a=image_task();b=replace(a,example=replace(a.example,polygons=()))
    torch.testing.assert_close(m.objective('image',[a,b]),(m.objective('image',[a])+m.objective('image',[b]))/2)
    torch.testing.assert_close(m.objective('image',[a,b]),m.objective('image',[b,a]))


def test_joint_checkpoint_exact_continuation(tmp_path):
    from mora3.model.training import save_checkpoint,load_checkpoint
    torch.manual_seed(67);m=CooperativeModel(('A','B'),width=8)
    opt=torch.optim.Adam(m.parameters(),lr=.001)
    pools={'image':(image_task(),),'response':(response_task(),),'intervention':(response_task('intervention'),)}
    binding=m.binding(pools,batch_size=1)
    def step(i):
        kind,ts=objective_schedule(pools,step=i,batch_size=1)
        opt.zero_grad(set_to_none=True); loss=m.objective(kind,ts);loss.backward();opt.step()
        return loss.item()
    step(0);path=tmp_path/'state.pt';save_checkpoint(path,m,opt,step=1,binding=binding)
    expected=step(1);state=copy.deepcopy(m.state_dict());optimizer=copy.deepcopy(opt.state_dict())
    assert load_checkpoint(path,m,opt,binding=binding)==1
    assert step(1)==expected
    for k,v in state.items():
        if isinstance(v,torch.Tensor):assert torch.equal(m.state_dict()[k],v)
        else:assert m.state_dict()[k]==v
    for k,v in optimizer['state'].items():
        for name,value in v.items():assert torch.equal(opt.state_dict()['state'][k][name],value)
    with pytest.raises(ValueError):load_checkpoint(path,m,opt,binding=m.binding(pools,batch_size=2))


def test_cpu_cuda_loss_and_gradient_parity():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(67);a=CooperativeModel(('A','B'),width=8).double();b=copy.deepcopy(a).cuda()
    for kind,t in [('image',image_task()),('response',response_task()),('intervention',response_task('intervention'))]:
        a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
        x=a.objective(kind,[t]);y=b.objective(kind,[t]);x.backward();y.backward()
        torch.testing.assert_close(x,y.cpu(),atol=1e-10,rtol=1e-10)
        for p,q in zip(a.parameters(),b.parameters()):
            assert (p.grad is None)==(q.grad is None)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad.cpu(),atol=1e-9,rtol=1e-8)


def test_joint_image_overlap_plane_order_and_exact_padded_identifiers():
    from mora3.model.instance_affinity import affinity_targets
    a=image_task();first=a.example.polygons[0]
    second=((2.,0.,5.,0.,5.,3.,2.,3.),)
    a=replace(a,example=replace(a.example,polygons=(first,second)))
    b=replace(a,example=replace(a.example,polygons=(second,first)))
    m=CooperativeModel(('A','B'),width=8).double()
    labels=targets(a,'cpu');assert (labels.sum(0)>1).any()
    assert torch.equal(m.image(a),m.image(b))
    torch.testing.assert_close(m.objective('image',[a]),m.objective('image',[b]),atol=0,rtol=0)
    _,valid=affinity_targets(labels)
    ambiguous=labels.sum(0)>1
    assert not valid[0,:,:-1][ambiguous[:,:-1]|ambiguous[:,1:]].any()
    assert not valid[1,:-1,:][ambiguous[:-1,:]|ambiguous[1:,:]].any()
    assert m.image.backbone.incidence.in_features==4
    # Capturing real backbone input verifies rank is padded, not approximated.
    graphs=[]
    handle=m.image.backbone.register_forward_pre_hook(lambda module,args:graphs.extend(args[0]))
    m.image(a);handle.remove()
    assert torch.equal(graphs[0]['identifiers'],torch.tensor([[1.,0.]],dtype=torch.float64))
