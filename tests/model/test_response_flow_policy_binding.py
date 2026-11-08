"""Flow policy drift refuses before stochastic or live-state activity."""
import copy
import pytest
import torch
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.response_flow import ResponseFlowHead, ResponseFlowDenoiserV2
from mora3.model.response_flow_trainer import ResponseFlowTrainer, ResponseFlowBatch
from mora3.model.response_sampling import sample_response, ResponseContext
from mora3.model.tokengt import TokenGT
from test_sciplex_shared import task
from test_response_flow_trainer import equal


def head(version):
    native = SciPlexSharedModel(('A','B'), width=8)
    return ResponseFlowHead(native) if version == 1 else ResponseFlowDenoiserV2(native)


def drift(m, kind):
    if kind == 'wrapper': m.config['native']['width'] = 16
    elif kind == 'native': m.native.config['attention'] = 'tropical'
    else: m.native.backbone = TokenGT(8, 2, width=8, heads=2, layers=2,
                                     scaling_policy='mup-v1')


@pytest.mark.parametrize('version', [1,2])
@pytest.mark.parametrize('kind', ['wrapper','native','policy'])
@pytest.mark.parametrize('entry', ['extra','restore-extra','forward','sample','sample-reuse','construct','step','save','load'])
def test_drift_refuses_before_rng_parameter_optimizer_or_mode_mutation(version, kind, entry):
    m=head(version); batch=ResponseFlowBatch((task(),))
    tr=ResponseFlowTrainer(m,batch,batch_size=1)
    saved=tr.state_dict()
    drift(m,kind)
    params={k:v.clone() for k,v in m.named_parameters()}
    opt=copy.deepcopy(tr.optimizer.state_dict());rng=torch.get_rng_state().clone()
    modes=[v.training for v in m.modules()]
    with pytest.raises(ValueError, match='configuration'):
        if entry=='extra':m.get_extra_state()
        elif entry=='restore-extra':m.set_extra_state(saved['model']['_extra_state'])
        elif entry=='forward':m([task()],torch.ones(1,2),torch.tensor([.3]))
        elif entry.startswith('sample'):
            sample_response(m,[ResponseContext.from_task(task())],steps=2,seed=1,
                sigma_min=.01,negative_policy='preserve',reuse_context=entry=='sample-reuse')
        elif entry=='construct':ResponseFlowTrainer(m,batch,batch_size=1)
        elif entry=='step':tr.step()
        elif entry=='save':tr.state_dict()
        else:tr.load_state_dict(saved)
    assert torch.equal(rng,torch.get_rng_state())
    assert modes==[v.training for v in m.modules()]
    equal(opt,tr.optimizer.state_dict())
    for k,v in m.named_parameters():assert torch.equal(params[k],v)
    assert tr.steps==0


@pytest.mark.parametrize('version',[1,2])
def test_staged_bad_checkpoint_metadata_refuses_without_live_mutation(version):
    tr=ResponseFlowTrainer(head(version),ResponseFlowBatch((task(),)),batch_size=1)
    before=tr.state_dict();bad=copy.deepcopy(before)
    bad['model']['_extra_state']['native']['width']=16
    # A tensor that would visibly corrupt a non-staged restore.
    bad['model']['state.weight'].fill_(123)
    with pytest.raises(ValueError,match='configuration'):tr.load_state_dict(bad)
    equal(before,tr.state_dict())


@pytest.mark.parametrize('version',[1,2])
@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_validation_preserves_legacy_equations_gradients_state_and_rng(version,device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(37101)
    m=head(version).to(device=device,dtype=dtype);ref=copy.deepcopy(m)
    x=torch.ones(1,2,device=device,dtype=dtype,requires_grad=True);y=x.detach().clone().requires_grad_()
    t=x.new_tensor([.3]);rng=torch.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    config=m.get_extra_state();m.set_extra_state(config)
    out=m([task()],x,t)
    context=ref.native.backbone([ref.native.graph(task())])[0][:,0]
    den=ref.output(torch.nn.functional.silu(context+ref.state(y)+ref.time(t[:,None])))
    expected=den if version==1 else (den-.99*y)/(1-.99*t[:,None])
    assert torch.equal(out,expected)
    out.sum().backward();expected.sum().backward()
    assert torch.equal(x.grad,y.grad)
    for p,q in zip(m.parameters(),ref.parameters()):
        assert (p.grad is None and q.grad is None) or torch.equal(p.grad,q.grad)
    equal(m.state_dict(),ref.state_dict())
    assert torch.equal(rng,torch.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
