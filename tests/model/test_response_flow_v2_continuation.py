import copy
import pytest
import torch
from test_sciplex_shared import task
from test_response_flow_trainer import equal
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.response_flow import ResponseFlowHead, ResponseFlowDenoiserV2
from mora3.model.response_flow_trainer import ResponseFlowBatch, ResponseFlowTrainer
from mora3.model.response_sampling import ResponseContext, sample_response


def make(device='cpu', sigma=.01, version=2, counts=(3,7), batch_size=1):
    native=SciPlexSharedModel(('A','B'),width=8)
    h=ResponseFlowDenoiserV2(native,sigma_min=sigma) if version==2 else ResponseFlowHead(native)
    return ResponseFlowTrainer(h.to(device).double(),ResponseFlowBatch((task(),task(counts=counts,cell='b'))),batch_size=batch_size,sigma_min=sigma)


def test_sampler_rejects_v2_path_mismatch_before_rng_or_mode_change():
    h=make().model;c=ResponseContext.from_task(task())
    rng=torch.get_rng_state();before=copy.deepcopy(h.state_dict())
    with pytest.raises(ValueError,match='sigma'):
        sample_response(h,[c],steps=4,seed=62,sigma_min=.02,negative_policy='preserve')
    equal(h.state_dict(),before);assert h.training
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_v2_exact_resume_and_target_free_sampling(device,tmp_path):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    torch.manual_seed(6201);tr=make(device)
    tr.step();tr.step();saved=tr.state_dict()
    path=tmp_path/'step2.pt';torch.save(saved,path)
    expected_steps=[tr.step(),tr.step()];expected=tr.state_dict()
    fresh=make(device);fresh.load_state_dict(torch.load(path,weights_only=False))
    assert [fresh.step(),fresh.step()]==expected_steps;equal(fresh.state_dict(),expected)
    c=ResponseContext.from_task(task());other=ResponseContext.from_task(task(counts=(99,88),cell='hidden'))
    args=dict(steps=16,seed=6202,sigma_min=.01,negative_policy='preserve')
    rng=fresh.state_dict();a=sample_response(fresh.model,[c],**args)
    b=sample_response(tr.model,[other],**args)
    equal(a,b);equal(fresh.state_dict(),rng)
    generator=torch.Generator(device=device).manual_seed(6202)
    x=torch.randn((1,2),device=device,dtype=torch.float64,generator=generator)
    with torch.no_grad():
        for i in range(16):x=x+fresh.model([c],x,x.new_full((1,),i/16))/16
    assert torch.equal(a['log_expression'],x)


def test_v2_binding_mismatches_are_atomic():
    tr=make();tr.step();saved=tr.state_dict()
    for other in [make(sigma=.02),make(version=1),make(counts=(99,88)),make(batch_size=2)]:
        before=other.state_dict()
        with pytest.raises(ValueError,match='binding'):other.load_state_dict(saved)
        equal(other.state_dict(),before)
    with pytest.raises(ValueError,match='axis'):
        ResponseFlowTrainer(ResponseFlowDenoiserV2(SciPlexSharedModel(('B','A'),width=8)),tr.batch,batch_size=1)
