"""Complete decoder/profile ownership must preserve all aliases and custom gates."""
import pytest
import torch
from torch import nn
from test_profile_owner_parameterization import model as annotation
from test_profile_physical_owner_parameterization import model as physical
from mora3.model.profile_decoder import ProfileSelfiesDecoder, SelfiesCodec
from mora3.model import parameterization as p


def inventory(m):
    return p.selfies_decoder_parameter_ledger(m,identifier_axis=('protein',))


@pytest.mark.parametrize('owner',[annotation,physical])
@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('w',[8,12,16])
def test_complete_count_and_readonly(owner,device,w):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(29201)
    m=ProfileSelfiesDecoder(owner(w),SelfiesCodec(('CCO','CN')),max_tokens=32).to(device).eval()
    before={n:t.clone() for n,t in m.named_parameters()}
    rng=torch.random.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    ledger=inventory(m);v=m.codec.size
    shared=w*(w+7)+2*(6*w*w+10*w)+519*w+w*(w+1)+w+4*(w+1)
    if owner is physical:shared+=3*w+7*(w+1)
    assert ledger.total_numel==shared+5520*w+2*w+2760+2*v*w+v+6*w*w+6*w
    assert ledger.total_numel==sum(t.numel() for t in m.parameters())
    rows={a:e for e in ledger.parameters for a in e.aliases}
    assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
    assert rows['rnn.weight_ih_l0'].role=='custom'
    assert torch.equal(rng,torch.random.get_rng_state()) and not m.training
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    assert all(torch.equal(t,before[n]) and t.grad is None for n,t in m.named_parameters())
    with pytest.raises(ValueError,match='custom'):p.width_family(ledger,ledger,ledger)


def test_compatible_cross_owner_alias_and_conflicting_role():
    m=ProfileSelfiesDecoder(annotation(),SelfiesCodec(('CCO',)),max_tokens=32)
    m.profiles.profile_input['L1000'].bias=m.profiles.shared.payload.projection.bias
    ledger=inventory(m)
    assert any(len(e.aliases)==2 for e in ledger.parameters)
    m.output.weight=m.embedding.weight
    with pytest.raises(ValueError,match='conflicting'):inventory(m)


@pytest.mark.parametrize('drift',['axis','parent-extra','local-extra','owner-buffer','root-buffer'])
def test_both_owner_validators_remain_active(drift):
    m=ProfileSelfiesDecoder(physical(),SelfiesCodec(('CCO',)),max_tokens=32)
    if drift=='axis':m.profiles.axes['L1000']=('x',)
    if drift=='parent-extra':m.profiles.extra=nn.Identity()
    if drift=='local-extra':m.extra=nn.Parameter(torch.ones(1))
    if drift=='owner-buffer':m.profiles.register_buffer('unexpected',torch.ones(1))
    if drift=='root-buffer':m.register_buffer('unexpected',torch.ones(1))
    with pytest.raises(ValueError):inventory(m)
