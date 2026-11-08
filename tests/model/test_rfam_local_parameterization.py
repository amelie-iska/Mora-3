"""Local Rfam readout binds an explicit registry, without inherited ownership."""
import pytest
import torch
from torch import nn
from test_nucleotide_local_parameterization import model
from mora3.model.rfam_family import RfamFamilyModel
from mora3.model import parameterization as p

AXIS=('RF00001','RF00002','RF00003')
PIN='a'*64

def make(width=8):return RfamFamilyModel(model(width),AXIS,manifest_sha256=PIN)
def roles(m,axis=AXIS,pin=PIN):
    return p.rfam_family_local_parameter_roles(m,family_axis=axis,manifest_sha256=pin)

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('width',[8,12,16])
def test_local_readout_count_axes_and_state(device,width):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(29901);m=make(width).to(device).eval()
    before={n:t.clone() for n,t in m.named_parameters()};rng=torch.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    r=roles(m)
    assert set(r)=={'family_output.weight','family_output.bias'}
    axis=r['family_output.bias'].axes[0]
    assert axis.kind=='finite'
    assert r['family_output.weight']==p.ParameterRole('output',(axis,p.Axis('width','hidden')))
    ledger=p.parameter_ledger(nn.ModuleDict({'family_output':m.family_output}),r)
    assert ledger.total_numel==len(AXIS)*(width+1)
    assert torch.equal(rng,torch.get_rng_state()) and not m.training
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    assert all(torch.equal(t,before[n]) and t.grad is None for n,t in m.named_parameters())

@pytest.mark.parametrize('drift',['axis-order','axis-list','axis-duplicate','axis-invalid','pin','pin-invalid',
    'extra-module','extra-parameter','root-buffer','output-buffer','output-child','output-subclass',
    'output-weight','output-bias','output-config','output-none','subclass'])
def test_local_drift_refuses(drift):
    m=make()
    if drift=='axis-order':m.family_axis=AXIS[::-1]
    elif drift=='axis-list':m.family_axis=list(AXIS)
    elif drift=='axis-duplicate':m.family_axis=(AXIS[0],)*3
    elif drift=='axis-invalid':m.family_axis=('bad',)
    elif drift=='pin':m.manifest_sha256='b'*64
    elif drift=='pin-invalid':m.manifest_sha256='bad'
    elif drift=='extra-module':m.extra=nn.Identity()
    elif drift=='extra-parameter':m.extra=nn.Parameter(torch.ones(1))
    elif drift=='root-buffer':m.register_buffer('extra',torch.ones(1))
    elif drift=='output-buffer':m.family_output.register_buffer('extra',torch.ones(1))
    elif drift=='output-child':m.family_output.extra=nn.Identity()
    elif drift=='output-subclass':
        class Other(nn.Linear):pass
        m.family_output.__class__=Other
    elif drift=='output-weight':m.family_output.weight=nn.Parameter(torch.ones(3,9))
    elif drift=='output-bias':m.family_output.bias=nn.Parameter(torch.ones(4))
    elif drift=='output-config':m.family_output.out_features=4
    elif drift=='output-none':m.family_output.bias=None
    elif drift=='subclass':
        class Other(RfamFamilyModel):pass
        m.__class__=Other
    with pytest.raises(ValueError):roles(m)

def test_registry_identity_and_caller_validation():
    m=make();first=roles(m)['family_output.bias'].axes
    m.manifest_sha256='b'*64
    assert roles(m,pin='b'*64)['family_output.bias'].axes!=first
    m.family_axis=('RF00004','RF00005','RF00006')
    assert roles(m,axis=m.family_axis,pin='b'*64)['family_output.bias'].axes!=first
    for axis,pin in ((list(m.family_axis),'b'*64),((),PIN),(AXIS,'bad')):
        with pytest.raises(ValueError):roles(m,axis,pin)
