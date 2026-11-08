"""Complete Rfam inventory retains strict inherited validation and alias accounting."""
import pytest
import torch
from torch import nn
from test_annotation_parameterization import model as annotation
from test_physical_owner_parameterization import model as physical
from mora3.model.nucleotide_decoder import NucleotideDecoder
from mora3.model.rfam_family import RfamFamilyModel
from mora3.model import parameterization as p
AXIS=('RF00001','RF00002','RF00003')
PIN='a'*64

def make(owner=physical,w=8):return RfamFamilyModel(NucleotideDecoder(owner(w)),AXIS,manifest_sha256=PIN)
def inventory(m):return p.rfam_family_parameter_ledger(m,identifier_axis=('protein',),family_axis=AXIS,manifest_sha256=PIN)

@pytest.mark.parametrize('owner',[annotation,physical])
@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('w',[8,12,16])
def test_complete_counts_and_immutable_state(owner,device,w):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(30001);m=make(owner,w).to(device).eval()
    before={n:t.clone() for n,t in m.named_parameters()};rng=torch.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    ledger=inventory(m)
    shared=w*(w+7)+2*(6*w*w+10*w)+519*w+w*(w+1)+w+4*(w+1)
    if owner is physical:shared+=3*w+7*(w+1)
    v=m.codec.size
    assert ledger.total_numel==shared+6*w*w+(2*v+6)*w+v+len(AXIS)*(w+1)
    assert ledger.total_numel==sum(t.numel() for t in m.parameters())
    rows={a:e for e in ledger.parameters for a in e.aliases}
    assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
    assert rows['family_output.weight'].axes[0]!=rows['output.weight'].axes[0]
    assert rows['family_output.weight'].axes[1]==rows['shared.payload.embedding.weight'].axes[1]
    assert torch.equal(rng,torch.get_rng_state()) and not m.training
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    assert all(torch.equal(t,before[n]) and t.grad is None for n,t in m.named_parameters())
    with pytest.raises(ValueError,match='custom'):p.width_family(ledger,ledger,ledger)

@pytest.mark.parametrize('drift',['codec','gru','shared-buffer','physical-axis','go-axis','backbone','local-buffer','extra','family-axis','family-pin','subclass','conflict'])
def test_all_native_validation_remains_active(drift):
    m=make()
    if drift=='codec':m.codec.index={}
    elif drift=='gru':m.rnn.batch_first=False
    elif drift=='shared-buffer':m.shared.register_buffer('extra',torch.ones(1))
    elif drift=='physical-axis':m.shared.physical.parameter_axis=('pH','mass','temperature')
    elif drift=='go-axis':m.shared.go.axis_ids=('GO:0000001',)*2
    elif drift=='backbone':m.shared.backbone.feature=nn.Linear(7,8,bias=False)
    elif drift=='local-buffer':m.embedding.register_buffer('extra',torch.ones(1))
    elif drift=='extra':m.extra=nn.Identity()
    elif drift=='family-axis':m.family_axis=AXIS[::-1]
    elif drift=='family-pin':m.manifest_sha256='b'*64
    elif drift=='conflict':m.family_output.weight=m.output.weight
    elif drift=='subclass':
        class Other(RfamFamilyModel):pass
        m.__class__=Other
    with pytest.raises(ValueError):inventory(m)

def test_compatible_recurrent_alias_counts_once():
    m=make();m.rnn.weight_hh_l0=m.rnn.weight_ih_l0
    ledger=inventory(m);rows={a:e for e in ledger.parameters for a in e.aliases}
    assert rows['rnn.weight_hh_l0'].aliases==('rnn.weight_hh_l0','rnn.weight_ih_l0')
    assert ledger.total_numel==sum(t.numel() for t in m.parameters())
