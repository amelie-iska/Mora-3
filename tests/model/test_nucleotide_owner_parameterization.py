"""Complete native nucleotide owners: exact axes, aliases and immutable inventory."""
import pytest
import torch
from torch import nn
from test_annotation_parameterization import model as annotation
from test_physical_owner_parameterization import model as physical
from mora3.model.nucleotide_decoder import NucleotideDecoder
from mora3.model import parameterization as p


def inventory(m, identifiers=('protein',)):
    return p.nucleotide_decoder_parameter_ledger(m, identifier_axis=identifiers)


@pytest.mark.parametrize('owner', [annotation, physical])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('w', [8, 12, 16])
def test_complete_owner_count_axes_and_readonly(owner, device, w):
    if device == 'cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(29701)
    m = NucleotideDecoder(owner(w)).to(device).eval()
    before = {n:t.clone() for n,t in m.named_parameters()}
    rng = torch.get_rng_state().clone()
    gpu = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
    ledger = inventory(m)
    shared = w*(w+7)+2*(6*w*w+10*w)+519*w+w*(w+1)+w+4*(w+1)
    if owner is physical:shared += 3*w+7*(w+1)
    v = m.codec.size
    assert ledger.total_numel == shared+6*w*w+(2*v+6)*w+v
    assert ledger.total_numel == sum(t.numel() for t in m.parameters())
    rows = {a:e for e in ledger.parameters for a in e.aliases}
    assert set(rows) == set(dict(m.named_parameters(remove_duplicate=False)))
    assert rows['embedding.weight'].axes[1] == rows['shared.payload.embedding.weight'].axes[1]
    assert rows['output.weight'].axes[0] != rows['shared.go.output.weight'].axes[0]
    assert rows['rnn.weight_ih_l0'].role == 'custom'
    assert torch.equal(rng, torch.get_rng_state()) and not m.training
    if gpu is not None:assert torch.equal(gpu, torch.cuda.get_rng_state())
    assert all(torch.equal(t,before[n]) and t.grad is None for n,t in m.named_parameters())
    with pytest.raises(ValueError, match='custom'):p.width_family(ledger,ledger,ledger)


def test_compatible_cross_owner_alias_and_conflicting_roles():
    m = NucleotideDecoder(physical())
    m.rnn.weight_hh_l0 = m.rnn.weight_ih_l0
    ledger = inventory(m)
    rows = {a:e for e in ledger.parameters for a in e.aliases}
    assert rows['rnn.weight_hh_l0'].aliases == ('rnn.weight_hh_l0','rnn.weight_ih_l0')
    assert ledger.total_numel == sum(t.numel() for t in m.parameters())
    m.embedding.weight = m.shared.payload.embedding.weight
    with pytest.raises(ValueError):inventory(m)


@pytest.mark.parametrize('drift', ['owner-buffer','child-buffer','owner-extra','root-buffer',
    'root-extra','physical-axis','annotation-axis','backbone-shape','identifier-list',
    'identifier-rank','subclass','decoder-tie'])
def test_both_owner_and_local_validation_remain_active(drift):
    m = NucleotideDecoder(physical());ids = ('protein',)
    if drift == 'owner-buffer':m.shared.register_buffer('unexpected',torch.ones(1))
    if drift == 'child-buffer':m.shared.physical.register_buffer('unexpected',torch.ones(1))
    if drift == 'owner-extra':m.shared.extra = nn.Identity()
    if drift == 'root-buffer':m.register_buffer('unexpected',torch.ones(1))
    if drift == 'root-extra':m.extra = nn.Parameter(torch.ones(1))
    if drift == 'physical-axis':m.shared.physical.parameter_axis = ('pH','mass','temperature')
    if drift == 'annotation-axis':m.shared.go.axis_ids = ('GO:0000001',)*2
    if drift == 'backbone-shape':m.shared.backbone.feature = nn.Linear(7,8,bias=False)
    if drift == 'identifier-list':ids = ['protein']
    if drift == 'identifier-rank':ids = ('protein','other')
    if drift == 'subclass':
        class Other(type(m.shared)):pass
        m.shared.__class__ = Other
    if drift == 'decoder-tie':m.output.weight = m.embedding.weight
    with pytest.raises(ValueError):inventory(m,ids)
