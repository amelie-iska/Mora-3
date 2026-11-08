"""Complete explicit GOA root composition rejects omissions and conflicting ties."""
import copy
import pytest
import torch
from test_goa_cooperative import build
from mora3.model import parameterization as p


def owners(m):
    ids=('context','observation')
    return dict(native=p.cooperative_grayscale_parameter_ledger(m.native),
        annotation=p.physical_parameter_ledger(m.annotation,identifier_axis=ids),
        flow=p.response_parameter_ledger(m.flow),
        residue=p.residue_parameter_ledger(m.residue,identifier_axis=ids),
        profile=p.molecule_profile_parameter_ledger(m.profile,identifier_axis=ids),
        decoder=p.selfies_decoder_parameter_ledger(m.decoder,identifier_axis=ids),
        tox21=p.tox21_parameter_ledger(m.tox21,identifier_axis=ids),
        nucleotide=p.nucleotide_decoder_parameter_ledger(m.nucleotide,identifier_axis=ids),
        rfam=p.rfam_family_parameter_ledger(m.rfam,identifier_axis=ids,
            family_axis=m.rfam.family_axis,manifest_sha256=m.rfam.manifest_sha256))


def inventory(m,o=None,binding=None):
    return p.goa_cooperative_parameter_ledger(m,owners=owners(m) if o is None else o,
        axis_ids=m.goa.axis_ids,source_binding=m.goa.get_extra_state() if binding is None else binding)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_complete_aliases_counts_and_read_only(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    torch.manual_seed(30301);m,v=build(tmp_path,device);m.eval()
    before=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    o=owners(m);base={id(t):t for n,c in m.named_children() if n!='goa' for t in c.parameters()}
    ledger=inventory(m,o)
    assert ledger.total_numel==sum(t.numel() for t in base.values())+len(v.axis_ids)*5
    assert ledger.total_numel==sum(t.numel() for t in m.parameters())
    rows={a:e for e in ledger.parameters for a in e.aliases}
    assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
    assert rows['annotation.backbone.feature.weight'] is rows['native.response.backbone.feature.weight']
    assert rows['goa.output.weight'].axes[1]==rows['annotation.payload.embedding.weight'].axes[1]
    assert torch.equal(rng,torch.get_rng_state()) and not m.training
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    for n,t in m.state_dict().items():
        if torch.is_tensor(t):assert torch.equal(t,before[n])
        else:assert t==before[n]
    assert all(t.grad is None for t in m.parameters())
    with pytest.raises(ValueError,match='custom'):p.width_family(ledger,ledger,ledger)


@pytest.mark.parametrize('drift',['missing','stale','extra','root-buffer','root-param','binding','tie','subclass'])
def test_composition_refuses_drift(tmp_path,drift):
    m,_=build(tmp_path,'cpu');o=owners(m);binding=m.goa.get_extra_state()
    if drift=='missing':o.pop('rfam')
    elif drift=='stale':m.residue.head.output.weight.requires_grad_(False)
    elif drift=='extra':m.extra=torch.nn.Identity()
    elif drift=='root-buffer':m.register_buffer('extra',torch.ones(1))
    elif drift=='root-param':m.extra=torch.nn.Parameter(torch.ones(1))
    elif drift=='binding':binding['generation_eligible']=True
    elif drift=='tie':m.goa.output.weight=m.annotation.go.output.weight
    elif drift=='subclass':
        class Other(type(m)):pass
        m.__class__=Other
    with pytest.raises(ValueError):inventory(m,o,binding)
