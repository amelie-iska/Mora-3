"""Full native molecule/profile fixed-axis ownership, never scaling qualification."""
import pytest
import torch
from torch import nn
from test_annotation_parameterization import model as annotation
from mora3.model.molecule_profile import SharedMoleculeProfileModel


def model(width=8):
    return SharedMoleculeProfileModel(annotation(width), {
        m:tuple(f'{m}:{i}' for i in range(n)) for m,n in [('CellPainting',1783),('L1000',977)]})


def inventory(m):
    from mora3.model.parameterization import molecule_profile_parameter_ledger
    return molecule_profile_parameter_ledger(m, identifier_axis=('protein',))


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('width',[8,12,16])
def test_independent_count_axes_and_readonly(device,width):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(28901)
    m=model(width).to(device).eval()
    state={n:p.clone() for n,p in m.named_parameters()}
    rng=torch.random.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    ledger=inventory(m);w=width
    shared=w*(w+7)+2*(6*w*w+10*w)+519*w+w*(w+1)+w+4*(w+1)
    assert ledger.total_numel==shared+2*2760*w+2*w+2760==sum(p.numel() for p in m.parameters())
    rows={a:e for e in ledger.parameters for a in e.aliases}
    for name,n in [('CellPainting',1783),('L1000',977)]:
        inp=rows[f'profile_input.{name}.weight'];out=rows[f'output.{name}.weight']
        assert inp.role=='input' and out.role=='output'
        assert inp.shape==(w,n) and out.shape==(n,w)
        assert inp.axes==out.axes[::-1]
        assert inp.axes[1].kind=='finite' and inp.axes[0].kind=='width'
    assert rows['profile_input.CellPainting.weight'].axes[1]!=rows['profile_input.L1000.weight'].axes[1]
    assert torch.equal(rng,torch.random.get_rng_state()) and not m.training
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    for n,p in m.named_parameters():assert torch.equal(p,state[n]) and p.grad is None
    from mora3.model.parameterization import width_family
    with pytest.raises(ValueError,match='custom'):width_family(ledger,ledger,ledger)


def test_order_identity_and_conflicting_tie():
    m=model();a=inventory(m)
    m.axes['L1000']=m.axes['L1000'][::-1]
    b=inventory(m)
    assert a!=b and a.total_numel==b.total_numel
    m.output['L1000'].bias=m.profile_input['L1000'].bias
    with pytest.raises(ValueError):inventory(m)


@pytest.mark.parametrize('drift',['root','extra-module','extra-parameter','axis-list','axis-duplicate',
    'axis-short','axis-type','axis-extra','linear-type','shape','bias','shared'])
def test_drift_refuses(drift):
    m=model()
    if drift=='root':m=nn.Sequential(m)
    if drift=='extra-module':m.extra=nn.Identity()
    if drift=='extra-parameter':m.extra=nn.Parameter(torch.ones(1))
    if drift=='axis-list':m.axes['L1000']=list(m.axes['L1000'])
    if drift=='axis-duplicate':m.axes['L1000']=('x',)*977
    if drift=='axis-short':m.axes['L1000']=('x',)
    if drift=='axis-type':m.axes['L1000']=(None,)+m.axes['L1000'][1:]
    if drift=='axis-extra':m.axes['other']=('x',)
    if drift=='linear-type':m.output['L1000']=nn.Identity()
    if drift=='shape':m.output['L1000'].weight=nn.Parameter(torch.ones(977,9))
    if drift=='bias':m.output['L1000'].bias=None
    if drift=='shared':m.shared=nn.Identity()
    with pytest.raises(ValueError):inventory(m)
