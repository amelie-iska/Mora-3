"""Exact complete LINCS axes; metadata inspection grants no scaling/admission."""
import copy
import hashlib
import json
import pytest
import torch
from torch import nn
from test_lincs_native import cases, project, api
from test_physical_owner_parameterization import model as physical
from mora3.model.lincs_model import SharedLincsModel


def make(cases, physical_owner=True):
    owner=physical(8)
    if not physical_owner:
        from test_annotation_parameterization import model as annotation
        owner=annotation(8)
    axes={project(api(),c).modality:c[7] for c in cases.values()}
    return SharedLincsModel(owner,axes)


def inventory(m):
    import mora3.model.parameterization as p
    assert hasattr(p,'lincs_parameter_ledger'), 'missing native LINCS owner ledger'
    return p.lincs_parameter_ledger(m,identifier_axis=('protein',))


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_native_counts_axes_aliases_and_numerical_nonmutation(cases,device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(36101)
    m=make(cases).to(device=device,dtype=dtype).eval()
    views=tuple(project(api(),c) for c in cases.values())
    reverse=tuple(v for v in views if v.direction=='profile_to_molecule')
    forward=tuple(v for v in views if v.direction=='molecule_to_profile')
    def values():return (*m(forward),m.condition(reverse))
    before=copy.deepcopy(m.state_dict());out=values();params=tuple(m.parameters())
    grad=torch.autograd.grad(sum(x.square().sum() for x in out),params,allow_unused=True)
    rng=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    ledger=inventory(m);w=8;n=1783+978
    shared=w*(w+7)+2*(6*w*w+10*w)+519*w+w*(w+1)+w+4*(w+1)+3*w+7*(w+1)
    assert ledger.total_numel==shared+3*n*w+2*w+n==sum(p.numel() for p in params)
    rows={a:e for e in ledger.parameters for a in e.aliases}
    assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
    for mod in m.axes:
        digest=hashlib.sha256(json.dumps(['lincs.native-source-axis.v1',mod,m.source_axes[mod]],
            sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        axis=rows['output.'+mod+'.weight'].axes[0]
        assert axis.kind=='finite' and axis.identity==digest
        assert rows['profile_input.'+mod+'.weight'].axes[1]==axis
        assert rows['profile_mask.'+mod+'.weight'].axes[1]==axis
        assert rows['output.'+mod+'.bias'].axes==(axis,)
        assert rows['output.'+mod+'.weight'].axes[1]==rows['shared.go.output.weight'].axes[1]
    assert torch.equal(rng,torch.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    after=values()
    for a,b in zip(out,after):torch.testing.assert_close(a,b,rtol=0,atol=0)
    g2=torch.autograd.grad(sum(x.square().sum() for x in after),params,allow_unused=True)
    for a,b in zip(grad,g2):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=0,atol=0)
    for key,t in m.state_dict().items():
        if isinstance(t,torch.Tensor):assert torch.equal(t,before[key])
        else:assert t==before[key]
    assert all(p.grad is None for p in params)
    from mora3.model.parameterization import compose_owner_parameter_ledgers,width_family
    b=SharedLincsModel(m.shared,m.source_axes).to(device=device,dtype=dtype)
    root=nn.ModuleDict({'a':m,'b':b})
    combined=compose_owner_parameter_ledgers(root,{'a':ledger,'b':inventory(b)})
    assert combined.total_numel==sum(p.numel() for p in root.parameters())
    assert next(e for e in combined.parameters if 'a.shared.physical.query.weight' in e.aliases).aliases==('a.shared.physical.query.weight','b.shared.physical.query.weight')
    with pytest.raises(ValueError,match='custom'):width_family(ledger,ledger,ledger)


@pytest.mark.parametrize('drift',['source','order','legacy','missing-axis','extra-axis','extra-module',
    'root-param','root-buffer','container-buffer','container-param','container-subclass','missing-layer',
    'extra-layer','shape','bias','size','layer-child','layer-buffer','layer-subclass','root-subclass','owner','tie'])
def test_native_axis_owner_and_structure_drift_refused(cases,drift):
    m=make(cases);h=m.output['L1000']
    if drift=='source':m.source_axes['L1000']['extra']='drift'
    elif drift=='order':m.axes['L1000']=tuple(reversed(m.axes['L1000']))
    elif drift=='legacy':m.axes['L1000']=('legacy',)*978
    elif drift=='missing-axis':del m.source_axes['L1000']
    elif drift=='extra-axis':m.axes['foreign']=('a',)
    elif drift=='extra-module':m.extra=nn.Identity()
    elif drift=='root-param':m.extra=nn.Parameter(torch.ones(1))
    elif drift=='root-buffer':m.register_buffer('x',torch.ones(1))
    elif drift=='container-buffer':m.profile_mask.register_buffer('x',torch.ones(1))
    elif drift=='container-param':m.profile_input.register_parameter('x',nn.Parameter(torch.ones(1)))
    elif drift=='container-subclass':
        class Other(nn.ModuleDict):pass
        m.output.__class__=Other
    elif drift=='missing-layer':del m.profile_input['L1000']
    elif drift=='extra-layer':m.profile_mask['foreign']=nn.Linear(1,8)
    elif drift=='shape':h.weight=nn.Parameter(torch.zeros(978,9))
    elif drift=='bias':m.profile_mask['L1000'].bias=nn.Parameter(torch.zeros(8))
    elif drift=='size':h.out_features=977
    elif drift=='layer-child':h.extra=nn.Identity()
    elif drift=='layer-buffer':h.register_buffer('x',torch.ones(1))
    elif drift=='layer-subclass':
        class Other(nn.Linear):pass
        h.__class__=Other
    elif drift=='root-subclass':
        class Other(SharedLincsModel):pass
        m.__class__=Other
    elif drift=='owner':m.shared=nn.Identity()
    else:m.profile_input['L1000'].weight=h.weight
    # Check feature availability separately so guards cannot pass on missing API.
    import mora3.model.parameterization as p
    assert hasattr(p,'lincs_parameter_ledger'), 'missing native LINCS owner ledger'
    with pytest.raises(ValueError):inventory(m)


def test_exact_annotation_owner_and_compatible_input_mask_tie(cases):
    m=make(cases,False)
    ledger=inventory(m)
    w=8;n=1783+978
    shared=w*(w+7)+2*(6*w*w+10*w)+519*w+w*(w+1)+w+4*(w+1)
    assert ledger.total_numel==shared+3*n*w+2*w+n
    m.profile_mask['L1000'].weight=m.profile_input['L1000'].weight
    tied=inventory(m)
    assert tied.total_numel==ledger.total_numel-978*w
    row=next(e for e in tied.parameters if 'profile_input.L1000.weight' in e.aliases)
    assert row.aliases==('profile_input.L1000.weight','profile_mask.L1000.weight')
