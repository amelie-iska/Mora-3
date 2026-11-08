"""GOA output axes retain the complete native source/policy binding."""
import copy
import pytest
import torch
from torch import nn
from test_goa_objective import view
from mora3.model.uniprot_go import NativeGOAHead
from mora3.model import parameterization as p


def make(tmp_path, width=8):
    return NativeGOAHead(width, (view(tmp_path),))


def roles(m, axis=None, binding=None):
    return p.goa_local_parameter_roles(m, axis_ids=m.axis_ids if axis is None else axis,
        source_binding=m.get_extra_state() if binding is None else binding)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('width', [8, 12, 16])
def test_complete_local_axes_count_and_read_only(tmp_path, device, width):
    if device == 'cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(30201)
    m=make(tmp_path,width).to(device).eval()
    state=copy.deepcopy(m.state_dict()); rng=torch.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    r=roles(m); a=r['output.bias'].axes[0]
    assert a.kind=='finite'
    assert r=={'output.weight':p.ParameterRole('output',(a,p.Axis('width','hidden'))),
               'output.bias':p.ParameterRole('vector',(a,))}
    ledger=p.parameter_ledger(m,r)
    assert ledger.total_numel==len(m.axis_ids)*(width+1)
    assert torch.equal(rng,torch.get_rng_state()) and not m.training
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    for n,t in m.state_dict().items():
        if torch.is_tensor(t):assert torch.equal(t,state[n])
        else:assert t==state[n]
    assert all(t.grad is None for t in m.parameters())


@pytest.mark.parametrize('drift', ['axis-order','axis-list','axis-duplicate','axis-invalid',
    'binding-axis','policy','generation','receipt','hash-set','extra-module','extra-parameter',
    'root-buffer','output-buffer','output-child','output-subclass','weight','bias','config',
    'no-bias','subclass','binding-version','binding-extra'])
def test_drift_refuses(tmp_path, drift):
    m=make(tmp_path);axis=m.axis_ids;binding=m.get_extra_state()
    if drift=='axis-order':m.axis_ids=m.axis_ids[::-1]
    elif drift=='axis-list':m.axis_ids=list(m.axis_ids)
    elif drift=='axis-duplicate':m.axis_ids=(axis[0],)*2
    elif drift=='axis-invalid':m.axis_ids=('bad',axis[1])
    elif drift=='binding-axis':m._native_binding['axis_ids']=axis[::-1]
    elif drift=='policy':m.policy='old'
    elif drift=='generation':m._native_binding['generation_eligible']=True
    elif drift=='receipt':m._native_binding['packets'][0]['subject']='UniProtKB:other'
    elif drift=='hash-set':m._packet_hashes=frozenset(('a'*64,))
    elif drift=='extra-module':m.extra=nn.Identity()
    elif drift=='extra-parameter':m.extra=nn.Parameter(torch.ones(1))
    elif drift=='root-buffer':m.register_buffer('extra',torch.ones(1))
    elif drift=='output-buffer':m.output.register_buffer('extra',torch.ones(1))
    elif drift=='output-child':m.output.extra=nn.Identity()
    elif drift=='output-subclass':
        class Other(nn.Linear):pass
        m.output.__class__=Other
    elif drift=='weight':m.output.weight=nn.Parameter(torch.ones(2,9))
    elif drift=='bias':m.output.bias=nn.Parameter(torch.ones(3))
    elif drift=='config':m.output.out_features=3
    elif drift=='no-bias':m.output.bias=None
    elif drift=='subclass':
        class Other(NativeGOAHead):pass
        m.__class__=Other
    elif drift=='binding-version':m._native_binding['version']=2
    elif drift=='binding-extra':m._native_binding['extra']='unknown'
    with pytest.raises(ValueError):roles(m,axis,binding)


def test_caller_binding_and_pool_identity(tmp_path):
    m=make(tmp_path); original=m.get_extra_state(); first=roles(m)
    for axis,binding in ((list(m.axis_ids),original),((),original),
                         (m.axis_ids,{}),(m.axis_ids,dict(original,policy='old'))):
        with pytest.raises(ValueError):roles(m,axis,binding)
    # A different exact pool changes axis identity, even with identical label IDs.
    other=copy.deepcopy(m)
    other._native_binding['packets'][0]['packet_sha256']='b'*64
    other._packet_hashes=frozenset(('b'*64,))
    assert roles(other)['output.bias'].axes!=first['output.bias'].axes
    assert m.get_extra_state()==original


def test_incompatible_local_tie_refuses(tmp_path):
    m=make(tmp_path,2);m.output.bias=nn.Parameter(m.output.weight[0])
    # A true identity tie cannot satisfy both matrix and vector shape declarations.
    m.output._parameters['bias']=m.output.weight
    with pytest.raises(ValueError):roles(m)


def test_equal_python_values_do_not_hide_binding_type_drift(tmp_path):
    m=make(tmp_path);binding=m.get_extra_state()
    m._native_binding['version']=True
    with pytest.raises(ValueError):roles(m,binding=binding)


@pytest.mark.parametrize('bad',['empty-pool','hash','subject','status','objective','unknown-negative'])
def test_self_consistent_malformed_pool_refuses(tmp_path,bad):
    m=make(tmp_path);r=m._native_binding['packets'][0]
    if bad=='empty-pool':m._native_binding['packets']=[]
    elif bad=='hash':r['packet_sha256']='bad'
    elif bad=='subject':r['subject']='UniProtKB:'
    elif bad=='status':r['status']='future'
    elif bad=='objective':r['objective']['policy']='old'
    elif bad=='unknown-negative':r['objective']['unknown_is_negative']=True
    m._packet_hashes=frozenset(r['packet_sha256'] for r in m._native_binding['packets'])
    with pytest.raises(ValueError):roles(m)
