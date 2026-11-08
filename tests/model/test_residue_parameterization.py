"""Local residue axes and explicit refusal of unqualified convolution scaling."""
import hashlib
import json
import pytest
import torch
from torch import nn
from mora3.model.residue_objective import ContextualResidueHead
from mora3.model.parameterization import Axis, parameter_ledger, width_family


def roles(m):
    from mora3.model.parameterization import residue_head_parameter_roles
    return residue_head_parameter_roles(m)


def head(w=8, axis=('Region','Disulfide bond')):
    return ContextualResidueHead(w,axis)


def finite(namespace,names):
    return Axis('finite',hashlib.sha256(json.dumps([namespace,names],ensure_ascii=True,
        separators=(',',':')).encode()).hexdigest())


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_local_orientation_count_readonly_and_unqualified_scaling(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(27701)
    m=head().to(device); before={n:p.clone() for n,p in m.named_parameters()}
    rng=torch.random.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    r=roles(m); h=Axis('width','hidden')
    f=finite('residue.feature-types.v1',m.feature_axis)
    k=finite('residue.local-offsets.v1',('-1','0','1'))
    assert r['context.weight'].role=='hidden' and r['context.weight'].axes==(h,h)
    assert r['context.bias'].role=='vector' and r['context.bias'].axes==(h,)
    assert r['local.weight'].role=='custom' and r['local.weight'].axes==(h,h,k)
    assert r['local.bias'].role=='vector' and r['local.bias'].axes==(h,)
    assert r['output.weight'].role=='output' and r['output.weight'].axes==(f,h)
    assert r['output.bias'].role=='vector' and r['output.bias'].axes==(f,)
    assert parameter_ledger(m,r).total_numel==4*8**2+2*8+2*(8+1)
    assert torch.equal(rng,torch.random.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    for n,p in m.named_parameters():assert torch.equal(p,before[n]) and p.grad is None
    with pytest.raises(ValueError,match='custom'):
        width_family(*(parameter_ledger(head(w),roles(head(w))) for w in (8,12,16)))


def test_ordered_finite_identity_preserved_across_widths():
    r=roles(head())
    assert r==roles(head(12))==roles(head(16))
    other=roles(head(axis=('Disulfide bond','Region')))
    assert r['output.weight']!=other['output.weight']
    assert r['local.weight']==other['local.weight']


@pytest.mark.parametrize('drift',['extra','module','root-subclass','linear-subclass','conv-subclass',
    'context-shape','output-shape','local-shape','local-padding','local-stride','local-dilation',
    'local-groups','local-mode','local-transposed','local-output-padding',
    'no-bias','axis-list','axis-duplicate','axis-empty','width-odd','width-bool','width-mismatch'])
def test_configuration_and_shape_drift_refused(drift):
    m=head()
    if drift=='extra':m.register_parameter('extra',nn.Parameter(torch.ones(1)))
    if drift=='module':m.extra=nn.Identity()
    if drift=='root-subclass':
        class Head(ContextualResidueHead):pass
        m=Head(8,('Region',))
    if drift=='linear-subclass':
        class Linear(nn.Linear):pass
        m.context=Linear(8,8)
    if drift=='conv-subclass':
        class Conv(nn.Conv1d):pass
        m.local=Conv(8,8,3,padding=1)
    if drift=='context-shape':m.context=nn.Linear(9,8)
    if drift=='output-shape':m.output=nn.Linear(8,3)
    if drift=='local-shape':m.local=nn.Conv1d(8,8,5,padding=2)
    for attr,value in [('padding',(0,)),('stride',(2,)),('dilation',(2,)),('groups',2),
                       ('mode','reflect'),('transposed',True),('output-padding',(1,))]:
        if drift=='local-'+attr:setattr(m.local,{'mode':'padding_mode','output-padding':'output_padding'}.get(attr,attr),value)
    if drift=='no-bias':m.local.bias=None
    if drift=='axis-list':m.feature_axis=list(m.feature_axis)
    if drift=='axis-duplicate':m.feature_axis=('Region','Region')
    if drift=='axis-empty':m.feature_axis=('Region',' ')
    if drift=='width-odd':m.width=7
    if drift=='width-bool':m.width=True
    if drift=='width-mismatch':m.width=10
    with pytest.raises(ValueError):roles(m)
