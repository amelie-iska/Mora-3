"""Full registry cooperation: exact shared ownership and reference gradients."""
import copy
import pytest
import torch
from mora3.model import nucleotide_cooperative as module
from test_nucleotide_cooperative import build as base_build
from test_annotation_axes import reader, REAL
from annotation_fixture import fixture_registry
from test_rfam_family_objective import view, AXIS
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.model.physical_trainer import SharedPhysicalModel


def build(device='cpu', bad=None):
    assert hasattr(module, 'RfamCooperativeModel'), 'shared full-axis Rfam interface missing'
    base=base_build(device).model
    r=fixture_registry()
    annotation=SharedPhysicalModel.from_registry(
        AnnotationVocabulary(base.annotation.payload.get_extra_state()['vocabulary']),r,
        parameter_axis=base.annotation.physical.parameter_axis,
        method_axis=base.annotation.physical.method_axis,
        width=4,heads=2,layers=2,backbone=base.native.response.backbone).to(device)
    if bad=='axis':annotation.go.axis_ids=('GO:0000001',)
    if bad=='binding':annotation._configuration.pop('annotation_registry')
    return module.RfamCooperativeModel(base.native,annotation,base.residue.head.feature_axis,
        base.profile.axes,codec=base.decoder.codec,max_tokens=64,
        annotation_registry=r,family_axis=AXIS,rfam_manifest_sha256='a'*64).to(device)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_full_axes_and_exact_decoder_parameter_ownership(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    m=build(device)
    assert len(m.annotation.go.axis_ids)==28079 and len(m.annotation.ec.axis_ids)==5124
    assert m.rfam.shared is m.annotation
    assert m.rfam.rnn is m.nucleotide.rnn and m.rfam.output is m.nucleotide.output
    assert m.rfam.embedding is m.nucleotide.embedding
    assert m.rfam.shared.backbone is m.native.image.backbone is m.profile.backbone
    old=set(map(id,m.nucleotide.parameters()))
    assert set(map(id,m.rfam.parameters()))-old==set(map(id,m.rfam.family_output.parameters()))
    ps=list(m.parameters());assert len(ps)==len(set(map(id,ps)))
    opt=torch.optim.Adam(ps)
    assert len(opt.param_groups[0]['params'])==len(ps)
    state=copy.deepcopy(m.state_dict())
    assert state['_extra_state']['annotation']['annotation_registry']==reader(REAL).binding()
    state['_extra_state']['rfam']['manifest_sha256']='b'*64
    with pytest.raises(ValueError,match='configuration'):m.set_extra_state(state['_extra_state'])


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('direction',['prediction','generation'])
def test_rfam_reference_loss_every_shared_gradient_and_exclusion(device,direction):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.backends.cudnn.allow_tf32=False
    m=build(device).double();f=m.rfam;vs=(view(direction),view(direction,seq='Nu',family='RF00001'))
    if direction=='prediction':
        z=f.predict(vs)
        oracle=(torch.logaddexp(z.new_zeros(()),-z[0,1])+torch.logaddexp(z.new_zeros(()),-z[1,0]))/2
        assert torch.equal(z,f.predict((view(family='RF00003'),vs[1])))
    else:
        prefix,target=f.teacher_batch(vs);z=f(tuple(v.inputs() for v in vs),prefix)
        logp=z.log_softmax(-1).gather(-1,target[...,None]).squeeze(-1)
        oracle=-logp[target!=f.codec.pad].mean()
        assert torch.equal(f.condition((vs[0].inputs(),)),f.condition((view('generation',seq='N'*9000).inputs(),)))
    actual=getattr(f,direction+'_objective')(vs)
    torch.testing.assert_close(actual,oracle,atol=1e-12,rtol=1e-12)
    ps=tuple(m.parameters())
    a=torch.autograd.grad(actual,ps,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(oracle,ps,allow_unused=True)
    for x,y in zip(a,b):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-10)
    getattr(f,direction+'_objective')(vs).backward()
    assert m.native.response.backbone.feature.weight.grad.abs().sum()>0
    assert m.annotation.payload.embedding.weight.grad.abs().sum()>0
    if direction=='prediction':assert f.family_output.weight.grad[2].count_nonzero()==0
    else:assert m.nucleotide.rnn.weight_ih_l0.grad.abs().sum()>0


@pytest.mark.parametrize('bad',['axis','binding'])
def test_incompatible_annotation_registry_rejected(bad):
    with pytest.raises(ValueError,match='registry'):build(bad=bad)
