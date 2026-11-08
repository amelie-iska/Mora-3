"""Construction metadata for legacy grayscale heads; no scaling qualification."""
import pytest
import torch
from torch import nn
from mora3.model.morphology import MorphologyForeground, MorphologyInstance
from mora3.model.parameterization import width_family


def ledger(m):
    from mora3.model.parameterization import morphology_parameter_ledger
    return morphology_parameter_ledger(m)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('cls',[MorphologyForeground,MorphologyInstance])
def test_native_count_axes_readonly(device,cls):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(28501)
    for w in (8,12,16):
        m=cls(w).to(device); before={n:p.clone() for n,p in m.named_parameters()}
        rng=torch.random.get_rng_state().clone()
        gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
        x=ledger(m); entries={a:e for e in x.parameters for a in e.aliases}
        # Independent sums: pixel convolutions + one-layer TokenGT + output conv.
        pixels=(9*(w+51*w*w)+20*w if cls is MorphologyInstance else 9*(w+w*w)+2*w)
        outputs=3 if cls is MorphologyInstance else 1
        assert x.total_numel==pixels+(w*(w+7)+6*w*w+10*w)+outputs*(2*w+1)
        assert entries['head.weight'].shape==(outputs,2*w,1,1)
        assert entries['head.weight'].axes[1].identity=='image.concat-local-context'
        assert entries['head.weight'].role=='custom'
        assert entries['backbone.feature.weight'].role=='hidden'
        assert torch.equal(rng,torch.random.get_rng_state())
        if gpu is not None: assert torch.equal(gpu,torch.cuda.get_rng_state())
        for n,p in m.named_parameters(): assert torch.equal(p,before[n]) and p.grad is None


def test_output_semantics_and_custom_refusal():
    a=ledger(MorphologyForeground(8)); b=ledger(MorphologyInstance(8))
    def head(x): return next(e for e in x.parameters if 'head.weight' in e.aliases)
    assert head(a).axes[0]!=head(b).axes[0]
    for cls in (MorphologyForeground,MorphologyInstance):
        with pytest.raises(ValueError,match='custom'):
            width_family(*(ledger(cls(w)) for w in (8,12,16)))


@pytest.mark.parametrize('drift',['type','normalization','extra','child','head_kernel','head_stride',
    'head_bias','head_shape','pixel_activation','pixel_padding','backbone_width','instance_width'])
def test_drift_refuses(drift):
    m=MorphologyInstance(8) if drift=='instance_width' else MorphologyForeground(8)
    if drift=='type': m=nn.Identity()
    if drift=='normalization': m.normalization='invented'
    if drift=='extra': m.register_parameter('extra',nn.Parameter(torch.ones(1)))
    if drift=='child': m.extra=nn.Identity()
    if drift=='head_kernel': m.head=nn.Conv2d(16,1,3)
    if drift=='head_stride': m.head.stride=(2,2)
    if drift=='head_bias': m.head.bias=None
    if drift=='head_shape': m.head.weight=nn.Parameter(torch.ones(2,16,1,1))
    if drift=='pixel_activation': m.pixels[1]=nn.ReLU()
    if drift=='pixel_padding': m.pixels[0].padding=(0,0)
    if drift=='backbone_width': m.backbone.feature=nn.Linear(8,12)
    if drift=='instance_width': m.width=12
    with pytest.raises(ValueError): ledger(m)
