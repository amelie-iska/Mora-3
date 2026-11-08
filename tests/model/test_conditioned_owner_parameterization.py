import pytest
import torch
from torch import nn
from mora3.model import parameterization as p
from mora3.model.color_conditioning import ConditionedColorInstance
from mora3.model.color_semantic import ColorInstancePredictor


def owner(w=8):
    return ConditionedColorInstance(ColorInstancePredictor(w,slots=3,max_output_bytes=24000000,alpha_policy='require-opaque'),id_width=2)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('w',[8,12,16])
def test_complete_count_aliases_and_preservation(device,w):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(28301)
    m=owner(w).to(device)
    state={n:t.clone() for n,t in m.named_parameters()}
    ledger=p.conditioned_instance_parameter_ledger(m,channel_axis=('r','g','b'),identifier_axis=('a','b'))
    # Existing color owner + independent rank2 incidence + native byte payload + offset.
    color=p.instance_parameter_ledger(m.color,channel_axis=('r','g','b'),identifier_axis=('image',))
    expected=color.total_numel+4*w+517*w+w*(w+1)+w+3*(w+1)
    assert ledger.total_numel==expected==sum(t.numel() for t in m.parameters())
    entry=next(e for e in ledger.parameters if 'color.backbone.feature.weight' in e.aliases)
    assert 'context.graph.feature.weight' in entry.aliases
    assert all(torch.equal(t,state[n]) and t.grad is None for n,t in m.named_parameters())
    assert any(e.role=='custom' for e in ledger.parameters)


@pytest.mark.parametrize('drift',['alias','config','payload','pool','token','shape','extra','rank'])
def test_drift_refuses(drift):
    m=owner()
    if drift=='alias': m.context.graph.feature=nn.Linear(8,8)
    elif drift=='config': m.context.config['width']=12
    elif drift=='payload': m.context.payload.offset=258
    elif drift=='pool': m.context.payload.pooling='bag'
    elif drift=='token': m.context.payload.codec.tokenizer._controls={'other':256}
    elif drift=='shape': m.context.payload.projection=nn.Linear(8,8)
    elif drift=='extra': m.context.extra=nn.Linear(8,8)
    else: m.config['id_width']=1
    with pytest.raises(ValueError):
        p.conditioned_instance_parameter_ledger(m,channel_axis=('r','g','b'),identifier_axis=('a','b'))
