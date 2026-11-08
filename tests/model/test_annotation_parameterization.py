"""Local annotation axes: ordered labels/tokens, exact ownership, read-only."""
import json
import pytest
import torch
from torch import nn
from mora3.model.annotation_multihead import SharedAnnotationModel
from mora3.model.annotation_vocabulary import AnnotationVocabulary


def model(width=8, learned=('4142','4344'), go=('GO:0000001','GO:0000002')):
    vocab=AnnotationVocabulary(json.dumps(dict(learned_hex=learned,controls=['pad','node','edge'])))
    return SharedAnnotationModel(vocab,go,('1.1.1.1','2.1.1.1'),width=width)


def roles(m):
    from mora3.model.parameterization import annotation_head_parameter_roles
    return annotation_head_parameter_roles(m)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_axes_orientation_count_and_readonly(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(27301)
    m=model().to(device)
    before={n:p.clone() for n,p in m.named_parameters()}
    rng=torch.random.get_rng_state().clone()
    r=roles(m)
    assert set(r)=={n for n,_ in m.named_parameters() if not n.startswith('backbone.')}
    assert r['payload.embedding.weight'].role=='embedding'
    assert r['payload.embedding.weight'].axes[0].kind=='finite'
    assert r['payload.projection.weight'].role=='custom'
    assert r['go.output.weight'].role==r['ec.output.weight'].role=='output'
    assert r['go.output.weight'].axes[1]==r['payload.embedding.weight'].axes[1]
    assert r['go.output.weight'].axes[0]!=r['ec.output.weight'].axes[0]
    assert torch.equal(rng,torch.random.get_rng_state())
    assert roles(model(12))==r
    assert roles(model(learned=('4344','4142')))['payload.embedding.weight']!=r['payload.embedding.weight']
    assert roles(model(go=('GO:0000002','GO:0000001')))['go.output.weight']!=r['go.output.weight']
    # Repeat map itself must preserve CPU and device RNG, parameters and gradients.
    rng=torch.random.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    assert roles(m)==r
    assert torch.equal(rng,torch.random.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    for n,p in m.named_parameters():assert torch.equal(p,before[n]) and p.grad is None
    local=sum(p.numel() for n,p in m.named_parameters() if not n.startswith('backbone.'))
    # 519 embedding rows, projection w*(w+1)+w, two two-label affine heads.
    assert local==519*8+8*9+8+4*8+4


@pytest.mark.parametrize('drift',['extra','module','shape','go-list','go-duplicate','ec-invalid',
    'token-order','offset','embedding','payload-type','root-type','projection-bias'])
def test_drift_refused(drift):
    m=model()
    if drift=='extra':m.register_parameter('extra',nn.Parameter(torch.ones(1)))
    if drift=='module':m.go.extra=nn.Identity()
    if drift=='shape':m.ec.output=nn.Linear(9,2)
    if drift=='go-list':m.go.axis_ids=list(m.go.axis_ids)
    if drift=='go-duplicate':m.go.axis_ids=('GO:0000001',)*2
    if drift=='ec-invalid':m.ec.axis_ids=('bad','2.1.1.1')
    if drift=='token-order':
        from mora3.tokenization.shortest_path import ByteTokenizer
        m.payload.codec.tokenizer=ByteTokenizer((b'CD',b'AB'),controls=('pad','node','edge'))
    if drift=='offset':m.payload.offset+=1
    if drift=='embedding':m.payload.embedding=nn.Embedding(518,8)
    if drift=='payload-type':m.payload=nn.Identity()
    if drift=='root-type':m=nn.Linear(8,2)
    if drift=='projection-bias':m.payload.projection.bias=None
    with pytest.raises(ValueError):roles(m)
