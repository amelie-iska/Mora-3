import copy
from dataclasses import replace
import pytest
import torch
from test_molecule_profile_model import model,view
from mora3.model.profile_decoder import ProfileSelfiesDecoder,SelfiesCodec
from mora3.model import decoder_assessment as module

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_target_free_assessment_restores_state_and_matches_direct_bos(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    assert hasattr(module,'predict_molecules'), 'missing state-preserving target-free decoder inference'
    torch.manual_seed(11501)
    a=view('profile_to_molecule');data=a.inputs();data['source_profile_array']['values'][0]+=10
    import json
    b=replace(a,input_bytes=json.dumps(data).encode())
    m=ProfileSelfiesDecoder(model(device),SelfiesCodec(('CCO',)),max_tokens=12).to(device)
    m.train();m.output.eval()
    m.objective((a,b)).backward()
    state=copy.deepcopy(m.state_dict());grads=[None if p.grad is None else p.grad.clone() for p in m.parameters()]
    modes=[x.training for x in m.modules()];rng=torch.get_rng_state().clone()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    vs=tuple(replace(v,target_bytes=b'unreadable',provenance_bytes=b'unreadable') for v in (a,b))
    result=module.predict_molecules(m,vs)
    assert result['rotation_indices']==(1,0)
    assert modes==[x.training for x in m.modules()]
    m.eval()
    with torch.no_grad():
        prefix=torch.full((2,1),m.codec.bos,device=device,dtype=torch.long)
        expected=m(vs,prefix)[:,0].cpu()
        assert torch.equal(result['bos_logits'],expected)
        assert torch.equal(result['rotated_bos_logits'],expected.flip(0))
        assert result['generation']==m.generate(vs)
    for x,mode in zip(m.modules(),modes):x.training=mode
    assert modes==[x.training for x in m.modules()]
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all()))
    for k,v in state.items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,m.state_dict()[k])
        else:assert v==m.state_dict()[k]
    for p,g in zip(m.parameters(),grads):
        assert (p.grad is None)==(g is None)
        if g is not None:assert torch.equal(p.grad,g)
    assert not result['bos_logits'].requires_grad
    with pytest.raises(ValueError):module.predict_molecules(m,(view(),))
    assert modes==[x.training for x in m.modules()]


def test_rotation_is_within_modality_and_singletons_are_explicit():
    assert hasattr(module,'predict_molecules'), 'missing target-free decoder inference'
    a=view('profile_to_molecule');b=view('profile_to_molecule',modality='CellPainting')
    m=ProfileSelfiesDecoder(model(),SelfiesCodec(('CCO',)),max_tokens=4)
    r=module.predict_molecules(m,(a,b,a))
    assert r['rotation_indices']==(2,1,0)
    assert r['sensitivity']['eligible_condition_count']==2
    assert r['sensitivity']['singleton_condition_count']==1
    assert r['sensitivity']['mean_abs_bos_logit_change']==0
    assert r['sensitivity']['changed_token_sequence_count']==0
