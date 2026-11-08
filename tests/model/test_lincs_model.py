"""Real admitted source views; no optimizer updates."""
import copy
import json
import pytest
import torch
from test_flow_unified import build
from test_lincs_native import cases, project, api


def make(cases, device):
    from mora3.model.lincs_model import SharedLincsModel
    views=tuple(project(api(),c) for c in cases.values())
    axes={v.modality:c[7] for v,c in zip(views,cases.values())}
    torch.manual_seed(14801)
    model=SharedLincsModel(build(device).model.annotation,axes).to(device=device,dtype=torch.float64).eval()
    return model,views


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_native_predictions_reference_and_gradients(cases,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    model,views=make(cases,device)
    views=tuple(v for v in views if v.direction=='molecule_to_profile')
    z=model(views)
    assert [len(p) for p in z]==[len(v.axis) for v in views]
    loss=model.objective(z,views)
    terms=[]
    for p,v in zip(z,views):
        a=v.native_profile(); mask=torch.tensor(a.observed_mask,device=device)
        y=p.new_tensor([x for x in a.values if x is not None])
        terms.append(((p[mask]-y)**2).sum()/int(mask.sum()))
    ref=torch.stack(terms).mean()
    assert torch.equal(loss,ref)
    params=tuple(model.parameters())
    g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    h=torch.autograd.grad(ref,params,allow_unused=True)
    for a,b in zip(g,h):
        assert (a is None)==(b is None)
        if a is not None: torch.testing.assert_close(a,b,rtol=1e-13,atol=1e-13)
    assert model.output['L1000'].out_features==978
    assert model.shared.payload.embedding.weight in set(params)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_reverse_condition_and_input_only_encoding(cases,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    model,views=make(cases,device)
    reverse=tuple(v for v in views if v.direction=='profile_to_molecule')
    c=model.condition(reverse)
    assert c.shape==(len(reverse),4) and torch.isfinite(c).all()
    # Conditioning must never request targets or provenance, in either direction.
    from mora3.data.lincs_native import LincsNativeView
    old_t,old_p=LincsNativeView.targets,LincsNativeView.provenance
    try:
        def forbidden(self): raise AssertionError('hidden target read')
        LincsNativeView.targets=forbidden;LincsNativeView.provenance=forbidden
        torch.testing.assert_close(c,model.condition(reverse),rtol=0,atol=0)
        model.condition(tuple(v for v in views if v.direction=='molecule_to_profile'))
    finally:
        LincsNativeView.targets=old_t;LincsNativeView.provenance=old_p
    c.square().sum().backward()
    assert model.profile_input['CellPainting'].weight.grad.abs().sum()>0
    assert model.profile_mask['CellPainting'].weight.grad.abs().sum()>0
    with pytest.raises(ValueError,match='decoder'):model.generate(reverse)
    with pytest.raises(ValueError,match='direction'):model(reverse)
    state=copy.deepcopy(model.get_extra_state());state['axes']['L1000']=('bad',)
    with pytest.raises(ValueError):model.set_extra_state(state)
