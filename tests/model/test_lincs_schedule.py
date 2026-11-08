"""Source-bound endpoint scheduling; all optimizer updates forbidden here."""
import copy
import pytest
import torch
from test_lincs_native import cases
from test_lincs_model import make


def test_schedule_binding_and_cyclic_source_coverage(cases):
    from mora3.model.lincs_schedule import LincsSchedule
    model, views = make(cases, 'cpu')
    forward = tuple(v for v in views if v.direction == 'molecule_to_profile')
    s = LincsSchedule(model, forward, weights={'CellPainting': .25, 'L1000': 2.}, batch_size=1)
    assert [s.select(i)[0] for i in range(4)] == ['CellPainting','L1000'] * 2
    assert s.select(0)[1] != s.select(2)[1]
    assert s.select(0) == s.select(4)
    state = s.state_dict()
    s.validate_state(copy.deepcopy(state))
    changed = copy.deepcopy(state); changed['weights']['L1000'] = 1.
    with pytest.raises(ValueError): s.validate_state(changed)
    swapped = LincsSchedule(model, forward[::-1], weights={'CellPainting': .25,'L1000':2.},batch_size=1)
    with pytest.raises(ValueError): swapped.validate_state(state)
    with pytest.raises(ValueError): s.select(True)
    with pytest.raises(ValueError): s.select(-1)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_weighted_native_objective_gradient_reference(cases, device):
    from mora3.model.lincs_schedule import LincsSchedule
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    model, views = make(cases,device)
    forward = tuple(v for v in views if v.direction == 'molecule_to_profile')
    s = LincsSchedule(model,forward,weights={'CellPainting':.25,'L1000':2.},batch_size=2)
    for step in (0,1):
        modality, selected = s.select(step)
        loss, count = s.objective(step)
        assert count == len(selected)
        terms=[]
        for pred,v in zip(model(selected),selected):
            a=v.native_profile(); mask=torch.tensor(a.observed_mask,device=device)
            target=pred.new_tensor([x for x in a.values if x is not None])
            terms.append((pred[mask]-target).square().mean())
        ref=torch.stack(terms).mean()*s.weights[modality]
        torch.testing.assert_close(loss,ref,rtol=0,atol=0)
        params=tuple(model.parameters())
        left=torch.autograd.grad(loss,params,allow_unused=True)
        right=torch.autograd.grad(ref,params,allow_unused=True)
        for a,b in zip(left,right):
            assert (a is None)==(b is None)
            if a is not None: torch.testing.assert_close(a,b,rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('weights',[{}, {'CellPainting':1.,'L1000':0.},
    {'CellPainting':True,'L1000':1.},{'CellPainting':1.,'L1000':float('nan')}])
def test_invalid_endpoint_weights_fail_closed(cases,weights):
    from mora3.model.lincs_schedule import LincsSchedule
    model, views=make(cases,'cpu')
    forward=tuple(v for v in views if v.direction=='molecule_to_profile')
    with pytest.raises(ValueError): LincsSchedule(model,forward,weights=weights,batch_size=1)


def test_reverse_duplicate_and_nontrain_views_rejected(cases):
    from mora3.model.lincs_schedule import LincsSchedule
    from mora3.data.lincs_native import LincsNativeView, _binding, _canonical
    model,views=make(cases,'cpu'); kw=dict(weights={'CellPainting':1.,'L1000':1.},batch_size=1)
    with pytest.raises(ValueError): LincsSchedule(model,views,**kw)
    f=tuple(v for v in views if v.direction=='molecule_to_profile')
    with pytest.raises(ValueError): LincsSchedule(model,f+f[:1],**kw)
    p=f[0].provenance();p['split']='test'; pb=_canonical(p)
    v=f[0]; bad=LincsNativeView(v.direction,v.modality,v.axis,v.input_bytes,v.target_bytes,pb,_binding(v.input_bytes,v.target_bytes,pb))
    with pytest.raises(ValueError):LincsSchedule(model,(bad,)+f[1:],**kw)
