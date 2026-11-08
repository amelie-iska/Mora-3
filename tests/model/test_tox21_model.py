import copy
import json
from dataclasses import replace
import pytest
import torch
from test_molecule_profile_model import model
from test_native_tox21 import fixture, view
from mora3.data.tox21 import INVERSE, tox21_call_tensors

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_masked_equation_gradients_and_target_exclusion(device):
    from mora3.model.tox21 import SharedTox21Model
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA absent')
    torch.manual_seed(8701)
    m=SharedTox21Model(model(device).shared).double();v=view(fixture()[0]);vs=(v,)
    z=m(vs);y,mask=tox21_call_tensors(vs,role='target',device=device)
    ref=copy.deepcopy(m);q=ref(vs)
    expected=(torch.nn.functional.softplus(q)-q*y)[mask].mean()
    loss=m.objective(z,vs);torch.testing.assert_close(loss,expected)
    loss.backward();expected.backward()
    for (n,p),(k,r) in zip(m.named_parameters(),ref.named_parameters()):
        assert n==k and (p.grad is None)==(r.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,r.grad,atol=1e-7,rtol=1e-5)
    assert m.shared.payload.embedding.weight.grad.abs().sum()>0
    assert m.backbone.feature.weight.grad.abs().sum()>0
    assert m.shared.go.output.weight.grad is None
    altered=replace(v,target_bytes=b'{}',provenance_bytes=b'{}',values=(1,)*12)
    assert torch.equal(m(vs),m((altered,)))
    logits=torch.zeros(1,12,device=device,requires_grad=True)
    m.objective(logits,vs).backward()
    assert logits.grad[0,:3].tolist()==[.0625,-.0625,0.]

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_desired_condition_missingness_and_inverse_target_exclusion(device):
    from mora3.model.tox21 import SharedTox21Model
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA absent')
    m=SharedTox21Model(model(device).shared);v=view(fixture(INVERSE)[0]);c=m.condition((v,))
    assert torch.equal(c,m.condition((replace(v,target_bytes=b'{}',provenance_bytes=b'{}'),)))
    x=v.inputs();x['desired_source_call_values'][0]=1
    changed=replace(v,input_bytes=json.dumps(x).encode())
    assert not torch.equal(c,m.condition((changed,)))
    c.square().sum().backward();assert m.call_input.weight.grad.abs().sum()>0
    assert m.backbone.feature.weight.grad.abs().sum()>0
    with pytest.raises(ValueError,match='direction'):m((v,))
    with pytest.raises(ValueError,match='decoder'):m.generate((v,))
    x['desired_source_call_values'][2]=0
    with pytest.raises(ValueError):m.condition((replace(v,input_bytes=json.dumps(x).encode()),))

def test_contract_rejections():
    from mora3.model.tox21 import SharedTox21Model
    m=SharedTox21Model(model().shared);v=view(fixture()[0])
    for vs in ((),(replace(v,axis=v.axis[::-1]),)):
        with pytest.raises(ValueError):m(vs)
    x=v.inputs();x['source_id']='hidden'
    with pytest.raises(ValueError):m((replace(v,input_bytes=json.dumps(x).encode()),))
    with pytest.raises(ValueError):m.objective(torch.zeros(1,11),(v,))
    state=copy.deepcopy(m.get_extra_state());state['axis']=()
    with pytest.raises(ValueError):m.set_extra_state(state)

def test_objective_rejects_divergent_target_bytes():
    from mora3.model.tox21 import SharedTox21Model
    m=SharedTox21Model(model().shared);v=view(fixture()[0])
    t=v.targets();t['source_call_values'][0]=1
    with pytest.raises(ValueError,match='target'):
        m.objective(torch.zeros(1,12),(replace(v,target_bytes=json.dumps(t).encode()),))
