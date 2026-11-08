from dataclasses import replace
import copy
import pytest
import torch
from test_molecule_profile_model import model, view
from mora3.model.molecule_profile_training import MoleculeProfileBatch, MoleculeProfileTrainer


def batch():
    return MoleculeProfileBatch((view(), view(modality='CellPainting')))


def test_binding_covers_actual_bytes_order_direction_and_source_gate():
    b=batch(); v=b.views[0]
    for field, value in [('target_bytes',v.target_bytes.replace(b'0.25',b'1.25')),
                         ('input_bytes',v.input_bytes.replace(b'CCO',b'CCN'))]:
        assert MoleculeProfileBatch((replace(v,**{field:value}),b.views[1])).binding()!=b.binding()
    assert MoleculeProfileBatch(b.views[::-1]).binding()!=b.binding()
    with pytest.raises(ValueError,match='prediction'):MoleculeProfileBatch((view('profile_to_molecule'),))
    with pytest.raises(ValueError,match='admission'):MoleculeProfileBatch((replace(v,provenance_bytes=b'{}'),))
    with pytest.raises(ValueError,match='nonempty'):MoleculeProfileBatch(())


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_native_objective_and_full_state_continuation(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(8101)
    m=model(device); b=batch(); t=MoleculeProfileTrainer(m,b)
    ref=copy.deepcopy(m);opt=torch.optim.Adam(ref.parameters(),lr=.001)
    loss=ref.objective(ref(b.views),b.views);loss.backward();opt.step()
    assert t.step()['total']==float(loss.detach())
    for p,q in zip(m.parameters(),ref.parameters()):assert torch.equal(p,q)
    state=t.state_dict(); expected=t.step(); end=t.state_dict()
    restored=MoleculeProfileTrainer(model(device),b);restored.load_state_dict(state)
    assert restored.step()==expected
    for k,v in end['model'].items():
        other=restored.state_dict()['model'][k]
        if torch.is_tensor(v):assert torch.equal(v,other),k
        else:assert v==other
    changed=MoleculeProfileBatch((replace(b.views[0],target_bytes=b.views[0].target_bytes.replace(b'0.25',b'1.25')),b.views[1]))
    with pytest.raises(ValueError,match='binding'):MoleculeProfileTrainer(model(device),changed).load_state_dict(state)


@pytest.mark.parametrize('lr',[0.,float('nan'),float('inf')])
def test_rejects_nonpositive_or_nonfinite_learning_rate(lr):
    with pytest.raises(ValueError,match='learning rate'):
        MoleculeProfileTrainer(model(),batch(),lr=lr)
