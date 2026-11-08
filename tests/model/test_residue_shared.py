from dataclasses import replace
import pytest
import torch
from test_annotation_multihead import model
from test_annotation_vocabulary import admitted
from mora3.model import residue_objective as module

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_backbone_equation_target_exclusion_and_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    assert hasattr(module,'SharedResidueModel'), 'native shared residue model missing'
    torch.manual_seed(6901)
    shared=model().to(device)
    m=module.SharedResidueModel(shared,('Region',))
    v=admitted()
    hidden=replace(v,target_bytes=b'{"annotation_packet":{"features":[{"type":"Region","location":{"start":{"value":1,"modifier":"EXACT"},"end":{"value":2,"modifier":"EXACT"}}}]}}')
    z=m([hidden])[0]
    assert torch.equal(z,m([v])[0])
    record,task=module.annotation_graph_task(v)
    graph=shared.payload.forward_task(record,task=task,identifiers=shared.sequence_identifiers())
    context=shared.backbone([graph])[0][0,0]
    tok=shared.payload.codec.tokenizer
    ids=torch.tensor(tok.encode(v.sequence.encode()),device=device)
    x=shared.payload.embedding(ids)
    expected=m.head(module.align_token_residues(v.sequence,tok,x),context)
    torch.testing.assert_close(z,expected)
    loss=m.objective([z],[hidden],applicable=torch.tensor([True],device=device))['total']
    loss.backward()
    assert shared.backbone.feature.weight.grad.abs().sum()>0
    assert shared.payload.embedding.weight.grad.abs().sum()>0
    assert m.head.output.weight.grad.abs().sum()>0
    assert shared.go.output.weight.grad is None and shared.ec.output.weight.grad is None
    assert len(list(m.parameters()))==len({id(p) for p in m.parameters()})
    clone=module.SharedResidueModel(model().to(device),('Region',))
    clone.load_state_dict(m.state_dict())
    assert torch.equal(clone([v])[0],z)
    inactive=m.objective(m([hidden]),[hidden],applicable=torch.tensor([False],device=device))['total']
    assert inactive.item()==0

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_trainer_bound_restore_and_learning(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from mora3.model import residue_trainer as r
    torch.manual_seed(6902)
    v=replace(admitted(),target_bytes=b'{"annotation_packet":{"features":[{"type":"Region","location":{"start":{"value":1,"modifier":"EXACT"},"end":{"value":2,"modifier":"EXACT"}}}]}}')
    batch=r.ResidueTrainingBatch((v,),('Region',),(True,))
    m=module.SharedResidueModel(model().to(device),('Region',))
    trainer=r.ResidueTrainer(m,batch)
    first=trainer.step()['total']; state=trainer.state_dict()
    expected=trainer.step()
    trainer.load_state_dict(state)
    assert trainer.step()==expected
    assert expected['total']<first
    bad=r.ResidueTrainingBatch((v,),('Region',),(False,))
    other=r.ResidueTrainer(m,bad)
    before=[p.clone() for p in m.parameters()]
    with pytest.raises(ValueError,match='binding'):other.load_state_dict(state)
    assert all(torch.equal(p,q) for p,q in zip(m.parameters(),before))
