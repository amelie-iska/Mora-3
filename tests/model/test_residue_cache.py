import copy
from dataclasses import replace
import pytest
import torch
from test_residue_cooperative import view
from test_flow_unified import build
from mora3.model.residue_objective import SharedResidueModel
from mora3.model.residue_preparation import prepare_residues


def test_cache_budget_lru_binding_and_failed_admission():
    from mora3.model.residue_preparation import ResiduePreparationCache
    base=build().model.annotation
    tok=base.payload.codec.tokenizer
    cache=ResiduePreparationCache(max_entries=2,max_residues=8)
    v=replace(view(),sequence='MAGK'); other=replace(v,sequence='MAAK'); third=replace(v,sequence='MAGG')
    def get(v,active=True):
        return cache.get((v,),tok,feature_axis=('Region',),applicable=(active,))
    a=get(v); b=get(other)
    assert get(v) is a
    get(third)
    assert get(other) is not b  # least recently used was evicted
    assert cache.entries==2 and cache.residues==8
    before=(cache.entries,cache.residues)
    with pytest.raises(ValueError,match='budget'):get(replace(v,sequence='M'*9))
    assert (cache.entries,cache.residues)==before
    inactive=get(v,False)
    assert inactive.binding!=a.binding and not inactive.positive_indices[0]
    with pytest.raises(ValueError):ResiduePreparationCache(max_entries=0,max_residues=8)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_opt_in_preparation_matches_every_model_gradient_and_rejects_stale(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(7701)
    shared=build(device).model.annotation
    ref=SharedResidueModel(shared,('Region',)); model=copy.deepcopy(ref)
    views=(view(),);mask=torch.tensor([True],device=device)
    p=prepare_residues(views,shared.payload.codec.tokenizer,feature_axis=('Region',),applicable=(True,),max_residues=8)
    expected=ref(views);actual=model(views,prepared=p,applicable=mask)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    a=model.objective(actual,views,applicable=mask,prepared=p)['total']
    b=ref.objective(expected,views,applicable=mask)['total']
    assert torch.equal(a,b)
    a.backward();b.backward()
    for (n,x),(m,y) in zip(model.named_parameters(),ref.named_parameters()):
        assert n==m
        assert (x.grad is None)==(y.grad is None)
        if x.grad is not None:assert torch.equal(x.grad,y.grad),n
    for changed,active in [((replace(views[0],sequence='MAAK'),),mask),(views,~mask)]:
        with pytest.raises(ValueError,match='binding'):model(changed,prepared=p,applicable=active)
        with pytest.raises(ValueError,match='binding'):model.objective(actual,changed,prepared=p,applicable=active)
