"""Streamed mode unit metadata never authorizes caller-created source batches."""
import copy
import hashlib
import json

import pytest
import torch
from torch.nn import functional as F


def stub_pool():
    # Explicit synthetic metadata-only fixture: no source_view, canonical index,
    # original rows or authorization. A source objective must therefore refuse.
    from mora3.data.goa_streamed import StreamedGOASourcePool,_code_hashes
    pool=object.__new__(StreamedGOASourcePool)
    pool.axis_ids=('GO:0000001','GO:0000002');pool.source_paths=('synthetic-only',)
    canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    sha=lambda x:hashlib.sha256(canonical(x)).hexdigest()
    pool._binding=dict(schema='Mora3.StreamedCanonicalGOASourcePool/v1',
        axis_sha256=sha(pool.axis_ids),source_paths_sha256=sha(pool.source_paths),
        source_training_admission=False,fixture='synthetic-metadata-only-no-source-batches')
    pool._binding['source_code']=_code_hashes()
    pool._binding_sha256=sha(pool._binding)
    return pool


def head(pool=None):
    from mora3.model.uniprot_go import NativeGOAHead
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261106)
        return NativeGOAHead(8,stub_pool()if pool is None else pool,
            parameterization_policy='mup-goa-v1',base_width=4,output_mult=1.5).double()


def test_streamed_head_has_full_axis_no_packet_pool_and_noncancelling_readout():
    model=head();x=torch.arange(24,dtype=torch.float64).reshape(3,8).requires_grad_()
    actual=model(x);expected=F.linear(x*.75,model.output.weight,model.output.bias)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    params=(x,*model.parameters())
    a=torch.autograd.grad(actual.square().sum(),params,retain_graph=True)
    b=torch.autograd.grad(expected.square().sum(),params)
    for left,right in zip(a,b):torch.testing.assert_close(left,right,rtol=0,atol=0)
    assert model.get_extra_state()['version']==3 and model._packet_hashes==frozenset()
    assert 'packets'not in model.get_extra_state()
    assert {name for name,_ in model.named_modules()}=={'','output'}


def test_streamed_mode_checkpoint_copy_and_source_binding_refusal_before_readout_copy():
    model=head();state=copy.deepcopy(model.state_dict());other=copy.deepcopy(model)
    other.load_state_dict(state);assert other.get_extra_state()==model.get_extra_state()
    before=model.output.weight.detach().clone()
    state['output.weight']+=1
    state['_extra_state']['streamed_pool']['axis_sha256']='b'*64
    with pytest.raises(ValueError):model.load_state_dict(state)
    assert torch.equal(model.output.weight,before)


@pytest.mark.parametrize('change',['axis','path','metadata'])
def test_mutable_source_pool_metadata_cannot_change_bound_constructor(change):
    pool=stub_pool()
    if change=='axis':pool.axis_ids=pool.axis_ids[::-1]
    elif change=='path':pool.source_paths=('other',)
    else:pool._binding['source_training_admission']=True
    with pytest.raises(ValueError):head(pool)


def test_streamed_mode_refuses_old_legacy_parameterization_before_rng():
    from mora3.model.uniprot_go import NativeGOAHead
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):NativeGOAHead(8,stub_pool())
    assert torch.equal(rng,torch.get_rng_state())


def test_metadata_only_pool_and_forged_source_batch_cannot_supply_source_objective():
    from mora3.data.goa_streamed import StreamedGOABatch
    model=head();pool=stub_pool()
    with pytest.raises(ValueError):model.objective(torch.zeros(1,2),(object(),))
    with pytest.raises(ValueError):model.objective(torch.zeros(1,2),StreamedGOABatch(None,(),pool))


def test_streamed_cooperative_constructor_keeps_source_pool_out_of_registered_model(tmp_path,request):
    from dataclasses import replace
    assets=request.getfixturevalue('actual_core_assets');pool=stub_pool()
    # _construct intentionally adds an old GOA fixture, so call factory directly.
    from mora3.model.full_owner import construct_primary_owner,PrimaryConfiguration
    model=construct_primary_owner(PrimaryConfiguration(8,8,'a'*64),replace(assets,goa_packets=pool))
    anchor=model.core.parent.parent
    assert anchor.goa.get_extra_state()['version']==3
    assert all(not isinstance(value,type(pool)) for module in model.modules()for value in vars(module).values())
    assert len(anchor.goa.axis_ids)==2 and not anchor.goa._packet_hashes
    clone=copy.deepcopy(model);clone.load_state_dict(copy.deepcopy(model.state_dict()))
    assert clone.get_extra_state()==model.get_extra_state()


from test_full_owner import actual_core_assets,cases
