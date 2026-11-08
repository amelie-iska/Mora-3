"""Exact ordered categorical identity; missing context is never a numeric default."""
from dataclasses import replace
import pytest
import torch
from mora3.data.native_atom_context import native_atom_context
import mora3.data.native_atom_context as api


def context(numbers=(8,6,1,118)):
    return native_atom_context(dict(source_record_id='r',source_reaction_group='g',
        source_native_split='val',atomic_numbers=list(numbers),electronic_charge=None,
        spin_multiplicity=None,electronic_charge_known=False,
        spin_multiplicity_known=False,task_admitted=False),
        ordered_atom_ids=tuple(str(i) for i in range(len(numbers))),source_payload_sha256='a'*64)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_exact_complete_axis_and_unknown_masks(device,dtype):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    c=context(tuple(range(1,119)))
    encoded=api.encode_native_atom_identity(c,device=device,dtype=dtype)
    assert encoded.context==c
    assert torch.equal(encoded.identity,torch.eye(118,device=device,dtype=dtype))
    assert encoded.context_fields==c.missing_context
    assert encoded.context_missing.dtype==torch.bool
    assert torch.equal(encoded.context_missing,torch.ones(118,5,device=device,dtype=torch.bool))
    assert not encoded.identity.requires_grad
    with pytest.raises(ValueError,match='admission'): encoded.require_model_use()


def test_order_and_fresh_storage():
    c=context();a=api.encode_native_atom_identity(c);b=api.encode_native_atom_identity(c)
    assert a.identity.argmax(-1).tolist()==[7,5,0,117]
    a.identity.zero_();a.context_missing.zero_()
    assert b.identity.sum().item()==4 and b.context_missing.all()
    assert c.atomic_numbers==(8,6,1,118)


@pytest.mark.parametrize('change',[
 {'version':'future'}, {'atomic_numbers':(0,6,1,118)},
 {'ordered_atom_ids':('x','x','z','w')}, {'missing_context':()},
 {'electronic_charge':0}, {'spin_multiplicity':1}, {'source_payload_sha256':'bad'}])
def test_forged_metadata_refused(change):
    with pytest.raises(ValueError):api.encode_native_atom_identity(replace(context(),**change))


@pytest.mark.parametrize('dtype',[torch.int64,torch.bool,torch.float16])
def test_unqualified_dtype_refused(dtype):
    with pytest.raises(ValueError):api.encode_native_atom_identity(context(),dtype=dtype)


def test_noncontract_refused():
    with pytest.raises(ValueError):api.encode_native_atom_identity({'atomic_numbers':[1]})
