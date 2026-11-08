import pytest
import torch

from mora3.model.uniprot_full_record_model import SharedUniProtFullRecordModel
from test_uniprot_full_record_model import axes,pubchem,row,native


def test_alternative_first_order_path_preserves_complete_native_field_and_gradients():
    m=SharedUniProtFullRecordModel(pubchem(16),axes()).double().eval()
    packets=(row(native(False)),row())
    p=tuple(m.parameters())
    a=m._reference_field_objective(packets,'comments',chunk_size=7,recompute=True)
    b=m._reference_field_objective(packets,'comments',chunk_size=7,recompute=True,
                                  backward_policy='bounded-chunk-adjoint-v1')
    assert all(b[k]==a[k]for k in a.keys()-{'loss'})
    assert torch.equal(a['loss'],b['loss'])
    g=torch.autograd.grad(a['loss'],p,allow_unused=True)
    h=torch.autograd.grad(b['loss'],p,allow_unused=True)
    assert all((x is y is None) or torch.allclose(x,y,rtol=1e-12,atol=1e-12)
               for x,y in zip(g,h,strict=True))
    with pytest.raises(ValueError):
        m._reference_field_objective(packets,'comments',chunk_size=7,recompute=False,
                                     backward_policy='bounded-chunk-adjoint-v1')
