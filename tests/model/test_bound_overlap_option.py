import copy
import numpy as np
import pytest
import torch
from test_bound_instance_inference import setup_case
from mora3.model.color_instance_inference import infer_nuclear_instances, instance_model_digest
from mora3.data.segmentation_shared import project_bound_slot_geometry

@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_exact_inference_and_graph_parity(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    m,image,pixels,kw=setup_case(device)
    with torch.no_grad():
        m.color.head.projection.weight.zero_();m.color.head.projection.bias.zero_()
        m.offset.weight.zero_();m.offset.bias.zero_()
        m.color.head.projection.bias[2]=-1
    kw['expected_model_sha256']=instance_model_digest(m)
    state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    ref=infer_nuclear_instances(m,image,pixels,**kw)
    exact=infer_nuclear_instances(m,image,pixels,overlap_method='bbox_exact',**kw)
    assert np.array_equal(ref['masks'],exact['masks'])
    assert {k:v for k,v in ref.items() if k!='masks'}=={k:v for k,v in exact.items() if k!='masks'}
    assert exact['measurements']['empty_slot_ids']==[3]
    assert exact['measurements']['edges'][0]['overlap_px']==63
    from mora3.model.color_semantic_inference import image_condition_graph
    prov=image_condition_graph(image,pixel_sha256=kw['expected_pixel_sha256'])[0].provenance
    args=dict(provenance=prov,requested_targets=('response',),max_mask_bytes=189)
    a=project_bound_slot_geometry(image,ref,**args)
    b=project_bound_slot_geometry(image,exact,overlap_method='bbox_exact',**args)
    for key in ('input_graph','task'): assert a[key].canonical_bytes()==b[key].canonical_bytes()
    assert {k:v for k,v in a.items() if k not in ('input_graph','task')}=={k:v for k,v in b.items() if k not in ('input_graph','task')}
    # Optimized replay cannot accept forged measurements.
    bad=copy.deepcopy(exact);bad['measurements']['edges'][0]['overlap_px']=62
    with pytest.raises(ValueError,match='geometry/provenance'):
        project_bound_slot_geometry(image,bad,overlap_method='bbox_exact',**args)
    assert torch.equal(rng,torch.get_rng_state())
    if cuda_rng is not None: assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
    for k,v in state.items():
        assert torch.equal(v,m.state_dict()[k]) if torch.is_tensor(v) else v==m.state_dict()[k]

@pytest.mark.parametrize('bad',[None,True,[],{},'bbox',''])
@pytest.mark.parametrize('entry',['inference','projection'])
def test_invalid_method_before_inputs(bad,entry):
    with pytest.raises(ValueError,match='overlap'):
        if entry=='inference':
            infer_nuclear_instances(None,None,None,expected_pixel_sha256='',expected_model_sha256='',expected_image_sha256='',mask_asset='',seed=1,max_mask_bytes=1,overlap_method=bad)
        else:
            project_bound_slot_geometry(None,None,provenance=(),requested_targets=(),max_mask_bytes=1,overlap_method=bad)
