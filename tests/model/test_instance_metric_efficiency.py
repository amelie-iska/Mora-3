"""Dense oracle catches changed overlap counts, float rounding and matching ties."""
import pytest
import torch
from mora3.model.instance_decoding import match_instances

@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_contingency_matches_dense_with_overlap_sparse_ids_and_ties(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    generator=torch.Generator().manual_seed(193)
    for n in (0,1,7):
        for threshold in (.2,.5,1.):
            p=torch.randint(0,5,(9,13),generator=generator)*999999999
            t=torch.rand((n,9,13),generator=generator)>.72
            p=p.to(device); t=t.to(device)
            assert match_instances(p,t,iou_threshold=threshold,backend='contingency') == match_instances(p,t,iou_threshold=threshold,backend='dense')
    p=torch.tensor([[1,1,1,1,2,2]],device=device)
    t=torch.tensor([[[1,1,0,0,1,1]],[[0,0,1,1,0,0]]],dtype=torch.bool,device=device)
    assert match_instances(p,t,iou_threshold=.3,backend='contingency')==match_instances(p,t,iou_threshold=.3,backend='dense')
    assert match_instances(p,t,iou_threshold=.3)['matched_iou']==[.5,.5]

def test_contingency_validation_empty_and_backend_guard():
    p=torch.zeros((2,3),dtype=torch.long)
    t=torch.zeros((0,2,3),dtype=torch.bool)
    assert match_instances(p,t,backend='contingency')==match_instances(p,t,backend='dense')
    for backend in ('dense','contingency'):
        with pytest.raises(ValueError,match='Empty truth'): match_instances(p,torch.zeros((1,2,3),dtype=torch.bool),backend=backend)
    with pytest.raises(ValueError,match='backend'): match_instances(p,t,backend='approximate')
