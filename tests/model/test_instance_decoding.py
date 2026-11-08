import pytest
import torch
from mora3.model.instance_decoding import decode_instances, match_instances


def test_touching_objects_decode_and_permuted_perfect_matching():
    labels=torch.tensor([[1,1,2],[0,0,2]])
    from mora3.model.instance_affinity import affinity_targets
    masks=labels[None]==torch.tensor([1,2])[:,None,None]
    affinity,_=affinity_targets(masks)
    logits=torch.cat(((labels>0)[None],affinity)).float()*20-10
    predicted=decode_instances(logits)
    assert torch.equal(predicted,labels)
    result=match_instances(predicted,masks.flip(0))
    assert result['tp']==2 and result['fp']==result['fn']==0
    assert result['matched_iou']==[1.,1.]


def test_splits_merges_empty_and_threshold_matching():
    truth=torch.tensor([[[1,1,1,1]]],dtype=torch.bool)
    split=match_instances(torch.tensor([[1,1,2,2]]),truth)
    assert (split['tp'],split['fp'],split['fn'],split['split_objects'])==(1,1,0,1)
    merge=match_instances(torch.ones((1,4),dtype=torch.long),torch.tensor([[[1,1,0,0]],[[0,0,1,1]]],dtype=torch.bool))
    assert merge['merge_objects']==1 and merge['fn']==1
    empty=match_instances(torch.zeros((1,4),dtype=torch.long),truth[:0])
    assert empty['tp']==empty['fp']==empty['fn']==0
    # Sparse arbitrary prediction IDs must not allocate max-ID-sized tables.
    assert match_instances(torch.tensor([[99999999,99999999,0,0]]),truth)['tp']==1
    with pytest.raises(ValueError): match_instances(torch.zeros((1,4),dtype=torch.long),truth,iou_threshold=0)
    with pytest.raises(ValueError): decode_instances(torch.full((3,2,2),float('nan')))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_decode_is_input_only_and_device_independent(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    z=torch.full((3,3,5),10.,device=device)
    z[0,1,:]=-10
    pred=decode_instances(z)
    assert pred.tolist()==[[1]*5,[0]*5,[2]*5]
    assert pred.device.type=='cpu'  # Explicit CPU discrete postprocessing.


def test_matching_reassigns_earlier_pair_instead_of_greedy_under_count():
    # IoUs [[1/3,1/2],[1/2,0]]; greedy first pair would lose a match.
    predicted=torch.tensor([[1,1,1,1,2,2]])
    truth=torch.tensor([[[1,1,0,0,1,1]],[[0,0,1,1,0,0]]],dtype=torch.bool)
    result=match_instances(predicted,truth,iou_threshold=.3)
    assert result['tp']==2
    assert result['matched_iou']==[.5,.5]
