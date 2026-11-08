import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from mora3.data.algorithmic_crop import AlgorithmicCrop

ROOT=Path('outputs/data-intake/20261031-subagent/cooperative-segmentation-v1/curated-v3')

@pytest.mark.parametrize('index', range(8))
def test_real_crop_preserves_all_channels_and_excludes_target(index):
    record=json.loads((ROOT/'records.json').read_text())[index]
    crop=AlgorithmicCrop.from_record(record)
    expected=np.load(record['image']['path'],allow_pickle=False)
    assert torch.equal(crop.input_image(),torch.from_numpy(expected.copy()))
    changed=copy.deepcopy(record)
    changed['mask']={'path':'must-not-be-read', 'evidence_class':'predicted_mask'}
    changed['source_context_file']='must-not-be-read'
    changed['source_profile_ref']={'values':[999]}
    assert torch.equal(AlgorithmicCrop.from_record(changed).input_image(),crop.input_image())
    target=crop.target_mask(record,allow_source_algorithmic=True)
    assert set(target.unique().tolist()) <= {0,1}
    with pytest.raises(ValueError,match='opt-in'): crop.target_mask(record)
    with pytest.raises(ValueError,match='admission'): crop.require_training()
    if torch.cuda.is_available():
        assert torch.equal(crop.input_image(device='cuda').cpu(),crop.input_image())


def test_hash_channel_and_evidence_validation():
    r=json.loads((ROOT/'records.json').read_text())[0]
    bad=copy.deepcopy(r);bad['image']['sha256']='0'*64
    with pytest.raises(ValueError,match='checksum'): AlgorithmicCrop.from_record(bad)
    bad=copy.deepcopy(r);bad['image']['channels'].reverse()
    with pytest.raises(ValueError,match='channel'): AlgorithmicCrop.from_record(bad)
    crop=AlgorithmicCrop.from_record(r)
    bad=copy.deepcopy(r);bad['mask']['evidence_class']='manual_annotation'
    with pytest.raises(ValueError,match='algorithmic'):crop.target_mask(bad,allow_source_algorithmic=True)
    bad=copy.deepcopy(r);bad['record_id']='other'
    with pytest.raises(ValueError,match='record'):crop.target_mask(bad,allow_source_algorithmic=True)
