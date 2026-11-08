"""Lossless five-tile derivative input, separate opt-in weak target access.

This candidate adapter grants no training rights or native head compatibility.
Channel names are unknown; indices refer only to source composite tile order.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import io
import numpy as np
import torch


def _array(spec):
    path=Path(spec['path'])
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('project-relative asset required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=spec['sha256']:
        raise ValueError('asset checksum mismatch')
    array=np.load(io.BytesIO(raw),allow_pickle=False)
    if list(array.shape)!=spec['shape'] or array.dtype!=np.uint8:
        raise ValueError('uint8 declared shape required')
    return array


@dataclass(frozen=True)
class AlgorithmicCrop:
    record_id: str
    pixels: bytes
    image_sha256: str
    channels: tuple = tuple(range(5))
    evidence: str = 'source_derived_image'

    @classmethod
    def from_record(cls, record):
        # Read image only: mask, composite, audit context and profiles never enter.
        spec=record['image']
        if spec['evidence_class']!='source_derived_image':
            raise ValueError('source-derived image evidence required')
        if [c['source_tile_index'] for c in spec['channels']]!=list(range(5)):
            raise ValueError('original five channel order required')
        a=_array(spec)
        if a.shape!=(5,160,160):
            raise ValueError('bounded five channel crop required')
        return cls(record['record_id'],a.tobytes(),spec['sha256'])

    def input_image(self, *, device='cpu'):
        return torch.from_numpy(np.frombuffer(self.pixels,dtype=np.uint8).reshape(5,160,160).copy()).to(device)

    def target_mask(self, record, *, allow_source_algorithmic=False, device='cpu'):
        if not allow_source_algorithmic:
            raise ValueError('source-algorithmic opt-in required')
        if record['record_id']!=self.record_id or record['image']['sha256']!=self.image_sha256:
            raise ValueError('target record/image binding mismatch')
        spec=record['mask']
        if spec['evidence_class']!='source_algorithmic_mask' or spec.get('independent_ground_truth') is not False:
            raise ValueError('explicit non-independent algorithmic evidence required')
        a=_array(spec)
        if a.shape!=(160,160) or not np.isin(a,[0,1]).all():
            raise ValueError('aligned binary source object required')
        return torch.from_numpy(a.copy()).to(device)

    def require_training(self):
        raise ValueError('candidate lacks rights/split/weak-label/head admission')
