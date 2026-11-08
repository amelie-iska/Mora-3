"""Bounded image-only native inference input; no mask or training admission.

Pixels are caller supplied, never fetched. Artifact provenance is audit metadata;
only pixels and a single known image node reach the current native head.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import torch
from .cooperative_segmentation import SegmentationArtifact
from .schema import GraphRecord, Node
from .provenance import Provenance


@dataclass(frozen=True)
class SegmentationImageInput:
    artifact: SegmentationArtifact
    image_dtype: str
    pixels: bytes
    direction: str = 'image_to_mask'

    def __post_init__(self):
        a=self.artifact
        if not isinstance(a,SegmentationArtifact) or a.kind!='image':
            raise ValueError('validated image artifact required')
        if a.role not in ('condition','desired') or self.direction!='image_to_mask':
            raise ValueError('visible image_to_mask input required')
        if a.channels!=('brightfield',) or a.representation!='source-yx-no-transform-v1':
            raise ValueError('unsupported pixel representation/channel signature')
        dt=np.dtype(self.image_dtype)
        if dt.kind!='u' or dt.itemsize>2 or max(a.spatial_shape)>512:
            raise ValueError('bounded unsigned image required')
        if type(self.pixels) is not bytes or len(self.pixels)!=self.height*self.width*dt.itemsize:
            raise ValueError('aligned immutable pixel bytes required')

    @property
    def height(self): return self.artifact.spatial_shape[0]

    @property
    def width(self): return self.artifact.spatial_shape[1]

    def input_image(self, *, device='cpu'):
        a=np.frombuffer(self.pixels,dtype=self.image_dtype).reshape(self.height,self.width)
        return torch.from_numpy(a.astype(a.dtype.newbyteorder('='),copy=True)).to(device)

    def input_graph(self):
        # No artifact IDs, origin labels, hidden targets or target topology encoded.
        a=self.artifact
        provenance=Provenance(a.source_id,a.checkpoint or 'source-artifact-v1',a.asset,
                              hashlib.sha256(self.pixels).hexdigest(),'unknown:no-training-admission')
        return GraphRecord('native-image-input',(Node('microscopy_image',()),),(),(provenance,),())

    def require_training(self):
        raise ValueError('image-only inference input has no training admission')
