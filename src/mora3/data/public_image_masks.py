"""Source-bound public raster supervision; format verification is not admission.

Preserves complete source pixels, ordered overlapping instances and semantic
supervision separately. No resizing, color conversion, joins or training rights
are inferred. The native input carries only the visible image, never masks.
"""
from dataclasses import dataclass
import hashlib
import io
import json

import numpy as np
from PIL import Image

from .cooperative_segmentation import SegmentationArtifact
from .segmentation_image import SegmentationImageInput


def _decode(raw, *, max_pixels, allow_binary=False):
    if type(raw) is not bytes or len(raw)>64*1024**2:
        raise ValueError('bounded immutable encoded bytes required')
    with Image.open(io.BytesIO(raw)) as im:
        if im.width*im.height>max_pixels or getattr(im,'n_frames',1)!=1:
            raise ValueError('bounded single-plane source required')
        a=np.array(im)
    if (a.dtype.kind!='u' and not (allow_binary and a.dtype.kind=='b')) or a.dtype.itemsize>2:
        raise ValueError('unsigned source pixels required')
    return a


@dataclass(frozen=True)
class PublicImageMasks:
    image: SegmentationArtifact
    image_encoded: bytes
    masks_encoded: tuple
    encoding: str
    split: str
    image_sha256: str
    mask_sha256: tuple

    def __post_init__(self):
        a=self.image
        if not isinstance(a,SegmentationArtifact) or a.kind!='image' or a.evidence!='source_observed' or a.role!='condition':
            raise ValueError('visible source-observed image required')
        if a.representation!='source-yx-no-transform-v1' or a.reference_frame!='source_pixel_yx':
            raise ValueError('untransformed source pixel frame required')
        if self.split not in ('train','validation','test','evaluation'):
            raise ValueError('explicit source split required')
        if self.encoding not in ('semantic_binary','instance_id_map','ordered_binary_instances'):
            raise ValueError('explicit supported mask encoding required')
        object.__setattr__(self,'masks_encoded',tuple(self.masks_encoded))
        object.__setattr__(self,'mask_sha256',tuple(self.mask_sha256))
        if len(self.masks_encoded)!=len(self.mask_sha256) or len(self.masks_encoded)>65535:
            raise ValueError('ordered mask hash alignment required')
        if self.encoding!='ordered_binary_instances' and len(self.masks_encoded)!=1:
            raise ValueError('single semantic or instance-ID map required')
        if hashlib.sha256(self.image_encoded).hexdigest()!=self.image_sha256:
            raise ValueError('source image checksum mismatch')
        pixels=self.image_array()
        if pixels.shape[:2]!=a.spatial_shape or pixels.ndim not in (2,3):
            raise ValueError('source image alignment mismatch')
        channels=1 if pixels.ndim==2 else pixels.shape[2]
        if channels!=len(a.channels): raise ValueError('source channel alignment mismatch')
        for raw,digest in zip(self.masks_encoded,self.mask_sha256):
            if hashlib.sha256(raw).hexdigest()!=digest:
                raise ValueError('source mask checksum mismatch')
            mask=_decode(raw,max_pixels=16_000_000,allow_binary=True)
            if mask.shape!=a.spatial_shape:
                raise ValueError('source mask alignment mismatch')
            if self.encoding!='instance_id_map' and not np.isin(mask,[0,1,255]).all():
                raise ValueError('binary supervision required')
            if self.encoding=='ordered_binary_instances' and not mask.any():
                raise ValueError('empty source instance unsupported')

    @classmethod
    def from_encoded(cls, image, image_bytes, masks, **kwargs):
        return cls(image,image_bytes,tuple(masks),**kwargs)

    def image_array(self):
        return _decode(self.image_encoded,max_pixels=16_000_000)

    def target_arrays(self):
        return tuple(self.iter_target_arrays())

    def iter_target_arrays(self):
        """Decode one original mask at a time; preserve source list order."""
        for raw in self.masks_encoded:
            mask=_decode(raw,max_pixels=16_000_000,allow_binary=True)
            yield mask if self.encoding=='instance_id_map' else mask!=0

    def native_input(self):
        if self.image.channels!=('brightfield',):
            raise ValueError('native channel signature unsupported; no color conversion')
        a=self.image_array()
        return SegmentationImageInput(self.image,a.dtype.str,a.tobytes())

    @property
    def binding_sha256(self):
        from dataclasses import asdict
        payload=dict(schema='mora3.public_image_masks.v1',image=asdict(self.image),
            image_sha256=self.image_sha256,mask_sha256=self.mask_sha256,
            encoding=self.encoding,split=self.split)
        return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    def require_training(self):
        raise ValueError('format verification is not corpus-owned training admission')
