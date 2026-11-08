"""Lossless LIVECell 2D polygon supervision; no rasterization approximation.

Polygons are whole-cell manual labels in pixel coordinates, not nuclei or
semantic cell identities. No cross-source pairing or physical scale is inferred.
This source adapter shares condition/target roles with scalar/array tasks but
is not yet a neural image encoder or a biological cell-graph builder.
"""
from dataclasses import dataclass, asdict
import json
import math
import re
import numpy as np
import torch


@dataclass(frozen=True)
class LiveCellExample:
    pixels_hex: str
    dtype: str
    height: int
    width: int
    image_id: int
    file_name: str
    polygons: tuple
    split: str
    source_sha256: str
    version: int = 1

    def __post_init__(self):
        if self.version != 1 or self.split not in ('train','val','test'):
            raise ValueError('unsupported version or split')
        if not re.fullmatch('[0-9a-f]{64}',self.source_sha256):
            raise ValueError('source SHA256 required')
        if self.dtype not in ('|u1','<u2') or any(type(v) is not int or v<=0 for v in (self.height,self.width)):
            raise ValueError('2D uint8/uint16 image required')
        if type(self.image_id) is not int or self.image_id<0:
            raise ValueError('image identity required')
        if not re.fullmatch(r'[A-Za-z0-9-]+_Phase_[A-Z][0-9]+_[0-9]+_[0-9]+d[0-9]+h[0-9]+m_[0-9]+\.tif', self.file_name):
            raise ValueError('unsupported LIVECell filename context')
        try: raw=bytes.fromhex(self.pixels_hex)
        except (ValueError,TypeError) as exc: raise ValueError('invalid pixels') from exc
        if len(raw)!=self.height*self.width*np.dtype(self.dtype).itemsize:
            raise ValueError('image dimensions do not match bytes')
        objects=[]
        for segments in self.polygons:
            if not segments: raise ValueError('empty instance')
            parts=[]
            for coords in segments:
                if len(coords)<6 or len(coords)%2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in coords):
                    raise ValueError('finite polygon coordinates required')
                if any(v<0 or v>(self.width if i%2==0 else self.height) for i,v in enumerate(coords)):
                    raise ValueError('polygon outside image')
                parts.append(tuple(float(v) for v in coords))
            objects.append(tuple(sorted(parts)))
        object.__setattr__(self,'polygons',tuple(sorted(objects)))

    @classmethod
    def from_coco(cls, *, image, image_info, annotations, split, source_sha256):
        a=np.asarray(image)
        if a.ndim!=2 or a.shape!=(image_info['height'],image_info['width']):
            raise ValueError('image alignment mismatch')
        if a.dtype not in (np.dtype('uint8'),np.dtype('uint16')):
            raise ValueError('unsigned integer source pixels required')
        seen=set(); polygons=[]
        for ann in annotations:
            if ann['image_id']!=image_info['id'] or ann['id'] in seen:
                raise ValueError('annotation identity/alignment mismatch')
            seen.add(ann['id'])
            if ann.get('iscrowd',0)!=0 or not isinstance(ann['segmentation'],list):
                raise ValueError('crowd/RLE not supported')
            polygons.append(ann['segmentation'])
        a=a.astype(a.dtype.newbyteorder('<'),copy=False)
        return cls(a.tobytes().hex(),a.dtype.str,a.shape[0],a.shape[1],image_info['id'],
                   image_info['file_name'],tuple(polygons),split,source_sha256)

    @property
    def context(self):
        cell,_,well,location,time,crop=self.file_name[:-4].split('_')
        return dict(cell_line=cell,well=well,location=location,elapsed_time=time,
                    crop=crop,parent_image=self.file_name.rsplit('_',1)[0],
                    leakage_group=f'LIVECell2021:{cell}:{well}',
                    microscopy='phase contrast',pixel_size_um=None,
                    perturbation=None,expression=None,organism=None)

    def input_metadata(self,direction):
        if direction not in ('image_to_mask','mask_to_image'):
            raise ValueError('unsupported task direction')
        return dict(version=1,direction=direction,height=self.height,width=self.width,
                    units='pixel',context=self.context,
                    roles={'image':'condition' if direction=='image_to_mask' else 'target',
                           'polygons':'target' if direction=='image_to_mask' else 'condition'})

    def project(self,direction,*,device='cpu'):
        self.input_metadata(direction)
        image=torch.from_numpy(np.frombuffer(bytes.fromhex(self.pixels_hex),dtype=self.dtype).copy().reshape(self.height,self.width)).to(device)
        # Ragged per-instance/per-component contours preserve every source vertex.
        polygons=tuple(tuple(torch.tensor(p,dtype=torch.float64,device=device).reshape(-1,2)
                             for p in obj) for obj in self.polygons)
        return dict(inputs={'image':image} if direction=='image_to_mask' else {'polygons':polygons},
                    targets={'polygons':polygons} if direction=='image_to_mask' else {'image':image})

    def require_training(self):
        if self.split!='train': raise ValueError('official split is held out')

    def canonical_bytes(self):
        return json.dumps(asdict(self),sort_keys=True,separators=(',',':'),allow_nan=False).encode()

    @classmethod
    def from_bytes(cls,raw):
        def unique(pairs):
            d={}
            for k,v in pairs:
                if k in d: raise ValueError('duplicate JSON key')
                d[k]=v
            return d
        return cls(**json.loads(raw,object_pairs_hook=unique))
