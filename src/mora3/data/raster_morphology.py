"""Source-neutral, full-resolution integer-instance morphology tasks.

Image/mask directions share observed acquisition context, never target topology.
Source IDs, hashes, arbitrary label IDs and split are audit-only. No biological
response pairing or spatial contacts are inferred. Train guard permits only
source train diagnostics; broader eligibility needs external leakage audits.
"""
from dataclasses import asdict, dataclass, fields
import json
import numpy as np
import torch
from .deepbacs import DeepBacsImage
from .schema import GraphRecord, Node, Edge, Measurement
from .provenance import Provenance, LeakageKey


def _canonical(d):
    return json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _m(k,v,u=None):
    return Measurement(k,v,u,'missing' if v is None else 'observed',v is not None)


@dataclass(frozen=True)
class RasterMorphologyTask:
    height: int
    width: int
    image_dtype: str
    label_dtype: str
    image_hex: str
    labels_hex: str
    source: str
    revision: str
    source_id: str
    archive_sha256: str
    image_sha256: str
    mask_sha256: str
    license: str
    parent_group: str
    split: str
    species: str
    strain: str
    modality: str
    annotation: str
    pixel_size_um: float
    direction: str = 'image_to_mask'
    version: int = 1

    def __post_init__(self):
        if type(self.version) is not int or self.version!=1:
            raise ValueError('unsupported morphology version')
        if self.direction not in ('image_to_mask','mask_to_image'):
            raise ValueError('unsupported direction')
        if self.split not in ('train','test','validation'):
            raise ValueError('unknown source split')
        if any(type(n) is not int or not 1<=n<=512 for n in (self.height,self.width)):
            raise ValueError('bounded image shape required')
        for raw,dtype in ((self.image_hex,self.image_dtype),(self.labels_hex,self.label_dtype)):
            dt=np.dtype(dtype)
            if dt.kind!='u' or dt.itemsize>2 or len(bytes.fromhex(raw))!=self.height*self.width*dt.itemsize:
                raise ValueError('aligned unsigned image/labels required')
        self._provenance()  # Validate source hash formats and textual context.
        LeakageKey('raster:parent-v1',self.parent_group)
        for k in ('species','strain','modality','annotation'):
            _m(k,getattr(self,k))
        if self.pixel_size_um is not None and self.pixel_size_um<=0:
            raise ValueError('positive observed pixel scale required')
        _m('pixel_size_um',self.pixel_size_um,'um/pixel')

    @classmethod
    def from_deepbacs(cls,x,direction='image_to_mask'):
        if not isinstance(x,DeepBacsImage): raise ValueError('validated DeepBacs image required')
        return cls(x.height,x.width,x.image_dtype,x.mask_dtype,x.image_bytes.hex(),x.mask_bytes.hex(),
                   x.source,x.source_version,x.image_member,x.archive_sha256,x.image_sha256,
                   x.mask_sha256,x.license,x.parent_group,x.split,x.species,x.strain,x.modality,
                   x.annotation,x.pixel_size_um,direction)

    def require_training(self):
        if self.split!='train': raise ValueError('Held-out split cannot be used for training')

    def _decode(self,raw,dtype,device):
        a=np.frombuffer(bytes.fromhex(raw),dtype=dtype).reshape(self.height,self.width)
        return torch.from_numpy(a.astype(a.dtype.newbyteorder('='),copy=True)).to(device)

    def input_image(self, *, device='cpu'):
        # This path never decodes labels or constructs target geometry.
        return self._decode(self.image_hex,self.image_dtype,device)

    def instances(self, *, device='cpu'):
        labels=self._decode(self.labels_hex,self.label_dtype,device).long()
        ids=torch.unique(labels); ids=ids[ids!=0]
        return labels[None]==ids[:,None,None]

    def _provenance(self):
        return tuple(Provenance(self.source,self.revision,self.source_id+suffix,sha,self.license)
                     for suffix,sha in (('',self.image_sha256),('#mask',self.mask_sha256),
                                        ('#archive',self.archive_sha256)))

    def _graph(self,include_cells):
        nodes=[Node('microscopy_image',tuple(_m(k,v,u) for k,v,u in (
            ('height',self.height,'pixel'),('width',self.width,'pixel'),
            ('organism',self.species,None),('strain',self.strain,None),
            ('microscopy',self.modality,None),('pixel_size_um',self.pixel_size_um,'um/pixel'),
            ('perturbation',None,None),('expression',None,None))))]
        edges=[]
        if include_cells:
            for mask in self.instances():
                yy,xx=mask.nonzero(as_tuple=True)
                nodes.append(Node('observed_whole_cell',tuple(_m(k,v,u) for k,v,u in (
                    ('area',int(mask.sum()),'pixel^2'),('x_min',int(xx.min()),'pixel'),
                    ('x_max',int(xx.max()),'pixel'),('y_min',int(yy.min()),'pixel'),
                    ('y_max',int(yy.max()),'pixel')))))
                edges.append(Edge(0,len(nodes)-1,'contains_observed_cell',True))
        return GraphRecord(self.source+':'+self.source_id,tuple(nodes),tuple(edges),self._provenance(),
                           (LeakageKey('raster:parent-v1',self.parent_group),
                            LeakageKey('raster:image-sha256-v1',self.image_sha256)))

    def input_graph(self):
        return self._graph(self.direction=='mask_to_image')

    def target_graph(self):
        return self._graph(self.direction=='image_to_mask')

    def project(self, *, device='cpu',for_training=False):
        if for_training: self.require_training()
        image={'image':self.input_image(device=device)}
        masks={'instances':self.instances(device=device)}
        inputs,targets=(image,masks) if self.direction=='image_to_mask' else (masks,image)
        return dict(inputs=inputs,targets=targets,input_graph=self.input_graph(),target_graph=self.target_graph())

    def canonical_bytes(self):
        return _canonical(asdict(self))

    @classmethod
    def from_bytes(cls,raw):
        def unique(pairs):
            d={}
            for k,v in pairs:
                if k in d: raise ValueError('duplicate snapshot key')
                d[k]=v
            return d
        try:
            d=json.loads(raw,object_pairs_hook=unique)
            if not isinstance(d,dict) or set(d)!={f.name for f in fields(cls)}:
                raise ValueError('explicit complete task snapshot required')
            return cls(**d)
        except (TypeError,KeyError,UnicodeError) as exc:
            raise ValueError('malformed morphology snapshot') from exc

    def assert_resume_matches(self,other):
        if not isinstance(other,RasterMorphologyTask) or self.canonical_bytes()!=other.canonical_bytes():
            raise ValueError('resume source/task mismatch')
