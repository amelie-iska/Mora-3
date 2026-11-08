"""CSEG-01 metadata contract; routing does not execute or certify a native head.

No pixel loading, source joins or training admission occurs here. Missing context
is explicit (None values); links require exact parent/context/frame agreement.
Only declared condition/desired artifacts are exposed as model inputs.
"""
from dataclasses import asdict, dataclass, fields
import json
import math


_EVIDENCE = {
    'image': {'source_observed', 'model_predicted', 'model_generated', 'desired'},
    'profile': {'source_observed', 'model_predicted', 'desired'},
    'mask': {'manual_annotation', 'source_annotation_unverified', 'source_algorithmic_mask', 'predicted_mask',
             'pseudo_label_mask', 'desired'},
    'objects': {'source_computed', 'model_predicted', 'desired'},
}
_MODEL = {'model_predicted', 'model_generated', 'predicted_mask', 'pseudo_label_mask'}


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('nonempty text required')


@dataclass(frozen=True)
class SegmentationArtifact:
    artifact_id: str
    kind: str
    evidence: str
    role: str
    source_id: str
    parent_id: str
    context: tuple
    asset: str
    representation: str
    channels: tuple = ()
    spatial_shape: tuple = ()
    links: tuple = ()
    reference_frame: str = 'source_pixel_yx'
    pixel_size_um: float | None = None
    checkpoint: str | None = None
    seed: int | None = None
    uncertainty: str | None = None
    quality_flags: tuple = ()

    def __post_init__(self):
        for key in ('artifact_id','source_id','parent_id','asset','representation','reference_frame'):
            _text(getattr(self,key))
        if self.kind not in _EVIDENCE or self.evidence not in _EVIDENCE[self.kind]:
            raise ValueError('incompatible artifact/evidence class')
        if self.role not in ('condition','desired','target','evaluation','unused'):
            raise ValueError('unknown task role')
        if (self.role == 'desired') != (self.evidence == 'desired'):
            raise ValueError('desired evidence and role must agree')
        if self.evidence in _MODEL and self.checkpoint is None:
            raise ValueError('model-derived artifact requires checkpoint/method version')
        for key in ('checkpoint','uncertainty'):
            if getattr(self,key) is not None: _text(getattr(self,key))
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError('nonnegative integer seed required')
        for key in ('channels','spatial_shape','quality_flags'):
            object.__setattr__(self,key,tuple(getattr(self,key)))
        for key in ('context','links'):
            pairs=tuple(tuple(p) for p in getattr(self,key))
            if any(len(p)!=2 for p in pairs): raise ValueError('named pairs required')
            object.__setattr__(self,key,pairs)
        if len(dict(self.context)) != len(self.context): raise ValueError('duplicate context')
        for name,value in self.context:
            _text(name)
            if value is not None: _text(value)
        for name,target in self.links:
            if name not in ('mask_of','objects_of'): raise ValueError('unsupported typed link')
            _text(target)
        for value in self.channels + self.quality_flags: _text(value)
        if len(set(self.channels)) != len(self.channels): raise ValueError('duplicate channel')
        if self.kind == 'profile':
            if self.channels or self.spatial_shape: raise ValueError('profile is not a raster')
        elif len(self.spatial_shape)!=2 or any(type(n) is not int or n<=0 for n in self.spatial_shape):
            raise ValueError('positive two dimensional pixel shape required')
        if self.kind=='image' and not self.channels: raise ValueError('explicit image channels required')
        if self.kind in ('mask','objects'):
            relation='mask_of' if self.kind=='mask' else 'objects_of'
            if len(self.links)!=1 or self.links[0][0]!=relation:
                raise ValueError('one typed parent image link required')
        elif self.links: raise ValueError('image/profile cannot contain derived links')
        scale=self.pixel_size_um
        if scale is not None and (type(scale) not in (float,int) or not math.isfinite(scale) or scale<=0):
            raise ValueError('positive finite physical scale required')


@dataclass(frozen=True)
class SegmentationRequest:
    artifacts: tuple
    supported_channels: tuple
    supported_objects: tuple
    object_type: str
    head_checkpoint: str | None
    version: int = 1

    def __post_init__(self):
        if type(self.version) is not int or self.version!=1: raise ValueError('unknown request version')
        object.__setattr__(self,'artifacts',tuple(self.artifacts))
        object.__setattr__(self,'supported_channels',tuple(tuple(c) for c in self.supported_channels))
        object.__setattr__(self,'supported_objects',tuple(self.supported_objects))
        _text(self.object_type)
        for obj in self.supported_objects: _text(obj)
        for channels in self.supported_channels:
            if not channels: raise ValueError('empty supported channel signature')
            for channel in channels: _text(channel)
        if self.head_checkpoint is not None: _text(self.head_checkpoint)
        if any(not isinstance(a,SegmentationArtifact) for a in self.artifacts):
            raise ValueError('validated artifacts required')
        indexed={a.artifact_id:a for a in self.artifacts}
        if len(indexed)!=len(self.artifacts): raise ValueError('duplicate artifact ID')
        for a in self.artifacts:
            for relation,target in a.links:
                parent=indexed.get(target)
                if parent is None or parent.kind!='image': raise ValueError('image link unavailable')
                if any(getattr(a,k)!=getattr(parent,k) for k in
                       ('source_id','parent_id','context','spatial_shape','reference_frame','pixel_size_um')):
                    raise ValueError('unaligned or incompatible image link')
                if a.evidence in ('manual_annotation','source_annotation_unverified','source_algorithmic_mask','source_computed') and parent.evidence!='source_observed':
                    raise ValueError('source evidence cannot annotate model/desired imagery')

    def condition_ids(self):
        return tuple(a.artifact_id for a in self.artifacts if a.role in ('condition','desired'))

    def supervision_ids(self, *, allow_source_algorithmic=False):
        # Explicit source-algorithm supervision remains a separate opt-in scope.
        evidence={'manual_annotation'}
        if allow_source_algorithmic: evidence.add('source_algorithmic_mask')
        return tuple(a.artifact_id for a in self.artifacts
                     if a.kind=='mask' and a.role=='target' and a.evidence in evidence)

    def route(self):
        result=[]
        for a in self.artifacts:
            if a.kind not in ('image','profile'): continue
            if a.kind!='image': reason='no_image'
            elif a.role not in ('condition','desired'): reason='held_out_target'
            elif self.head_checkpoint is None: reason='missing_native_head'
            elif self.object_type not in self.supported_objects: reason='unsupported_object'
            elif a.channels not in self.supported_channels: reason='unsupported_channels'
            else: reason=None
            result.append((a.artifact_id,'unavailable:'+reason if reason else 'native_segmentation'))
        return tuple(result)

    def canonical_bytes(self):
        return json.dumps(asdict(self),sort_keys=True,separators=(',',':'),allow_nan=False).encode()

    @classmethod
    def from_bytes(cls, raw):
        def unique(pairs):
            d={}
            for key,value in pairs:
                if key in d: raise ValueError('duplicate snapshot key')
                d[key]=value
            return d
        try:
            d=json.loads(raw,object_pairs_hook=unique)
            if not isinstance(d,dict) or set(d)!={f.name for f in fields(cls)}:
                raise ValueError('complete versioned request required')
            d['artifacts']=tuple(SegmentationArtifact(**a) for a in d['artifacts'])
            return cls(**d)
        except (TypeError,KeyError,UnicodeError) as exc:
            raise ValueError('malformed segmentation snapshot') from exc
