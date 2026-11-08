"""Ordered numeric array task contract, separate from scalar GraphRecord v1.

Projection is lossless FP64 by default, not a learned encoder or eligibility grant.
Axes are retained in full. No evaluation fields or hidden target metadata enter
input_metadata. Source-specific context/provenance stays in the owning manifest.
"""
from dataclasses import dataclass, asdict
import json
import math
import re
import torch


def _text(x):
    return isinstance(x, str) and bool(x.strip())


def _bytes(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


@dataclass(frozen=True)
class OrderedArray:
    name: str
    axis_ids: tuple
    values: tuple | None
    axis_identity: str
    units: str
    status: str

    def __post_init__(self):
        axis=tuple(self.axis_ids)
        if not all(_text(x) for x in (self.name,self.axis_identity,self.units)) or not axis or not all(_text(x) for x in axis):
            raise ValueError('nonempty array name, ordered axis, identity and units required')
        if self.status not in ('observed','source_predicted','model_generated','unavailable'):
            raise ValueError('unknown array status')
        values=None if self.values is None else tuple(self.values)
        if self.status == 'unavailable':
            if values is not None:
                raise ValueError('unavailable values must be absent')
        elif values is None or len(values)!=len(axis) or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
            raise ValueError('finite numeric array must match axis')
        object.__setattr__(self,'axis_ids',axis)
        object.__setattr__(self,'values',values)

    def metadata(self):
        return {k:v for k,v in asdict(self).items() if k!='values'}


@dataclass(frozen=True)
class ArrayTask:
    arrays: tuple
    roles: tuple
    source_sha256: str
    version: int = 1

    def __post_init__(self):
        arrays=tuple(self.arrays)
        roles=tuple((name,role) for name,role in self.roles)
        if type(self.version) is not int or self.version!=1 or not isinstance(self.source_sha256,str) or not re.fullmatch('[0-9a-f]{64}',self.source_sha256):
            raise ValueError('version and source SHA256 required')
        if not arrays or any(not isinstance(a,OrderedArray) for a in arrays):
            raise ValueError('validated arrays required')
        names=[a.name for a in arrays]
        if len(set(names))!=len(names) or len(dict(roles))!=len(roles) or set(dict(roles))!=set(names):
            raise ValueError('each array requires exactly one role')
        if any(r not in ('condition','desired','target','evaluation','unused') for _,r in roles):
            raise ValueError('unknown role')
        object.__setattr__(self,'arrays',tuple(sorted(arrays,key=lambda a:a.name)))
        object.__setattr__(self,'roles',tuple(sorted(roles)))

    def input_metadata(self):
        roles=dict(self.roles)
        # Deliberately exclude source hash: source artifacts can contain targets.
        return {'version':self.version,'arrays':[dict(a.metadata(),role=roles[a.name])
                for a in self.arrays if roles[a.name] in ('condition','desired')]}

    def project(self, *, device='cpu', dtype=torch.float64):
        if dtype not in (torch.float32,torch.float64):
            raise ValueError('only explicit FP32/FP64 projection supported')
        result=dict(inputs={},targets={},target_masks={})
        for a in self.arrays:
            role=dict(self.roles)[a.name]
            if role in ('evaluation','unused'):
                continue
            if role in ('condition','desired') and a.values is None:
                raise ValueError('unavailable input')
            if role=='target':
                if a.status not in ('observed','unavailable'):
                    raise ValueError('only observed supervision allowed')
                result['target_masks'][a.name]=torch.full((len(a.axis_ids),),a.values is not None,dtype=torch.bool,device=device)
            if a.values is not None:
                value=torch.tensor(a.values,dtype=dtype,device=device)
                if not torch.isfinite(value).all():
                    raise ValueError('projection overflow')
                result['targets' if role=='target' else 'inputs'][a.name]=value
        return result

    def canonical_bytes(self):
        return _bytes(asdict(self))

    @classmethod
    def from_bytes(cls, raw):
        def unique(pairs):
            d={}
            for k,v in pairs:
                if k in d: raise ValueError('duplicate JSON key')
                d[k]=v
            return d
        try:
            d=json.loads(raw,object_pairs_hook=unique)
            if set(d)!={'arrays','roles','source_sha256','version'}:
                raise ValueError('invalid task fields')
            d['arrays']=tuple(OrderedArray(**a) for a in d['arrays'])
            return cls(**d)
        except (TypeError,KeyError,UnicodeError) as exc:
            raise ValueError('malformed array task') from exc
