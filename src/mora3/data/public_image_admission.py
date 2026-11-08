"""Pinned external admission for public, whole-cell instance supervision.

The corpus owner supplies a versioned sidecar after rights/lineage/leakage review.
A digest authenticates the supplied bytes, not the truth of those reviews. This
module never writes admission, infers rights, or promotes a held-out source.
Existing intake/format-only manifests are deliberately not compatible receipts.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .public_image_masks import PublicImageMasks


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError('duplicate admission key')
        result[key]=value
    return result


@dataclass(frozen=True)
class PublicImageAdmission:
    source: PublicImageMasks
    admission_bytes: bytes
    admission_sha256: str

    def __post_init__(self):
        if not isinstance(self.source,PublicImageMasks):
            raise ValueError('validated public source required')
        if type(self.admission_bytes) is not bytes or len(self.admission_bytes)>16*1024**2:
            raise ValueError('bounded immutable admission required')
        if hashlib.sha256(self.admission_bytes).hexdigest()!=self.admission_sha256:
            raise ValueError('admission checksum mismatch')
        try:
            document=json.loads(self.admission_bytes,object_pairs_hook=_unique)
            if document['schema']!='mora3.public_image_admission.v1':
                raise ValueError('unsupported corpus admission schema')
            rows=document['records']
            if not isinstance(rows,list) or not all(isinstance(r,dict) for r in rows):
                raise ValueError('explicit admission records required')
            matches=[r for r in rows if r.get('binding_sha256')==self.source.binding_sha256]
            if len(matches)!=1: raise ValueError('unique source-bound admission required')
            row=matches[0]
            expected=dict(source_id=self.source.image.source_id,parent_id=self.source.image.parent_id,
                          split=self.source.split,encoding=self.source.encoding)
            if any(row.get(k)!=v for k,v in expected.items()):
                raise ValueError('admission source/split/parent/encoding mismatch')
            if self.source.split!='train': raise ValueError('held-out source split cannot train')
            if row.get('training_eligible') is not True:
                raise ValueError('explicit corpus training admission required')
            for key in ('rights_review','lineage_review','leakage_group'):
                if not isinstance(row.get(key),str) or not row[key].strip():
                    raise ValueError('explicit corpus rights/lineage/leakage review required')
            if not isinstance(row.get('allowed_objectives'),list) or 'whole-cell-instance-v1' not in row['allowed_objectives']:
                raise ValueError('whole-cell instance objective admission required')
        except (KeyError,TypeError,UnicodeError,json.JSONDecodeError) as exc:
            raise ValueError('malformed corpus admission') from exc
        if self.source.encoding!='instance_id_map':
            raise ValueError('native whole-cell instance objective requires instance-ID map')
        self.source.native_input()  # Fail closed on incompatible channels/size; no conversion.

    @classmethod
    def from_file(cls, source, path, *, sha256):
        with Path(path).open('rb') as stream:
            raw=stream.read(16*1024**2+1)
        return cls(source,raw,sha256)

    def require_training(self):
        # Immutable sources/receipt were authenticated during construction.
        return None

    def native_input(self):
        return self.source.native_input()

    def instances(self, *, device='cpu'):
        labels=next(self.source.iter_target_arrays())
        ids=np.unique(labels); ids=ids[ids!=0]
        return torch.from_numpy(labels[None]==ids[:,None,None]).to(device)

    def canonical_bytes(self):
        return _canonical(dict(schema='mora3.public_image_supervision.v1',
            source_sha256=self.source.binding_sha256,admission_sha256=self.admission_sha256,
            direction='image_to_mask',objective='whole-cell-instance-v1'))

    @property
    def binding_sha256(self):
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def assert_resume_matches(self,other):
        if not isinstance(other,PublicImageAdmission) or self.canonical_bytes()!=other.canonical_bytes():
            raise ValueError('resume source/admission/objective mismatch')
