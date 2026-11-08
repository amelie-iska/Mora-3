"""Bounded full-image reader for Zenodo 5550933 v1 S.aureus.

No extraction, patches, inferred response pairing, resizing or relabeling.
Archive SHA must be supplied from an independently reviewed acquisition receipt.
Parent groups deliberately omit split, so duplicate identities cannot hide across
splits. Cross-source overlap auditing remains a downstream eligibility gate.
"""
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

SOURCE = 'https://zenodo.org/records/5550933'
PINNED_SHA256 = '4047792f1248ee82fce34121d0ade84828e55db5a34656cc25beec46eacaf307'
MAX_MEMBER_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class DeepBacsImage:
    split: str
    image_member: str
    mask_member: str
    archive_sha256: str
    image_sha256: str
    mask_sha256: str
    height: int
    width: int
    image_dtype: str
    mask_dtype: str
    image_bytes: bytes
    mask_bytes: bytes
    instance_ids: tuple
    parent_group: str
    source: str = SOURCE
    source_version: str = 'v1'
    license: str = 'CC-BY-4.0'
    species: str = 'Staphylococcus aureus'
    strain: str = 'JE2'
    modality: str = 'DIC/brightfield archive channel'
    annotation: str = 'manual ellipses drawn on Nile Red fluorescence; whole-cell instance labels'
    pixel_size_um: float = .08

    def project(self, *, for_training=False):
        """Image-only conditioning and aligned integer instance-label targets.

        Original arbitrary IDs have no semantic identity. Native-endian copies
        permit later Torch conversion without mutating source storage.
        """
        if for_training and self.split != 'train':
            raise ValueError('Held-out split cannot be used for training')
        def decode(payload, dtype):
            array=np.frombuffer(payload,dtype=dtype).reshape(self.height,self.width)
            return array.astype(array.dtype.newbyteorder('='),copy=True)
        return {'inputs': {'image': decode(self.image_bytes,self.image_dtype)},
                'targets': {'labels': decode(self.mask_bytes,self.mask_dtype)}}


def read_full_images(path, *, expected_sha256=PINNED_SHA256, limit_per_split=1):
    """Read at most a bounded number of full images in each official split.

    Inspect the archive directory but decode selected pairs only. Ignore train
    patches and the other imaging channel; they share biological parents.
    This is an adapter, not a completed leakage audit or training eligibility report.
    """
    if type(limit_per_split) is not int or not 1 <= limit_per_split <= 12:
        raise ValueError('Bounded limit_per_split must be 1..12')
    path=Path(path)
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
    if digest.hexdigest()!=expected_sha256:
        raise ValueError('Archive checksum mismatch')
    rows=[]
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)): raise ValueError('Duplicate archive member')
        for split,folder in [('train','train/full_images'),('test','test')]:
            prefix=f'brightfield_dataset/{folder}/'
            images=sorted(n for n in names if n.startswith(prefix+'brightfield/') and n.endswith('.tif'))
            masks={n for n in names if n.startswith(prefix+'masks/') and n.endswith('.tif')}
            expected={prefix+'masks/'+n.rsplit('/',1)[1] for n in images}
            if not images or expected!=masks: raise ValueError('Missing or unmatched image/mask pair')
            for image_member in images[:limit_per_split]:
                mask_member=prefix+'masks/'+image_member.rsplit('/',1)[1]
                raw=[]; arrays=[]
                for member in (image_member,mask_member):
                    if z.getinfo(member).file_size>MAX_MEMBER_BYTES:
                        raise ValueError('Member exceeds bounded decode budget')
                    payload=z.read(member)
                    with Image.open(io.BytesIO(payload)) as im:
                        if im.width*im.height>512*512 or getattr(im,'n_frames',1)!=1:
                            raise ValueError('Only bounded single-plane images supported')
                        array=np.array(im)
                    if array.ndim!=2 or array.dtype.kind!='u' or array.dtype.itemsize>2:
                        raise ValueError('Unsigned 2D image/instance mask required')
                    raw.append(payload); arrays.append(array)
                image,mask=arrays
                if image.shape!=mask.shape: raise ValueError('Image and mask must be aligned')
                ids=tuple(int(i) for i in np.unique(mask) if i!=0)
                rows.append(DeepBacsImage(split,image_member,mask_member,expected_sha256,
                    hashlib.sha256(raw[0]).hexdigest(),hashlib.sha256(raw[1]).hexdigest(),
                    *image.shape,image.dtype.str,mask.dtype.str,image.tobytes(),mask.tobytes(),ids,
                    'deepbacs:5550933:'+Path(image_member).stem))
    return rows
