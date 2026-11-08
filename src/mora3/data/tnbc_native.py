"""Read bounded TNBC v1 pairs as nuclear *semantic* evaluation candidates.

Zenodo 1175282 supplies no source train/test assignment in this archive. Folder
IDs are retained as conservative slide groups, not certified patient identities.
This format adapter grants no training admission and makes no response joins.
RGB/RGBA input remains unsupported by the current brightfield whole-cell native head.
"""
import hashlib
import io
import re
import zipfile

from PIL import Image

from .cooperative_segmentation import SegmentationArtifact
from .imaging import archive_members, file_sha256
from .public_image_masks import PublicImageMasks


_IMAGE = re.compile(r'TNBC_NucleiSegmentation/Slide_(0[1-9]|1[01])/\1_([1-9][0-9]*)\.png')
_SOURCE = 'zenodo:1175282:v1.0'


def load_tnbc_pairs(path, *, image_members, archive_sha256):
    """Authenticate an immutable archive, then load at most 50 explicit pairs.

    Member matching is exact within the same source slide folder. Masks stay
    binary; disconnected regions never become inferred instance annotations.
    Returned image artifacts contain no mask/foreground/count/archive digest.
    The caller retains the checked archive digest as audit-only provenance.
    """
    if not isinstance(archive_sha256,str) or re.fullmatch('[0-9a-f]{64}',archive_sha256) is None:
        raise ValueError('explicit archive checksum required')
    if isinstance(image_members,(str,bytes)) or not isinstance(image_members,(tuple,list)):
        raise ValueError('bounded explicit image member list required')
    members=tuple(image_members)
    if not 1<=len(members)<=50 or any(not isinstance(n,str) for n in members):
        raise ValueError('one to fifty explicit source members required')
    if len(set(members))!=len(members):
        raise ValueError('duplicate source member request')
    matches=[_IMAGE.fullmatch(n) for n in members]
    if any(m is None for m in matches):
        raise ValueError('source slide and filename must agree')
    if file_sha256(path)!=archive_sha256:
        raise ValueError('archive checksum mismatch')
    inventory={m['name']:m for m in archive_members(path,max_uncompressed_bytes=64*1024**2)}
    result=[]
    with zipfile.ZipFile(path) as archive:
        for member,match in zip(members,matches):
            group,index=match.groups()
            mask_member=f'TNBC_NucleiSegmentation/GT_{group}/{group}_{index}.png'
            for name in (member,mask_member):
                info=inventory.get(name)
                if info is None or info['directory'] or info['bytes']>16*1024**2:
                    raise ValueError('missing or oversized source pair')
            raw=archive.read(member); mask=archive.read(mask_member)
            with Image.open(io.BytesIO(raw)) as image:
                if image.mode not in ('RGB','RGBA'):
                    raise ValueError('source RGB/RGBA image required; conversion forbidden')
                shape=(image.height,image.width)
                channels=tuple({'R':'red','G':'green','B':'blue','A':'alpha'}[b] for b in image.getbands())
            artifact=SegmentationArtifact(
                artifact_id=_SOURCE+':'+member,kind='image',evidence='source_observed',
                role='condition',source_id=_SOURCE,parent_id=_SOURCE+':slide-group:'+group,
                context=(('object_type','nucleus'),('source_slide_group',group),
                         ('source_split',None),('patient_identity',None),
                         ('perturbation',None),('expression',None),('physical_scale',None)),
                asset='https://zenodo.org/records/1175282#'+member,
                representation='source-yx-no-transform-v1',channels=channels,
                spatial_shape=shape)
            result.append(PublicImageMasks.from_encoded(
                artifact,raw,(mask,),encoding='semantic_binary',split='evaluation',
                image_sha256=hashlib.sha256(raw).hexdigest(),
                mask_sha256=(hashlib.sha256(mask).hexdigest(),)))
    return tuple(result)
