"""Pinned MoNuSeg mirror RGB/ordered-mask candidates; no training admission.

Preserves native source order and overlap, with no image transforms or inferred
response joins. Patient IDs are source strings, not certified donor identities.
The mirror's NC-SA terms and cross-source exposure require corpus-owner review.
"""
import hashlib
import io

from PIL import Image

from .cooperative_segmentation import SegmentationArtifact
from .imaging import file_sha256
from .public_image_masks import PublicImageMasks

_REVISION='4812ea499ff8f6fad2df7af0c9aa459773e22459'
_SOURCES={
    'test':'6fdf47f191bb4fa6b11b347423a168f24087781f109f0fd418b2dfa8e922e1e2',
    'train':'025c1a8b47e2f899daede2f482bde0a2ef85e905cdef75bd58c7c8609c369c94',
}
_SOURCE='hf:RationAI/MoNuSeg:'+_REVISION


def load_monuseg_pairs(path, *, split, row_indices):
    """Load one or two explicit rows from an exact pinned source file.

    Full file authentication occurs on each call. Only image provenance enters
    the condition artifact; ordered mask bytes/hashes remain target-side.
    No source file path, mask count/hash or mask-derived measurement enters the
    visible-image graph. Return order follows the requested row order.
    """
    import pyarrow.parquet as pq
    if split not in _SOURCES:
        raise ValueError('pinned source split required')
    if not isinstance(row_indices,(tuple,list)) or not 1<=len(row_indices)<=2:
        raise ValueError('one or two explicit row indices required')
    indices=tuple(row_indices)
    if any(type(i) is not int or i<0 for i in indices) or len(set(indices))!=len(indices):
        raise ValueError('unique nonnegative integer row indices required')
    if file_sha256(path)!=_SOURCES[split]:
        raise ValueError('source checksum mismatch')
    parquet=pq.ParquetFile(path)
    if max(indices)>=parquet.metadata.num_rows:
        raise ValueError('requested row absent')
    selected={}
    for index,batch in enumerate(parquet.iter_batches(batch_size=1)):
        if index not in indices:
            continue
        row=batch.to_pylist()[0]
        raw=row['image']['bytes']
        if type(raw) is not bytes:
            raise ValueError('embedded source image required')
        with Image.open(io.BytesIO(raw)) as image:
            if image.mode!='RGB' or image.width*image.height>16_000_000:
                raise ValueError('bounded native RGB image required')
            shape=(image.height,image.width)
        patient=row['patient']
        if not isinstance(patient,str) or not patient.strip():
            raise ValueError('source patient string required')
        masks=tuple(a['bytes'] for a in row['instances'])
        artifact=SegmentationArtifact(
            artifact_id=f'{_SOURCE}:{split}:{index}',kind='image',
            evidence='source_observed',role='condition',source_id=_SOURCE,
            parent_id=f'{_SOURCE}:patient:{patient}',
            context=(('object_type','nucleus'),('patient',patient),
                     ('source_tissue_code',str(row['tissue'])),('source_split',split),
                     ('license','CC-BY-NC-SA-4.0'),('perturbation',None),
                     ('expression',None),('physical_scale',None)),
            asset=f'https://huggingface.co/datasets/RationAI/MoNuSeg/resolve/{_REVISION}/data/{split}-00000-of-00001.parquet#row={index}',
            representation='source-yx-no-transform-v1',
            channels=('red','green','blue'),spatial_shape=shape)
        selected[index]=PublicImageMasks.from_encoded(
            artifact,raw,masks,encoding='ordered_binary_instances',split=split,
            image_sha256=hashlib.sha256(raw).hexdigest(),
            mask_sha256=tuple(hashlib.sha256(m).hexdigest() for m in masks))
        if len(selected)==len(indices):
            break
    return tuple(selected[i] for i in indices)
