"""Native TIFF pixel QC using installed Pillow/libtiff; no transformed copies."""
import hashlib,json,numbers,pathlib
import numpy as np
from PIL import Image,TiffTags

def tag_value(v):
    if isinstance(v,(str,int,float,bool)) or v is None:return v
    if isinstance(v,bytes):return dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest(),utf8=v.decode('utf8') if len(v)<65536 else None)
    if isinstance(v,(tuple,list)):return [tag_value(x) for x in v]
    if hasattr(v,'numerator') and hasattr(v,'denominator'):return dict(numerator=int(v.numerator),denominator=int(v.denominator))
    return str(v)
def decode_qc(path,expected_shape=(2160,2160)):
    path=pathlib.Path(path)
    with Image.open(path) as image:
        if image.format!='TIFF' or image.n_frames!=1:raise ValueError('exact one native TIFF plane required; no page selection')
        tags={str(k):dict(name=TiffTags.TAGS.get(k,str(k)),value=tag_value(v)) for k,v in image.tag_v2.items()};mode=image.mode;image.load();a=np.asarray(image)
        if a.shape!=expected_shape or a.ndim!=2 or a.dtype.kind not in 'uif':raise ValueError('native full-plane shape/dtype drift; no crop/downcast')
        dtype=a.dtype;valid=np.isfinite(a);canonical=a.astype(dtype.newbyteorder('<'),copy=False);pixel_sha=hashlib.sha256(canonical.tobytes(order='C')).hexdigest();valid_sha=hashlib.sha256(valid.astype(np.uint8).tobytes(order='C')).hexdigest();n=int(a.size);minimum=a[valid].min().item() if valid.any() else None;maximum=a[valid].max().item() if valid.any() else None;zero=int((a==0).sum());saturation=int((a==np.iinfo(dtype).max).sum()) if dtype.kind in 'ui' else None
    with Image.open(path) as second:
        second.load();b=np.asarray(second)
        if b.shape!=expected_shape or b.dtype!=dtype or hashlib.sha256(b.astype(dtype.newbyteorder('<'),copy=False).tobytes(order='C')).hexdigest()!=pixel_sha:raise ValueError('independent full native pixel decode mismatch')
    return dict(format='TIFF',frames=1,mode=mode,dtype=str(dtype),dtype_str=dtype.str,shape=list(expected_shape),axes='YX',pixels=n,native_pixel_bytes=n*dtype.itemsize,pixels_sha256_little_endian_row_major=pixel_sha,finite_validity_mask_sha256_uint8_row_major=valid_sha,finite_pixels=int(valid.sum()),nonfinite_pixels=n-int(valid.sum()),minimum=minimum,maximum=maximum,zero_pixels=zero,integer_dtype_max_pixels=saturation,tags=tags,independent_full_decode_equal=True,no_crop_downcast_normalization_or_derived_pixel_copy=True)
