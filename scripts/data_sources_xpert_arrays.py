"""Streaming numeric NPY transport preserving native dtype/endian and validity bits.

No HDF5 import or source execution. These helpers do not enable model use.
"""
from __future__ import annotations
import hashlib,math,mmap,os,stat
from pathlib import Path
import numpy as np

def file_identity(path):
    s=os.stat(path,follow_symlinks=False)
    if not stat.S_ISREG(s.st_mode):raise ValueError("numeric regular file identity required")
    return {"dev":s.st_dev,"ino":s.st_ino,"size":s.st_size,"mtime_ns":s.st_mtime_ns,"ctime_ns":s.st_ctime_ns}

def header_sha(path,size):
    with Path(path).open("rb") as f:return hashlib.sha256(f.read(size)).hexdigest()

def _chunk_record(c,numeric_offset,body,validity_offset,bits):
    return {"axis0_start":c["axis0_start"],"axis0_stop":c["axis0_stop"],"native_elements":c["native_elements"],
            "numeric_offset":numeric_offset,"numeric_bytes":len(body),"numeric_sha256":hashlib.sha256(body).hexdigest(),
            "validity_offset":validity_offset,"validity_bytes":len(bits),"validity_sha256":hashlib.sha256(bits).hexdigest()}

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def validate_expected(e):
    dtype=np.dtype(e["native_dtype"]);shape=tuple(e["native_shape"])
    if dtype.hasobject or dtype.fields is not None or dtype.kind not in "iufb":
        raise ValueError("only plain native numeric dtype allowed; object forbidden")
    if len(shape) not in (1,2) or any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in shape):
        raise ValueError("bounded native 1D/2D numeric shape required")
    blocks=e["chunks"];start=0;trailing=math.prod(shape[1:]);packed=0;largest=0
    for c in blocks:
        if (c["axis0_start"]!=start or not isinstance(c["axis0_stop"],int)
            or not start<c["axis0_stop"]<=shape[0]
            or c["native_elements"]!=(c["axis0_stop"]-start)*trailing):
            raise ValueError("original chunk partition gap/duplicate/shape mismatch")
        if c["native_elements"]*dtype.itemsize>32*1024*1024:
            raise ValueError("native source block exceeds bounded 32MiB transport")
        packed+=(c["native_elements"]+7)//8;largest=max(largest,c["native_elements"]);start=c["axis0_stop"]
    if start!=shape[0]:raise ValueError("chunk partition does not cover full native axis")
    for name in ["native_value_SHA256","native_validity_SHA256"]:
        value=e[name]
        if not isinstance(value,str) or len(value)!=64 or any(x not in "0123456789abcdef" for x in value):
            raise ValueError("pinned native expected digest required")
    return dtype,shape,packed,largest

def _write_header(f,shape,dtype):
    np.lib.format.write_array_header_1_0(f,{"descr":np.lib.format.dtype_to_descr(np.dtype(dtype)),"fortran_order":False,"shape":tuple(shape)})
    return f.tell()

def _check_native_hashes(value,validity,e):
    if value!=e["native_value_SHA256"]:raise ValueError("native value SHA digest mismatch; retain unpromoted payload")
    if validity!=e["native_validity_SHA256"]:raise ValueError("native validity SHA digest mismatch; retain unpromoted payload")

def write_numeric_stream(path,validity_path,e,read_block,checkpoint=None):
    dtype,shape,packed,_=validate_expected(e);p=Path(path);v=Path(validity_path)
    if p==v or p.exists() or p.is_symlink() or v.exists() or v.is_symlink():raise FileExistsError("fresh distinct numeric output paths required")
    hv=hashlib.sha256();hm=hashlib.sha256();elements=0;chunks=[]
    with p.open("xb") as fout,v.open("xb") as mask:
        offset=_write_header(fout,shape,dtype);moffset=_write_header(mask,(packed,),"u1")
        for c in e["chunks"]:
            if checkpoint:checkpoint()
            start,stop=c["axis0_start"],c["axis0_stop"]
            a=np.asarray(read_block(start,stop))
            if a.dtype.str!=dtype.str or a.shape!=(stop-start,)+shape[1:]:
                raise ValueError("source block dtype/shape changed; no casting or repair")
            body=a.tobytes(order="C")
            validity=np.packbits(np.isfinite(a).reshape(-1),bitorder="little").tobytes()
            chunks.append(_chunk_record(c,fout.tell(),body,mask.tell(),validity))
            fout.write(body);mask.write(validity);hv.update(body);hm.update(validity);elements+=a.size
            del a,body,validity
        fout.flush();os.fsync(fout.fileno());mask.flush();os.fsync(mask.fileno())
    _check_native_hashes(hv.hexdigest(),hm.hexdigest(),e)
    if p.stat().st_size!=offset+math.prod(shape)*dtype.itemsize or v.stat().st_size!=moffset+packed:
        raise ValueError("numeric NPY expected size mismatch")
    return {"path":str(p),"bytes":p.stat().st_size,"sha256":_sha(p),"native_data_offset":offset,
            "validity_path":str(v),"validity_bytes":v.stat().st_size,"validity_sha256":_sha(v),"validity_data_offset":moffset,
            "native_value_sha256":hv.hexdigest(),"native_validity_sha256":hm.hexdigest(),"all_native_elements":elements,\
            "numeric_header_sha256":header_sha(p,offset),"validity_header_sha256":header_sha(v,moffset),\
            "numeric_file_identity":file_identity(p),"validity_file_identity":file_identity(v),"chunks":chunks}

def _header(f,expected_shape,expected_dtype):
    if np.lib.format.read_magic(f)!=(1,0):raise ValueError("exact numeric NPY v1 header required")
    shape,fortran,dtype=np.lib.format.read_array_header_1_0(f,max_header_size=65536)
    if tuple(shape)!=tuple(expected_shape) or fortran or dtype.hasobject or dtype.str!=np.dtype(expected_dtype).str:
        raise ValueError("numeric NPY header shape/dtype/order mismatch")
    offset=f.tell()
    if os.fstat(f.fileno()).st_size!=offset+math.prod(shape)*dtype.itemsize:
        raise ValueError("numeric NPY file length/header mismatch")
    return offset

def verify_numeric_pair(path,validity_path,e,checkpoint=None,transport_manifest=None):
    dtype,shape,packed,largest=validate_expected(e)
    identity=file_identity(path);mid=file_identity(validity_path)
    if transport_manifest is not None:
        if identity!=transport_manifest["numeric_file_identity"] or mid!=transport_manifest["validity_file_identity"]:
            raise ValueError("numeric verified file stat identity mismatch")
    with Path(path).open("rb") as f:
        data_offset=_header(f,shape,dtype);f.seek(0);numeric_header=f.read(data_offset)
    hv=hashlib.sha256();hm=hashlib.sha256();elements=0;releases=0;chunks=[]
    numeric_file_hash=hashlib.sha256(numeric_header)
    with Path(validity_path).open("rb") as mask:
        mask_offset=_header(mask,(packed,),"u1");mask.seek(0);mask_header=mask.read(mask_offset)
        validity_file_hash=hashlib.sha256(mask_header)
        if transport_manifest is not None:
            if hashlib.sha256(numeric_header).hexdigest()!=transport_manifest["numeric_header_sha256"] or hashlib.sha256(mask_header).hexdigest()!=transport_manifest["validity_header_sha256"]:
                raise ValueError("numeric NPY header SHA mismatch")
        a=np.load(path,mmap_mode="r",allow_pickle=False,max_header_size=65536)
        try:
            if file_identity(path)!=identity:raise ValueError("numeric file stat identity changed before mmap read")
            if not hasattr(a,"_mmap") or not hasattr(a._mmap,"madvise") or not hasattr(mmap,"MADV_DONTNEED"):
                raise ValueError("bounded resident mmap page release unavailable")
            for c in e["chunks"]:
                if checkpoint:checkpoint()
                block=a[c["axis0_start"]:c["axis0_stop"]]
                body=block.tobytes(order="C");bits=np.packbits(np.isfinite(block).reshape(-1),bitorder="little").tobytes()
                bit_offset=mask.tell();recorded=mask.read(len(bits))
                if bits!=recorded:raise ValueError("numeric finite-validity bit payload mismatch")
                chunks.append(_chunk_record(c,data_offset+c["axis0_start"]*math.prod(shape[1:])*dtype.itemsize,body,bit_offset,recorded))
                hv.update(body);hm.update(recorded);numeric_file_hash.update(body);validity_file_hash.update(recorded);elements+=block.size
                del block,body,bits,recorded
                a._mmap.madvise(mmap.MADV_DONTNEED);releases+=1
            if mask.read(1):raise ValueError("trailing validity payload bytes")
        finally:a._mmap.close()
    if file_identity(path)!=identity or file_identity(validity_path)!=mid:raise ValueError("numeric file identity changed during verification")
    _check_native_hashes(hv.hexdigest(),hm.hexdigest(),e)
    if transport_manifest is not None:
        if chunks!=transport_manifest["chunks"]:raise ValueError("numeric per-chunk transport proof mismatch")
        if numeric_file_hash.hexdigest()!=transport_manifest["sha256"] or validity_file_hash.hexdigest()!=transport_manifest["validity_sha256"]:
            raise ValueError("numeric full transport artifact SHA mismatch")
    return {"native_value_sha256":hv.hexdigest(),"native_validity_sha256":hm.hexdigest(),
            "numeric_file_sha256":numeric_file_hash.hexdigest(),"validity_file_sha256":validity_file_hash.hexdigest(),
            "numeric_header_sha256":hashlib.sha256(numeric_header).hexdigest(),"validity_header_sha256":hashlib.sha256(mask_header).hexdigest(),
            "native_data_offset":data_offset,"validity_data_offset":mask_offset,"chunks":chunks,
            "numeric_file_identity":identity,"validity_file_identity":mid,
            "all_native_elements":elements,"mmap_page_release_calls":releases,"largest_readback_block_elements":largest,
            "native_dtype":dtype.str,"native_shape":list(shape),"allow_pickle":False,"source_h5py_runtime_dependency":False}
